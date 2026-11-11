"""Medium/high fixes for the dashboard server, cloud store and hosted control plane.

Covers: workflow ``callback_url`` SSRF guard and background-task references,
workflow routes staying admin-only, cloud endpoint status codes, throttled
``last_used_at`` writes off the event loop, the JWKS negative cache and lock,
Stripe checkout/webhook ordering, and lazily created asyncio locks.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import os
import socket
import sqlite3
import sys
import threading
import time
import types
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import agent_vcr.hosted as hosted_module
import agent_vcr.server as server_module
from agent_vcr.cloud_store import CloudStore
from agent_vcr.hosted import (
    HostedAccount,
    HostedAccountStore,
    HostedAuthenticationError,
    HostedConfig,
    HostedConflictError,
    HostedControlPlane,
    HostedUnavailableError,
    HostedUser,
    StripeBilling,
    SupabaseJWTVerifier,
)
from agent_vcr.server import (
    VCRFileWatcher,
    VCRServer,
    callback_origin,
    resolve_callback_url,
)

ADMIN = "admin-secret"
ADMIN_H = {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        server_module.AUTH_TOKEN_ENV,
        server_module.ADMIN_TOKEN_ENV,
        server_module.ALLOW_UNAUTHENTICATED_ENV,
        server_module.ALLOW_PRIVATE_CALLBACKS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _server(tmp_path: Path, **kwargs: Any) -> VCRServer:
    return VCRServer(
        vcr_dir=str(tmp_path / "legacy"), data_dir=str(tmp_path / "cloud"), **kwargs
    )


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    return TestClient(_server(tmp_path, **kwargs).app, raise_server_exceptions=False)


def _fake_dns(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, list[str]]) -> None:
    def getaddrinfo(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
        if host not in mapping:
            raise socket.gaierror("unknown host")
        return [
            (socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port))
            for a in mapping[host]
        ]

    monkeypatch.setattr(server_module.socket, "getaddrinfo", getaddrinfo)


# -- 1. SSRF guard on workflow callback_url ----------------------------------


class TestCallbackValidation:
    def test_public_https_target_is_pinned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_dns(monkeypatch, {"hooks.example.com": ["93.184.216.34"]})
        target = resolve_callback_url(
            "https://hooks.example.com:8443/cb?sig=abc", allow_private=False
        )
        assert (target.scheme, target.host, target.port) == ("https", "hooks.example.com", 8443)
        assert target.path == "/cb?sig=abc"
        assert target.address == "93.184.216.34"
        assert target.origin == "https://hooks.example.com"

    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.1.2.3",
            "192.168.0.10",
            "172.16.5.4",
            "169.254.169.254",  # cloud metadata
            "100.100.100.200",  # CGNAT range (Alibaba metadata)
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd00:ec2::254",  # AWS IPv6 metadata
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
        ],
    )
    def test_internal_addresses_rejected(
        self, monkeypatch: pytest.MonkeyPatch, address: str
    ) -> None:
        _fake_dns(monkeypatch, {"evil.example.com": [address]})
        with pytest.raises(ValueError, match=server_module.ALLOW_PRIVATE_CALLBACKS_ENV):
            resolve_callback_url("http://evil.example.com/", allow_private=False)

    def test_any_private_record_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_dns(monkeypatch, {"mixed.example.com": ["93.184.216.34", "10.0.0.1"]})
        with pytest.raises(ValueError):
            resolve_callback_url("http://mixed.example.com/", allow_private=False)

    @pytest.mark.parametrize(
        ("url", "message"),
        [
            ("file:///etc/passwd", "http"),
            ("gopher://example.com/", "http"),
            ("http:///nohost", "host"),
            ("http://user:pw@example.com/", "credentials"),
            ("http://example.com:99999/", "valid URL"),
            ("http://unresolvable.invalid/", "cannot be resolved"),
        ],
    )
    def test_malformed_urls_rejected(
        self, monkeypatch: pytest.MonkeyPatch, url: str, message: str
    ) -> None:
        _fake_dns(monkeypatch, {"example.com": ["93.184.216.34"]})
        with pytest.raises(ValueError, match=message):
            resolve_callback_url(url, allow_private=True)

    def test_opt_in_allows_private(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_dns(monkeypatch, {"intranet": ["10.0.0.7"]})
        assert resolve_callback_url("http://intranet/x", allow_private=True).address == "10.0.0.7"

    def test_env_opt_in(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _server(tmp_path).allow_private_callbacks is False
        monkeypatch.setenv(server_module.ALLOW_PRIVATE_CALLBACKS_ENV, "1")
        assert _server(tmp_path).allow_private_callbacks is True

    def test_origin_never_contains_path_or_query(self) -> None:
        assert callback_origin("https://h.example/p/secret?token=x") == "https://h.example"

    def test_trigger_rejects_internal_callback(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        client.post(
            "/api/workflows",
            json={"name": "cp", "steps": [{"id": "a", "kind": "checkpoint", "ref": "cp"}]},
            headers=ADMIN_H,
        )
        for url in ("http://169.254.169.254/latest/meta-data", "http://127.0.0.1:1/", "ftp://x/"):
            r = client.post(
                "/api/workflows/cp/trigger", json={"callback_url": url}, headers=ADMIN_H
            )
            assert r.status_code == 422, (url, r.text)


class _CallbackRecorder(http.server.BaseHTTPRequestHandler):
    hits: list[tuple[str, bytes]] = []
    reply_status = 200

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("content-length", "0")))
        type(self).hits.append((self.path, body))
        self.send_response(type(self).reply_status)
        if type(self).reply_status == 302:
            self.send_header("Location", "/redirected")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def callback_server() -> Iterator[tuple[str, type[_CallbackRecorder]]]:
    handler = type("Recorder", (_CallbackRecorder,), {"hits": [], "reply_status": 200})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def _wait_for(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


class TestCallbackDelivery:
    def test_posts_result_without_following_redirects(
        self,
        tmp_path: Path,
        callback_server: tuple[str, type[_CallbackRecorder]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base, handler = callback_server
        handler.reply_status = 302
        server = _server(tmp_path, admin_token=ADMIN, allow_private_callbacks=True)
        caplog.set_level(logging.WARNING, logger="agent_vcr.server")
        with TestClient(server.app, raise_server_exceptions=False) as client:
            client.post(
                "/api/workflows",
                json={"name": "cp", "steps": [{"id": "a", "kind": "checkpoint", "ref": "cp"}]},
                headers=ADMIN_H,
            )
            r = client.post(
                "/api/workflows/cp/trigger",
                json={"callback_url": f"{base}/hook/s3cr3t?sig=abc"},
                headers=ADMIN_H,
            )
            assert r.status_code == 202, r.text
            _wait_for(lambda: handler.hits and not server._background_tasks)
        assert len(handler.hits) == 1
        path, body = handler.hits[0]
        assert path == "/hook/s3cr3t?sig=abc"
        assert json.loads(body)["status"] in {"committed", "failed"}
        # 302 was not followed, and the log names only scheme://host.
        assert all(p != "/redirected" for p, _ in handler.hits)
        logged = caplog.text
        assert "HTTP 302" in logged
        assert "s3cr3t" not in logged and "sig=abc" not in logged

    def test_background_task_is_strongly_referenced(
        self, tmp_path: Path, callback_server: tuple[str, type[_CallbackRecorder]]
    ) -> None:
        base, handler = callback_server
        server = _server(tmp_path, admin_token=ADMIN, allow_private_callbacks=True)
        release = threading.Event()
        seen: list[int] = []

        original = server._run_and_callback

        async def slow(*args: Any) -> None:
            seen.append(len(server._background_tasks))
            await asyncio.to_thread(release.wait, 5)
            await original(*args)

        server._run_and_callback = slow  # type: ignore[method-assign]
        with TestClient(server.app, raise_server_exceptions=False) as client:
            client.post(
                "/api/workflows",
                json={"name": "cp", "steps": [{"id": "a", "kind": "checkpoint", "ref": "cp"}]},
                headers=ADMIN_H,
            )
            r = client.post(
                "/api/workflows/cp/trigger", json={"callback_url": f"{base}/h"}, headers=ADMIN_H
            )
            assert r.status_code == 202
            _wait_for(lambda: bool(seen))
            assert seen == [1]
            release.set()
            _wait_for(lambda: bool(handler.hits) and not server._background_tasks)

    def test_callback_refused_if_dns_changes_to_private(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        answers = iter([["93.184.216.34"], ["169.254.169.254"]])

        def getaddrinfo(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in next(answers)]

        monkeypatch.setattr(server_module.socket, "getaddrinfo", getaddrinfo)
        posted: list[Any] = []
        monkeypatch.setattr(server_module, "post_callback", lambda *a: posted.append(a) or 200)
        caplog.set_level(logging.WARNING, logger="agent_vcr.server")
        server = _server(tmp_path, admin_token=ADMIN)
        with TestClient(server.app, raise_server_exceptions=False) as client:
            client.post(
                "/api/workflows",
                json={"name": "cp", "steps": [{"id": "a", "kind": "checkpoint", "ref": "cp"}]},
                headers=ADMIN_H,
            )
            r = client.post(
                "/api/workflows/cp/trigger",
                json={"callback_url": "http://rebind.example/x"},
                headers=ADMIN_H,
            )
            assert r.status_code == 202
            _wait_for(lambda: not server._background_tasks)
        assert posted == []
        assert "refused" in caplog.text


# -- 2. Workflows stay admin-only -------------------------------------------


def test_project_tokens_cannot_reach_workflow_routes(tmp_path: Path) -> None:
    client = _client(tmp_path, admin_token=ADMIN)
    pid = client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H).json()["id"]
    owner = client.post(
        f"/api/projects/{pid}/tokens", json={"role": "owner"}, headers=ADMIN_H
    ).json()["token"]
    client.post(
        "/api/workflows",
        json={"name": "cp", "steps": [{"id": "a", "kind": "checkpoint", "ref": "cp"}]},
        headers=ADMIN_H,
    )
    run_id = client.post("/api/workflows/cp/trigger", json={}, headers=ADMIN_H).json()["run_id"]
    owner_h = {"Authorization": f"Bearer {owner}"}
    for method, url in [
        ("GET", "/api/workflows"),
        ("POST", "/api/workflows"),
        ("GET", "/api/workflows/cp"),
        ("POST", "/api/workflows/cp/trigger"),
        ("GET", "/api/workflows/cp/runs"),
        ("GET", f"/api/workflow-runs/{run_id}"),
        ("POST", f"/api/workflow-runs/{run_id}/resume"),
    ]:
        r = client.request(method, url, json={}, headers=owner_h)
        assert r.status_code == 403, (method, url, r.status_code)


# -- 3. Cloud endpoint status codes -----------------------------------------


class TestCloudEndpoints:
    def test_create_project_honours_id_and_description(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        r = client.post(
            "/api/projects",
            json={"name": "Refunds", "id": "refunds-prod", "description": "prod agents"},
            headers=ADMIN_H,
        )
        assert r.status_code == 201
        assert r.json()["id"] == "refunds-prod"
        assert r.json()["description"] == "prod agents"
        # cloud.py's client sends "id": null when none is given.
        r = client.post("/api/projects", json={"name": "Other", "id": None}, headers=ADMIN_H)
        assert r.status_code == 201 and r.json()["id"] == "other"

    def test_create_project_errors(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        assert client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H).status_code == 201
        dup = client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H)
        assert dup.status_code == 409 and "already exists" in dup.json()["detail"]
        bad = client.post("/api/projects", json={"name": "!!!"}, headers=ADMIN_H)
        assert bad.status_code == 422 and "explicit id" in bad.json()["detail"]
        for body in ({"name": "x", "id": "../evil"}, {"name": ""}, {"name": 3},
                     {"name": "x", "description": 5}, {"name": "x", "id": 7}):
            assert client.post("/api/projects", json=body, headers=ADMIN_H).status_code == 422

    def test_token_errors(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H)
        bad_role = client.post("/api/projects/p/tokens", json={"role": "root"}, headers=ADMIN_H)
        assert bad_role.status_code == 422 and "owner" in bad_role.json()["detail"]
        missing = client.post("/api/projects/nope/tokens", json={}, headers=ADMIN_H)
        assert missing.status_code == 404 and "nope" in missing.json()["detail"]
        assert client.get("/api/projects/nope/tokens", headers=ADMIN_H).status_code == 404
        assert client.get("/api/projects/nope/sessions", headers=ADMIN_H).status_code == 404

    def test_revoke_unknown_is_404_and_repeat_is_idempotent(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H)
        tid = client.post("/api/projects/p/tokens", json={}, headers=ADMIN_H).json()["id"]
        assert client.post("/api/projects/p/tokens/zzz/revoke", headers=ADMIN_H).status_code == 404
        assert client.post("/api/projects/nope/tokens/x/revoke", headers=ADMIN_H).status_code == 404
        for _ in range(2):
            r = client.post(f"/api/projects/p/tokens/{tid}/revoke", headers=ADMIN_H)
            assert r.status_code == 200 and r.json() == {"status": "revoked"}

    def test_missing_sessions_are_404(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H)
        assert client.delete("/api/projects/p/sessions/none", headers=ADMIN_H).status_code == 404
        assert client.get("/api/projects/p/sessions/none", headers=ADMIN_H).status_code == 404
        assert (
            client.get("/api/projects/p/sessions/bad%20id/download", headers=ADMIN_H).status_code
            == 404
        )
        assert client.get("/api/projects/nope/intel/risk", headers=ADMIN_H).status_code == 404


# -- 4. Token store: throttled writes, off the event loop -------------------


class TestTokenStore:
    def test_last_used_written_at_most_once_per_interval(self, tmp_path: Path) -> None:
        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        _, secret = store.mint_token("p", "t", role="owner")
        tokens_file = tmp_path / "tokens.json"

        first = store.verify_token(secret)
        assert first is not None and first.last_used_at is not None
        before = tokens_file.read_bytes()
        mtime = os.stat(tokens_file).st_mtime_ns
        for _ in range(5):
            assert store.verify_token(secret) is not None
        assert tokens_file.read_bytes() == before
        assert os.stat(tokens_file).st_mtime_ns == mtime

        # Once the recorded use is older than the interval, it is refreshed.
        raw = json.loads(before)
        stale = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        raw["tokens"][0]["last_used_at"] = stale
        tokens_file.write_text(json.dumps(raw))
        refreshed = store.verify_token(secret)
        assert refreshed is not None and refreshed.last_used_at != stale
        assert json.loads(tokens_file.read_text())["tokens"][0]["last_used_at"] != stale

    def test_revoked_tokens_still_rejected(self, tmp_path: Path) -> None:
        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        public, secret = store.mint_token("p", "t")
        assert store.verify_token(secret) is not None
        assert store.revoke_token("p", public.id)
        assert store.verify_token(secret) is None

    def test_store_calls_run_off_the_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_loop: list[bool] = []

        def _running_loop() -> bool:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return False
            return True

        for name in ("verify_token", "list_session_headers", "create_project", "mint_token"):
            original = getattr(CloudStore, name)

            def spy(self: CloudStore, *a: Any, __o: Any = original, **k: Any) -> Any:
                on_loop.append(_running_loop())
                return __o(self, *a, **k)

            monkeypatch.setattr(CloudStore, name, spy)
        client = _client(tmp_path, admin_token=ADMIN)
        client.post("/api/projects", json={"name": "P"}, headers=ADMIN_H)
        token = client.post("/api/projects/p/tokens", json={}, headers=ADMIN_H).json()["token"]
        r = client.get("/api/projects/p/sessions", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert on_loop and not any(on_loop)


# -- 5. JWKS negative cache, lock-free hits, timeouts ------------------------


def _hosted_env(**overrides: str) -> dict[str, str]:
    values = {
        "AGENT_VCR_SUPABASE_URL": "https://tenant.supabase.co",
        "DATABASE_URL": "postgresql://app:password@postgres:5432/agent_vcr",
        "AGENT_VCR_APP_URL": "https://app.agent-vcr.example",
        "AGENT_VCR_STRIPE_SECRET_KEY": "sk_test_123",
        "AGENT_VCR_STRIPE_WEBHOOK_SECRET": "whsec_test_123",
        "AGENT_VCR_STRIPE_PRICE_ID": "price_pro_123",
    }
    values.update(overrides)
    return values


@pytest.fixture
def fake_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The token string doubles as its ``kid``; any known key's signature is accepted."""
    module: Any = types.ModuleType("jwt")
    module.get_unverified_header = lambda token: {"kid": token, "alg": "RS256"}
    module.PyJWK = SimpleNamespace(from_dict=lambda value: SimpleNamespace(key="k"))
    module.decode = lambda token, key, **kw: {"sub": f"user-{token}"}
    monkeypatch.setitem(sys.modules, "jwt", module)


class TestJWKS:
    def _verifier(
        self, monkeypatch: pytest.MonkeyPatch, fetches: list[float]
    ) -> SupabaseJWTVerifier:
        def fetch(url: str, timeout: float) -> dict[str, Any]:
            fetches.append(time.monotonic())
            return {"keys": [{"kid": "good", "alg": "RS256", "kty": "RSA"}]}

        monkeypatch.setattr(hosted_module, "_fetch_json", fetch)
        return SupabaseJWTVerifier(HostedConfig.from_environment(_hosted_env()))

    def test_unknown_kids_do_not_refetch_within_interval(
        self, monkeypatch: pytest.MonkeyPatch, fake_jwt: None
    ) -> None:
        fetches: list[float] = []
        verifier = self._verifier(monkeypatch, fetches)

        async def scenario() -> None:
            assert (await verifier.verify("good")).id == "user-good"
            for i in range(20):
                with pytest.raises(HostedAuthenticationError):
                    await verifier.verify(f"random-{i}")
            assert (await verifier.verify("good")).id == "user-good"

        asyncio.run(scenario())
        assert len(fetches) == 1

        # After the refetch interval an unknown kid may trigger one more fetch.
        verifier._last_fetch_at = time.monotonic() - verifier._MIN_REFETCH_SECONDS - 1
        with pytest.raises(HostedAuthenticationError):
            asyncio.run(verifier.verify("rotated"))
        assert len(fetches) == 2

    def test_cache_hits_do_not_wait_for_the_refresh_lock(
        self, monkeypatch: pytest.MonkeyPatch, fake_jwt: None
    ) -> None:
        fetches: list[float] = []
        verifier = self._verifier(monkeypatch, fetches)

        async def scenario() -> None:
            await verifier.verify("good")
            lock = verifier._refresh_lock()
            await lock.acquire()  # simulate a slow refresh in progress
            try:
                user = await asyncio.wait_for(verifier.verify("good"), timeout=0.5)
                assert user.id == "user-good"
            finally:
                lock.release()

        asyncio.run(scenario())

    def test_concurrent_refresh_is_single_flight(
        self, monkeypatch: pytest.MonkeyPatch, fake_jwt: None
    ) -> None:
        calls: list[int] = []

        def fetch(url: str, timeout: float) -> dict[str, Any]:
            calls.append(1)
            time.sleep(0.2)
            return {"keys": [{"kid": "good", "alg": "RS256"}]}

        monkeypatch.setattr(hosted_module, "_fetch_json", fetch)
        verifier = SupabaseJWTVerifier(HostedConfig.from_environment(_hosted_env()))

        async def scenario() -> None:
            results = await asyncio.gather(
                *(verifier.verify("good") for _ in range(5)),
                *(verifier.verify(f"r{i}") for i in range(5)),
                return_exceptions=True,
            )
            assert all(isinstance(r, HostedUser) for r in results[:5])
            assert all(isinstance(r, HostedAuthenticationError) for r in results[5:])

        asyncio.run(scenario())
        assert len(calls) == 1

    def test_fetch_failure_is_unavailable_with_backoff(
        self, monkeypatch: pytest.MonkeyPatch, fake_jwt: None
    ) -> None:
        calls: list[int] = []

        def fetch(url: str, timeout: float) -> dict[str, Any]:
            calls.append(1)
            raise OSError("connection refused")

        monkeypatch.setattr(hosted_module, "_fetch_json", fetch)
        verifier = SupabaseJWTVerifier(HostedConfig.from_environment(_hosted_env()))
        for _ in range(3):
            with pytest.raises(HostedUnavailableError):
                asyncio.run(verifier.verify("good"))
        assert len(calls) == 1

    def test_server_maps_auth_timeout_to_503(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = HostedControlPlane(HostedConfig.from_environment(_hosted_env()))

        async def hang(token: str) -> HostedUser:
            await asyncio.sleep(5)
            raise AssertionError("unreachable")

        monkeypatch.setattr(plane.auth, "verify", hang)
        monkeypatch.setattr(server_module, "_HOSTED_AUTH_TIMEOUT_S", 0.05)
        client = TestClient(VCRServer(vcr_dir=str(tmp_path), hosted=plane).app,
                            raise_server_exceptions=False)
        r = client.get("/api/hosted/me", headers={"Authorization": "Bearer t"})
        assert r.status_code == 503


# -- 6. Stripe: checkout guard, ordered and idempotent webhooks -------------


class _SqliteCursor:
    def __init__(self, db: sqlite3.Connection):
        self._cursor = db.cursor()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        if sql.strip().upper().startswith("SET LOCAL"):
            return
        self._cursor.execute(sql.replace("%s", "?"), params)

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def __enter__(self) -> _SqliteCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cursor.close()


class _SqliteConnection:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path)

    def cursor(self) -> _SqliteCursor:
        return _SqliteCursor(self._db)

    def __enter__(self) -> _SqliteConnection:
        return self

    def __exit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is None:
            self._db.commit()
        else:
            self._db.rollback()
        self._db.close()


@pytest.fixture
def fake_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A psycopg stand-in backed by SQLite, which speaks the same upsert dialect."""
    state: dict[str, Any] = {"down": False, "path": str(tmp_path / "hosted.sqlite")}
    module = types.ModuleType("psycopg")
    module.Error = sqlite3.Error  # type: ignore[attr-defined]

    def connect(url: str, **kwargs: Any) -> _SqliteConnection:
        if state["down"]:
            raise sqlite3.OperationalError("connection refused")
        return _SqliteConnection(state["path"])

    module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return state


def _store() -> HostedAccountStore:
    store = HostedAccountStore("postgresql://test/db")
    store.ensure_schema()
    return store


_counter = iter(range(1, 10_000))


def _sub_event(
    sub: str, status: str, created: int, *, user: str | None = "u1", customer: str = "cus_1",
    kind: str = "updated",
) -> dict[str, Any]:
    obj: dict[str, Any] = {"id": sub, "status": status, "customer": customer}
    if user:
        obj["metadata"] = {"agent_vcr_user_id": user}
    return {
        "id": f"evt_{next(_counter)}",
        "type": f"customer.subscription.{kind}",
        "created": created,
        "data": {"object": obj},
    }


def _checkout_event(sub: str, *, user: str = "u1", customer: str = "cus_1") -> dict[str, Any]:
    return {
        "id": f"evt_{next(_counter)}",
        "type": "checkout.session.completed",
        "created": 1,
        "data": {"object": {"client_reference_id": user, "customer": customer,
                            "subscription": sub}},
    }


def _account(store: HostedAccountStore, user: str = "u1") -> HostedAccount:
    account = store.get_account(user)
    assert account is not None
    return account


class TestStripeWebhooks:
    def test_happy_path_and_idempotent_redelivery(self, fake_db: dict[str, Any]) -> None:
        store = _store()
        store.ensure_account(HostedUser("u1", "dev@example.com"))
        assert store.apply_stripe_event(_checkout_event("sub_A"))
        assert _account(store).subscription_status == "pending"
        event = _sub_event("sub_A", "active", 100, kind="created")
        assert store.apply_stripe_event(event) is True
        assert store.apply_stripe_event(event) is False  # same event id
        account = _account(store)
        assert account.entitled and account.stripe_subscription_id == "sub_A"
        assert account.email == "dev@example.com"

    def test_stale_update_after_delete_is_ignored(self, fake_db: dict[str, Any]) -> None:
        store = _store()
        store.apply_stripe_event(_sub_event("sub_A", "active", 100, kind="created"))
        store.apply_stripe_event(_sub_event("sub_A", "canceled", 300, kind="deleted"))
        # Out of order: an older update, and even a "newer" one, cannot revive it.
        store.apply_stripe_event(_sub_event("sub_A", "active", 200))
        store.apply_stripe_event(_sub_event("sub_A", "active", 400))
        account = _account(store)
        assert account.plan == "free" and account.subscription_status == "canceled"

    def test_older_event_does_not_override_newer(self, fake_db: dict[str, Any]) -> None:
        store = _store()
        store.apply_stripe_event(_sub_event("sub_A", "past_due", 200))
        store.apply_stripe_event(_sub_event("sub_A", "active", 100))
        assert _account(store).subscription_status == "past_due"

    def test_two_subscriptions_cancel_either(self, fake_db: dict[str, Any]) -> None:
        store = _store()
        store.apply_stripe_event(_sub_event("sub_A", "active", 100, kind="created"))
        store.apply_stripe_event(_sub_event("sub_B", "active", 110, kind="created"))
        store.apply_stripe_event(_sub_event("sub_B", "canceled", 120, kind="deleted"))
        account = _account(store)
        assert account.entitled and account.stripe_subscription_id == "sub_A"
        store.apply_stripe_event(_sub_event("sub_A", "canceled", 130, kind="deleted"))
        assert _account(store).plan == "free"

    def test_subscription_event_before_checkout_links_later(
        self, fake_db: dict[str, Any]
    ) -> None:
        store = _store()
        # No metadata and an unknown customer: stored unlinked.
        store.apply_stripe_event(_sub_event("sub_A", "active", 100, user=None, customer="cus_9"))
        assert store.get_account("u9") is None
        store.apply_stripe_event(_checkout_event("sub_A", user="u9", customer="cus_9"))
        account = _account(store, "u9")
        assert account.entitled and account.stripe_subscription_id == "sub_A"

    def test_legacy_rows_are_backfilled(self, fake_db: dict[str, Any]) -> None:
        store = _store()
        db = sqlite3.connect(fake_db["path"])
        db.execute(
            "INSERT INTO agent_vcr_hosted_accounts (user_id, stripe_customer_id, "
            "stripe_subscription_id, plan, subscription_status) "
            "VALUES ('u1', 'cus_1', 'sub_OLD', 'pro', 'active')"
        )
        db.commit()
        db.close()
        store.ensure_schema()  # the migration step
        store.apply_stripe_event(_sub_event("sub_NEW", "canceled", 50, kind="deleted"))
        assert _account(store).entitled  # sub_OLD still counts

    def test_database_outage_is_503_not_502(
        self, tmp_path: Path, fake_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = HostedControlPlane(HostedConfig.from_environment(_hosted_env()))
        plane.accounts.ensure_schema()

        async def verify(token: str) -> HostedUser:
            return HostedUser("u1", None)

        monkeypatch.setattr(plane.auth, "verify", verify)
        assert plane.billing is not None
        monkeypatch.setattr(
            plane.billing, "verify_webhook", lambda payload, sig: _sub_event("s", "active", 1)
        )
        client = TestClient(VCRServer(vcr_dir=str(tmp_path / "v"), hosted=plane).app,
                            raise_server_exceptions=False)
        fake_db["down"] = True
        bearer = {"Authorization": "Bearer t"}
        assert client.get("/api/hosted/me", headers=bearer).status_code == 503
        assert client.post("/api/hosted/billing/checkout", headers=bearer).status_code == 503
        hook = client.post("/api/hosted/billing/webhook", content=b"{}",
                           headers={"stripe-signature": "x"})
        assert hook.status_code == 503
        fake_db["down"] = False
        assert client.post("/api/hosted/billing/webhook", content=b"{}",
                           headers={"stripe-signature": "x"}).json()["processed"] is True

    def test_malformed_event_is_400(
        self, tmp_path: Path, fake_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = HostedControlPlane(HostedConfig.from_environment(_hosted_env()))
        assert plane.billing is not None
        monkeypatch.setattr(plane.billing, "verify_webhook", lambda payload, sig: {"type": "x"})
        client = TestClient(VCRServer(vcr_dir=str(tmp_path / "v"), hosted=plane).app,
                            raise_server_exceptions=False)
        r = client.post("/api/hosted/billing/webhook", content=b"{}",
                        headers={"stripe-signature": "x"})
        assert r.status_code == 400


class TestCheckoutGuard:
    @pytest.mark.parametrize("status", ["active", "trialing", "past_due", "pending"])
    def test_billing_refuses_second_subscription(self, status: str) -> None:
        billing = StripeBilling(HostedConfig.from_environment(_hosted_env()))
        account = HostedAccount("u1", None, "cus_1", "sub_A",
                                "pro" if status in {"active", "trialing"} else "free", status)
        with pytest.raises(HostedConflictError):
            billing.create_checkout(HostedUser("u1", None), account)

    def test_checkout_route_returns_409_when_subscribed(
        self, tmp_path: Path, fake_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plane = HostedControlPlane(HostedConfig.from_environment(_hosted_env()))
        plane.accounts.ensure_schema()
        plane.accounts.apply_stripe_event(_sub_event("sub_A", "active", 1, kind="created"))

        async def verify(token: str) -> HostedUser:
            return HostedUser("u1", None)

        monkeypatch.setattr(plane.auth, "verify", verify)
        client = TestClient(VCRServer(vcr_dir=str(tmp_path / "v"), hosted=plane).app,
                            raise_server_exceptions=False)
        r = client.post("/api/hosted/billing/checkout", headers={"Authorization": "Bearer t"})
        assert r.status_code == 409
        assert "portal" in r.json()["detail"]


# -- 7/8. Lazy locks and locked broadcast ------------------------------------


class TestLocks:
    def test_locks_are_created_inside_the_running_loop(self) -> None:
        watcher = VCRFileWatcher(Path("."))
        verifier = SupabaseJWTVerifier(HostedConfig.from_environment(_hosted_env()))
        assert watcher._lock_obj is None and verifier._lock is None

        async def use() -> None:
            async with watcher._lock, verifier._refresh_lock():
                pass

        # Two separate loops (as with uvicorn after import-time construction).
        asyncio.run(use())
        asyncio.run(use())

    def test_broadcast_drops_dead_clients_and_survives_concurrent_disconnects(self) -> None:
        watcher = VCRFileWatcher(Path("."))
        received: list[dict] = []

        class Good:
            async def send_json(self, message: dict) -> None:
                await asyncio.sleep(0)
                received.append(message)

        class Dead:
            async def send_json(self, message: dict) -> None:
                raise RuntimeError("gone")

        async def scenario() -> None:
            good, dead = Good(), Dead()
            await watcher.connect(good)  # type: ignore[arg-type]
            await watcher.connect(dead)  # type: ignore[arg-type]
            await asyncio.gather(
                watcher.broadcast({"n": 1}),
                watcher.disconnect(good),  # type: ignore[arg-type]
                watcher.broadcast({"n": 2}),
            )
            assert watcher.connections == []

        asyncio.run(scenario())
        assert {"n": 1} in received

    def test_scheduled_broadcasts_are_referenced(self) -> None:
        watcher = VCRFileWatcher(Path("."))

        async def scenario() -> None:
            watcher.set_event_loop(asyncio.get_running_loop())
            await asyncio.to_thread(watcher._schedule_broadcast, {"x": 1})
            await asyncio.sleep(0)
            # The task is held until it completes, then discarded.
            for _ in range(50):
                if not watcher._tasks:
                    break
                await asyncio.sleep(0.01)
            assert not watcher._tasks

        asyncio.run(scenario())
