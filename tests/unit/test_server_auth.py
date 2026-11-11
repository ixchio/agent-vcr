"""Auth and hardening tests for the dashboard server (``agent_vcr.server``).

Covers: refusing unauthenticated public binds, env-var token defaults, admin-only
project creation / token minting, the token matrix on HTTP and WebSocket routes,
bounded uploads, and HTTPException passthrough in generic error handlers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import agent_vcr.server as server_module
from agent_vcr.hosted import HostedConfig, HostedControlPlane
from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder
from agent_vcr.server import VCRServer, bind_safety_problem, is_loopback_host, main

ADMIN = "admin-secret"
AUTH = "auth-secret"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        server_module.AUTH_TOKEN_ENV,
        server_module.ADMIN_TOKEN_ENV,
        server_module.ALLOW_UNAUTHENTICATED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def _record(directory: Path, session_id: str = "sess-1") -> Path:
    rec = VCRRecorder(output_dir=str(directory), auto_save=False)
    rec.start_session(session_id)
    rec.record_step("step", {"a": 1}, {"b": 2}, metadata=FrameMetadata(tokens_used=5))
    return Path(rec.save())


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    _record(tmp_path / "legacy")
    server = VCRServer(vcr_dir=str(tmp_path / "legacy"), data_dir=str(tmp_path / "cloud"), **kwargs)
    return TestClient(server.app, raise_server_exceptions=False)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _assert_ws_rejected(
    client: TestClient, url: str, headers: dict[str, str] | None = None
) -> None:
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(url, headers=headers or {}) as ws,
    ):
        ws.receive_json()


def _project_tokens(client: TestClient, admin: str) -> tuple[str, str, str]:
    pid = client.post("/api/projects", json={"name": "Proj"}, headers=_bearer(admin)).json()["id"]
    owner = client.post(
        f"/api/projects/{pid}/tokens", json={"role": "owner"}, headers=_bearer(admin)
    ).json()["token"]
    member = client.post(
        f"/api/projects/{pid}/tokens", json={"role": "member"}, headers=_bearer(admin)
    ).json()["token"]
    return pid, owner, member


# -- Bug 1: startup safety, env vars, admin-only project creation -------------


class _RunCapture:
    """Stands in for ``VCRServer.run``: performs the safety check, skips uvicorn."""

    def __init__(self) -> None:
        self.servers: list[VCRServer] = []


@pytest.fixture
def run_capture(monkeypatch: pytest.MonkeyPatch) -> _RunCapture:
    capture = _RunCapture()

    def fake_run(server: VCRServer) -> None:
        server.check_bind_safety()
        capture.servers.append(server)

    monkeypatch.setattr(VCRServer, "run", fake_run)
    return capture


class TestBindSafety:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "LOCALHOST", "127.0.0.5"])
    def test_loopback_hosts(self, host: str) -> None:
        assert is_loopback_host(host)
        assert bind_safety_problem(host, None, None) is None

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com"])
    def test_public_hosts_need_a_token(self, host: str) -> None:
        assert not is_loopback_host(host)
        assert bind_safety_problem(host, None, None) is not None
        assert bind_safety_problem(host, "t", None) is None
        assert bind_safety_problem(host, None, "t") is None

    def test_cli_refuses_public_bind_without_tokens(
        self, tmp_path: Path, run_capture: _RunCapture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--host", "0.0.0.0", "--vcr-dir", str(tmp_path)])
        assert exc.value.code == 2
        assert "--allow-unauthenticated" in capsys.readouterr().err
        assert run_capture.servers == []

    def test_cli_loopback_without_tokens_still_works(
        self, tmp_path: Path, run_capture: _RunCapture
    ) -> None:
        main(["--vcr-dir", str(tmp_path)])
        (server,) = run_capture.servers
        assert server.host == "127.0.0.1"
        assert server.auth_token is None and server.admin_token is None

    def test_cli_reads_tokens_from_env(
        self, tmp_path: Path, run_capture: _RunCapture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(server_module.AUTH_TOKEN_ENV, "env-auth")
        monkeypatch.setenv(server_module.ADMIN_TOKEN_ENV, "env-admin")
        main(["--host", "0.0.0.0", "--vcr-dir", str(tmp_path)])
        (server,) = run_capture.servers
        assert (server.auth_token, server.admin_token) == ("env-auth", "env-admin")

    def test_cli_flag_wins_over_env(
        self, tmp_path: Path, run_capture: _RunCapture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(server_module.ADMIN_TOKEN_ENV, "env-admin")
        main(["--host", "0.0.0.0", "--vcr-dir", str(tmp_path), "--admin-token", "cli-admin"])
        assert run_capture.servers[0].admin_token == "cli-admin"

    def test_blank_env_token_does_not_count(
        self, tmp_path: Path, run_capture: _RunCapture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(server_module.ADMIN_TOKEN_ENV, "   ")
        with pytest.raises(SystemExit):
            main(["--host", "0.0.0.0", "--vcr-dir", str(tmp_path)])

    @pytest.mark.parametrize("via_env", [False, True])
    def test_allow_unauthenticated_warns_loudly(
        self,
        tmp_path: Path,
        run_capture: _RunCapture,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        via_env: bool,
    ) -> None:
        argv = ["--host", "0.0.0.0", "--vcr-dir", str(tmp_path)]
        if via_env:
            monkeypatch.setenv(server_module.ALLOW_UNAUTHENTICATED_ENV, "1")
        else:
            argv.append("--allow-unauthenticated")
        main(argv)
        assert len(run_capture.servers) == 1
        assert "UNAUTHENTICATED" in capsys.readouterr().err

    def test_programmatic_run_also_refuses(self, tmp_path: Path) -> None:
        server = VCRServer(vcr_dir=str(tmp_path), host="0.0.0.0")
        with pytest.raises(RuntimeError, match="non-loopback"):
            server.check_bind_safety()


class TestAdminOnlyProvisioning:
    def test_disabled_without_admin_token(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        assert client.post("/api/projects", json={"name": "x"}).status_code == 403
        assert client.post("/api/projects/p/tokens", json={"role": "owner"}).status_code == 403
        # Local dev ergonomics for read-only/legacy routes are unchanged.
        assert client.get("/api/sessions").status_code == 200
        assert client.get("/api/projects").status_code == 200

    def test_anonymous_and_project_tokens_cannot_provision(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        pid, owner, _ = _project_tokens(client, ADMIN)
        assert client.post("/api/projects", json={"name": "x"}).status_code == 401
        assert client.post(f"/api/projects/{pid}/tokens", json={"role": "owner"}).status_code == 401
        minted = client.post(
            f"/api/projects/{pid}/tokens", json={"role": "owner"}, headers=_bearer(owner)
        )
        assert minted.status_code == 403
        assert (
            client.post("/api/projects", json={"name": "x"}, headers=_bearer(owner)).status_code
            == 403
        )


# -- Bug 2: admin + project tokens accepted when both tokens are configured ----


class TestBothTokensConfigured:
    def test_token_matrix(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN, auth_token=AUTH)
        pid, owner, member = _project_tokens(client, ADMIN)

        for token in (ADMIN, AUTH):
            assert client.get("/api/sessions", headers=_bearer(token)).status_code == 200
            assert (
                client.get("/api/projects", headers={"x-agent-vcr-token": token}).status_code == 200
            )

        assert (
            client.get(f"/api/projects/{pid}/sessions", headers=_bearer(member)).status_code == 200
        )
        src = _record(tmp_path / "rec", "up-1")
        with open(src, "rb") as f:
            up = client.post(
                f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=_bearer(owner)
            )
        assert up.status_code == 200, up.text

        # Project tokens stay out of server-global routes.
        assert client.get("/api/sessions", headers=_bearer(owner)).status_code == 403
        assert client.post("/api/push", json={}, headers=_bearer(owner)).status_code == 403
        assert client.get("/api/workflows", headers=_bearer(owner)).status_code == 403
        assert client.get("/api/projects", headers=_bearer(owner)).status_code == 403
        # ...and out of other projects.
        other = client.post("/api/projects", json={"name": "Other"}, headers=_bearer(ADMIN)).json()
        assert (
            client.get(f"/api/projects/{other['id']}/sessions", headers=_bearer(member)).status_code
            == 403
        )

        assert client.get("/api/sessions", headers=_bearer("wrong")).status_code == 401

    def test_project_tokens_scoped_with_only_auth_token(self, tmp_path: Path) -> None:
        client = _client(tmp_path, auth_token=AUTH)
        pid, _, member = _project_tokens(client, AUTH)
        other = client.post("/api/projects", json={"name": "Other"}, headers=_bearer(AUTH)).json()
        assert (
            client.get(f"/api/projects/{pid}/sessions", headers=_bearer(member)).status_code == 200
        )
        assert (
            client.get(f"/api/projects/{other['id']}/sessions", headers=_bearer(member)).status_code
            == 403
        )


# -- Bug 3: any configured token is required on HTTP and WebSocket ------------


class TestAdminTokenOnly:
    def test_http_routes_require_token(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        assert client.get("/api/sessions").status_code == 401
        assert client.get("/api/tags").status_code == 401
        assert client.get("/api/sessions/sess-1").status_code == 401
        assert client.post("/api/push", json={"session_id": "s"}).status_code == 401
        assert client.get("/api/workflows").status_code == 401
        assert client.get("/api/sessions", headers=_bearer(ADMIN)).status_code == 200
        assert client.post("/api/push", json={}, headers=_bearer(ADMIN)).status_code == 200

    def test_query_token_not_accepted_over_http(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        assert client.get(f"/api/sessions?token={ADMIN}").status_code == 401

    def test_health_stays_public(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and body["auth"] is True

    @pytest.mark.parametrize("kwargs", [{"admin_token": ADMIN}, {"auth_token": ADMIN}])
    def test_websocket_requires_token(self, tmp_path: Path, kwargs: dict[str, str]) -> None:
        client = _client(tmp_path, **kwargs)
        _assert_ws_rejected(client, "/ws/live")
        _assert_ws_rejected(client, "/ws/live?token=wrong")
        for url, headers in (
            ("/ws/live", _bearer(ADMIN)),
            ("/ws/live", {"x-agent-vcr-token": ADMIN}),
            (f"/ws/live?token={ADMIN}", {}),
        ):
            with client.websocket_connect(url, headers=headers or {}) as ws:
                ws.send_json({"action": "ping"})
                assert ws.receive_json()["type"] == "pong"

    def test_websocket_rejects_project_tokens(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        _, owner, _ = _project_tokens(client, ADMIN)
        _assert_ws_rejected(client, "/ws/live", _bearer(owner))

    def test_non_ascii_token_is_rejected_not_crashing(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=ADMIN)
        resp = client.get("/api/sessions", headers={"x-agent-vcr-token": "tök".encode()})
        assert resp.status_code == 401

    def test_hosted_namespace_uses_its_own_auth(self, tmp_path: Path) -> None:
        env = {
            "AGENT_VCR_SUPABASE_URL": "https://tenant.supabase.co",
            "DATABASE_URL": "postgresql://app:password@postgres:5432/agent_vcr",
            "AGENT_VCR_APP_URL": "https://app.agent-vcr.example",
        }
        hosted = HostedControlPlane(HostedConfig.from_environment(env))
        server = VCRServer(vcr_dir=str(tmp_path), admin_token=ADMIN, hosted=hosted)
        client = TestClient(server.app, raise_server_exceptions=False)
        resp = client.get("/api/hosted/me")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Bearer token required"  # hosted check, not the VCR gate


# -- Bug 4: bounded uploads ----------------------------------------------------


class TestUploadLimit:
    def _setup(self, tmp_path: Path) -> tuple[TestClient, str, str]:
        client = _client(tmp_path, admin_token=ADMIN)
        pid, owner, _ = _project_tokens(client, ADMIN)
        return client, pid, owner

    def test_streamed_read_aborts_over_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, pid, owner = self._setup(tmp_path)
        monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 1000)
        monkeypatch.setattr(server_module, "_UPLOAD_CHUNK_BYTES", 256)
        resp = client.post(
            f"/api/projects/{pid}/sessions/upload",
            files={"file": ("big.vcr", b"x" * 5000)},
            headers=_bearer(owner),
        )
        assert resp.status_code == 413

    def test_content_length_precheck(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client, pid, owner = self._setup(tmp_path)
        monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 10)
        resp = client.post(
            f"/api/projects/{pid}/sessions/upload",
            files={"file": ("big.vcr", b"x" * (server_module._MULTIPART_OVERHEAD_BYTES + 100))},
            headers=_bearer(owner),
        )
        assert resp.status_code == 413

    def test_small_upload_still_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, pid, owner = self._setup(tmp_path)
        monkeypatch.setattr(server_module, "_UPLOAD_CHUNK_BYTES", 64)  # force several chunks
        src = _record(tmp_path / "rec", "small-1")
        with open(src, "rb") as f:
            resp = client.post(
                f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=_bearer(owner)
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["session_id"] == "small-1"


# -- Bug 5: HTTPException is not turned into a 500 -----------------------------


class TestHttpExceptionPassthrough:
    def test_unsupported_export_format_is_400(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.get("/api/sessions/sess-1/export?export_format=csv")
        assert resp.status_code == 400
        assert "Unsupported format" in resp.json()["detail"]

    def test_missing_frame_is_still_404(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        assert client.get("/api/sessions/sess-1/frames/99").status_code == 404
