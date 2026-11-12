"""Browser sign-in for the bundled web dashboard on a token-protected server.

The built SPA cannot send a token, so the server lets a browser exchange it once for a
short-lived HttpOnly session cookie that /api and /ws/live accept. Cookie-only
state-changing requests must come from the server's own origin (CSRF guard).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import agent_vcr.server as server_module
from agent_vcr.recorder import VCRRecorder
from agent_vcr.server import DASHBOARD_COOKIE, VCRServer

COOKIE = f"{DASHBOARD_COOKIE}_8000"  # per port; the default port in tests

ADMIN = "admin-secret"
AUTH = "auth-secret"
BUNDLE = Path(server_module.__file__).parent / "dashboard" / "assets"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (server_module.AUTH_TOKEN_ENV, server_module.ADMIN_TOKEN_ENV,
                 server_module.ALLOW_UNAUTHENTICATED_ENV):
        monkeypatch.delenv(name, raising=False)


def _server(tmp_path: Path, **kwargs: Any) -> VCRServer:
    vcr_dir = tmp_path / "vcr"
    rec = VCRRecorder(output_dir=str(vcr_dir), auto_save=False)
    rec.start_session("sess-1")
    rec.record_step("step", {"a": 1}, {"b": 2})
    rec.save()
    return VCRServer(vcr_dir=str(vcr_dir), **kwargs)


def _client(tmp_path: Path, base_url: str = "http://testserver", **kwargs: Any) -> TestClient:
    kwargs.setdefault("admin_token", ADMIN)
    return TestClient(_server(tmp_path, **kwargs).app, base_url=base_url,
                      raise_server_exceptions=False)


def _sign_in(client: TestClient, token: str = ADMIN) -> Any:
    return client.get(f"/dashboard?token={token}", follow_redirects=False)


def _origin(client: TestClient) -> dict[str, str]:
    return {"Origin": str(client.base_url).rstrip("/")}


class TestCookieSignIn:
    def test_without_a_session_the_dashboard_sends_you_to_sign_in(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.get("/dashboard/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/dashboard/login"
        form = client.get("/dashboard/login")
        assert form.status_code == 200
        assert '<form method="post" action="/dashboard/login"' in form.text
        assert form.headers["cache-control"] == "no-store"

    @pytest.mark.parametrize("token", [ADMIN, AUTH])
    def test_token_link_sets_a_cookie_and_strips_the_token(self, tmp_path: Path,
                                                           token: str) -> None:
        client = _client(tmp_path, auth_token=AUTH)
        resp = _sign_in(client, token)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/dashboard/"
        assert token not in resp.headers["location"]
        cookie = resp.headers["set-cookie"]
        assert cookie.startswith(f"{COOKIE}=")
        assert token not in cookie  # an opaque signed value, never the token itself
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert "Path=/" in cookie
        assert re.search(r"Max-Age=\d+", cookie)
        assert "Secure" not in cookie  # plain-HTTP request

        assert client.get("/api/sessions").status_code == 200
        assert client.get("/api/sessions/sess-1").json()["session"]["session_id"] == "sess-1"
        page = client.get("/dashboard/")
        assert page.status_code == 200 and '<div id="root">' in page.text

    def test_trailing_slash_link_and_login_form_work_too(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.get(f"/dashboard/?token={ADMIN}", follow_redirects=False)
        assert resp.status_code == 303 and resp.headers["location"] == "/dashboard/"
        other = _client(tmp_path / "b")
        posted = other.post("/dashboard/login", data={"token": ADMIN}, headers=_origin(other),
                            follow_redirects=False)
        assert posted.status_code == 303 and COOKIE in posted.headers["set-cookie"]
        assert other.get("/api/sessions").status_code == 200

    def test_login_page_lets_the_browser_send_its_origin(self, tmp_path: Path) -> None:
        # Under "no-referrer" browsers send `Origin: null` on the form POST, so the
        # same-origin check refused every real sign-in (seen in a real browser).
        client = _client(tmp_path)
        policy = client.get("/dashboard/login").headers.get("referrer-policy", "")
        assert policy and policy != "no-referrer"
        # ...and the token link's redirect still keeps the token out of any Referer.
        assert _sign_in(client).headers["referrer-policy"] == "no-referrer"

    def test_https_requests_get_a_secure_cookie(self, tmp_path: Path) -> None:
        client = _client(tmp_path, base_url="https://vcr.example.test")
        cookie = _sign_in(client).headers["set-cookie"]
        assert "Secure" in cookie
        assert client.get("/api/sessions").status_code == 200

    def test_tls_terminating_proxy_header_marks_the_cookie_secure(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.get(f"/dashboard?token={ADMIN}", headers={"X-Forwarded-Proto": "https"},
                          follow_redirects=False)
        assert "Secure" in resp.headers["set-cookie"]

    @pytest.mark.parametrize("token", ["wrong", "", "tök", ADMIN + "x"])
    def test_wrong_token_sets_nothing(self, tmp_path: Path, token: str) -> None:
        client = _client(tmp_path)
        resp = _sign_in(client, token)
        assert resp.status_code == 401
        assert "set-cookie" not in resp.headers
        assert "Invalid token" in resp.text
        posted = client.post("/dashboard/login", data={"token": token}, follow_redirects=False)
        assert posted.status_code == 401 and "set-cookie" not in posted.headers
        assert client.get("/api/sessions").status_code == 401

    def test_project_tokens_cannot_sign_in(self, tmp_path: Path) -> None:
        client = _client(tmp_path, data_dir=str(tmp_path / "cloud"))
        bearer = {"Authorization": f"Bearer {ADMIN}"}
        pid = client.post("/api/projects", json={"name": "P"}, headers=bearer).json()["id"]
        owner = client.post(f"/api/projects/{pid}/tokens", json={"role": "owner"},
                            headers=bearer).json()["token"]
        assert _sign_in(client, owner).status_code == 401

    def test_forged_and_expired_cookies_are_rejected(self, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        value = client.cookies[COOKIE]
        version, expires, sig = value.split(".")
        client.cookies.set(COOKIE, f"{version}.{int(expires) + 999}.{sig}")
        assert client.get("/api/sessions").status_code == 401
        client.cookies.set(COOKIE, "garbage")
        assert client.get("/api/sessions").status_code == 401

        client.cookies.set(COOKIE, value)
        assert client.get("/api/sessions").status_code == 200
        real_time = server_module.time.time
        monkeypatch.setattr(server_module.time, "time",
                            lambda: real_time() + server_module.DASHBOARD_COOKIE_TTL_S + 1)
        assert client.get("/api/sessions").status_code == 401

    def test_cookie_from_another_server_instance_is_rejected(self, tmp_path: Path) -> None:
        first = _client(tmp_path / "a")
        _sign_in(first)
        second = _client(tmp_path / "b")
        second.cookies.set(COOKIE, first.cookies[COOKIE])
        assert second.get("/api/sessions").status_code == 401

    def test_logout_clears_the_cookie(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        resp = client.get("/dashboard/logout", follow_redirects=False)
        assert resp.status_code == 303 and resp.headers["location"] == "/dashboard/login"
        assert f'{COOKIE}=""' in resp.headers["set-cookie"]
        assert client.get("/api/sessions").status_code == 401

    def test_tokenless_local_mode_needs_no_sign_in(self, tmp_path: Path) -> None:
        client = _client(tmp_path, admin_token=None)
        assert client.get("/dashboard/", follow_redirects=False).status_code == 200
        login = client.get("/dashboard/login", follow_redirects=False)
        assert login.status_code == 303 and login.headers["location"] == "/dashboard/"


class TestCsrfGuard:
    def test_cross_origin_post_with_only_the_cookie_is_rejected(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        body = {"from_frame": 0}
        url = "/api/sessions/sess-1/resume"
        for headers in ({"Origin": "http://evil.example"},
                        {"Origin": "http://testserver:9999"},  # same site, other port
                        {"Origin": "null"},
                        {"Referer": "http://evil.example/page"},
                        {}):  # neither Origin nor Referer
            resp = client.post(url, json=body, headers=headers)
            assert resp.status_code == 403, headers
            assert "origin" in resp.json()["detail"].lower()

        assert client.post(url, json=body, headers=_origin(client)).status_code == 200
        referer = {"Referer": "http://testserver/dashboard/sessions/sess-1"}
        assert client.post(url, json=body, headers=referer).status_code == 200

    def test_header_tokens_are_not_subject_to_the_origin_check(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.post("/api/sessions/sess-1/resume", json={"from_frame": 0},
                           headers={"Authorization": f"Bearer {ADMIN}",
                                    "Origin": "http://evil.example"})
        assert resp.status_code == 200

    def test_allowed_origin_and_forwarded_host_count_as_same_origin(self, tmp_path: Path) -> None:
        client = _client(tmp_path, allowed_origins=["https://vcr.example.com"])
        _sign_in(client)
        url, body = "/api/sessions/sess-1/resume", {"from_frame": 0}
        ok = client.post(url, json=body, headers={"Origin": "https://vcr.example.com"})
        assert ok.status_code == 200
        proxied = client.post(url, json=body, headers={"Origin": "https://proxy.example",
                                                       "X-Forwarded-Host": "proxy.example"})
        assert proxied.status_code == 200

    def test_cross_origin_login_post_is_rejected(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        resp = client.post("/dashboard/login", data={"token": ADMIN},
                           headers={"Origin": "http://evil.example"}, follow_redirects=False)
        assert resp.status_code == 403 and "set-cookie" not in resp.headers


class TestLiveFeedCookie:
    def test_websocket_accepts_the_cookie_from_the_same_origin(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        with client.websocket_connect("/ws/live", headers=_origin(client)) as ws:
            ws.send_json({"action": "ping"})
            assert ws.receive_json()["type"] == "pong"

    @pytest.mark.parametrize("headers", [{"Origin": "http://evil.example"}, {}])
    def test_websocket_rejects_cookie_from_elsewhere(self, tmp_path: Path,
                                                     headers: dict[str, str]) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        with pytest.raises(WebSocketDisconnect), \
                client.websocket_connect("/ws/live", headers=headers) as ws:
            ws.receive_json()

    def test_websocket_without_cookie_still_rejected(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        with pytest.raises(WebSocketDisconnect), \
                client.websocket_connect("/ws/live", headers=_origin(client)) as ws:
            ws.receive_json()


def test_built_bundle_uses_wss_on_https_pages() -> None:
    bundles = list(BUNDLE.glob("index-*.js"))
    assert len(bundles) == 1
    js = bundles[0].read_text(encoding="utf-8")
    assert "`ws://${window.location.host}`" not in js
    assert "${location.protocol===`https:`?`wss`:`ws`}://${window.location.host}" in js


class TestReviewFixes:
    def test_https_cookie_uses_the_host_prefix(self, tmp_path: Path) -> None:
        client = _client(tmp_path, base_url="https://vcr.example.test")
        cookie = _sign_in(client).headers["set-cookie"]
        assert cookie.startswith(f"__Host-{COOKIE}=")
        assert "Secure" in cookie and "Path=/" in cookie and "Domain" not in cookie
        assert client.get("/api/sessions").status_code == 200
        resp = client.get("/dashboard/logout", follow_redirects=False)
        assert any(c.startswith(f'__Host-{COOKIE}=""') and "Secure" in c
                   for c in resp.headers.get_list("set-cookie"))

    def test_two_servers_on_one_host_keep_separate_sessions(self, tmp_path: Path) -> None:
        first = _client(tmp_path / "a", port=8000)
        second = _client(tmp_path / "b", port=8001)
        _sign_in(first)
        _sign_in(second)
        jar = {**dict(first.cookies), **dict(second.cookies)}  # one browser, both cookies
        assert set(jar) == {f"{DASHBOARD_COOKIE}_8000", f"{DASHBOARD_COOKIE}_8001"}
        for client in (first, second):
            client.cookies.clear()
            for name, value in jar.items():
                client.cookies.set(name, value)
            assert client.get("/api/sessions").status_code == 200

    @pytest.mark.parametrize("value", ["tök", "v1.9999999999.ä" + "0" * 63, "v1.1." + "Z" * 64])
    def test_malformed_and_non_ascii_cookies_are_401_not_500(self, tmp_path: Path,
                                                             value: str) -> None:
        client = _client(tmp_path)
        raw: Any = f"{COOKIE}={value}".encode()
        assert client.get("/api/sessions", headers={"cookie": raw}).status_code == 401
        entry = client.get("/dashboard/", headers={"cookie": raw}, follow_redirects=False)
        assert entry.status_code == 303
        with pytest.raises(WebSocketDisconnect), client.websocket_connect(
            "/ws/live", headers={"cookie": raw, **_origin(client)}
        ) as ws:
            ws.receive_json()

    def test_cors_preflight_reaches_the_cors_layer(self, tmp_path: Path) -> None:
        client = _client(tmp_path, allowed_origins=["https://ui.example.com"])
        preflight = {"Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "authorization,content-type"}
        ok = client.options("/api/sessions/sess-1/resume",
                            headers={"Origin": "https://ui.example.com", **preflight})
        assert ok.status_code == 200
        assert ok.headers["access-control-allow-origin"] == "https://ui.example.com"
        refused = client.options("/api/sessions", headers={"Origin": "https://evil.example",
                                                           **preflight})
        assert refused.status_code == 400  # refused by CORS, not a 401
        # A plain OPTIONS (not a preflight) still needs authentication.
        assert client.options("/api/sessions").status_code == 401

    def test_failed_sign_ins_are_rate_limited_per_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _client(tmp_path)
        for _ in range(server_module.SIGN_IN_FAILURE_LIMIT):
            assert _sign_in(client, "wrong").status_code == 401
        blocked = _sign_in(client)  # even the right token, until the window passes
        assert blocked.status_code == 429 and "set-cookie" not in blocked.headers
        assert blocked.headers["retry-after"] == "60"
        posted = client.post("/dashboard/login", data={"token": ADMIN}, headers=_origin(client),
                             follow_redirects=False)
        assert posted.status_code == 429

        real = server_module.time.monotonic
        monkeypatch.setattr(server_module.time, "monotonic",
                            lambda: real() + server_module.SIGN_IN_WINDOW_S + 1)
        assert _sign_in(client).status_code == 303

    def test_successful_sign_ins_are_not_counted(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        for _ in range(server_module.SIGN_IN_FAILURE_LIMIT + 5):
            assert _sign_in(client).status_code == 303


class TestAccessLogRedaction:
    def test_uvicorn_access_record_is_redacted(self) -> None:
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1:5000", "GET", f"/dashboard?token={ADMIN}&x=1", "1.1", 303), None,
        )
        assert server_module.RedactTokenFilter().filter(record)
        message = record.getMessage()
        assert ADMIN not in message
        assert "/dashboard?token=***&x=1" in message

    def test_server_installs_the_filter_on_uvicorn_loggers(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _server(tmp_path, admin_token=ADMIN)
        _server(tmp_path / "again", admin_token=ADMIN)  # idempotent
        for name in ("uvicorn.access", "uvicorn.error"):
            filters = logging.getLogger(name).filters
            assert sum(isinstance(f, server_module.RedactTokenFilter) for f in filters) == 1
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            logging.getLogger("uvicorn.error").info(
                '%s - "WebSocket %s" [accepted]', "127.0.0.1:1", f"/ws/live?token={ADMIN}"
            )
        assert ADMIN not in caplog.text and "token=***" in caplog.text
