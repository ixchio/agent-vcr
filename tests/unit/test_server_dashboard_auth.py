"""Browser sign-in for the bundled web dashboard on a token-protected server.

The built SPA cannot send a token, so the server lets a browser exchange it once for a
short-lived HttpOnly session cookie that /api and /ws/live accept. Cookie-only
state-changing requests must come from the server's own origin (CSRF guard).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import agent_vcr.server as server_module
from agent_vcr.recorder import VCRRecorder
from agent_vcr.server import DASHBOARD_COOKIE, VCRServer

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
        assert cookie.startswith(f"{DASHBOARD_COOKIE}=")
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
        assert posted.status_code == 303 and DASHBOARD_COOKIE in posted.headers["set-cookie"]
        assert other.get("/api/sessions").status_code == 200

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
        value = client.cookies[DASHBOARD_COOKIE]
        version, expires, sig = value.split(".")
        client.cookies.set(DASHBOARD_COOKIE, f"{version}.{int(expires) + 999}.{sig}")
        assert client.get("/api/sessions").status_code == 401
        client.cookies.set(DASHBOARD_COOKIE, "garbage")
        assert client.get("/api/sessions").status_code == 401

        client.cookies.set(DASHBOARD_COOKIE, value)
        assert client.get("/api/sessions").status_code == 200
        real_time = server_module.time.time
        monkeypatch.setattr(server_module.time, "time",
                            lambda: real_time() + server_module.DASHBOARD_COOKIE_TTL_S + 1)
        assert client.get("/api/sessions").status_code == 401

    def test_cookie_from_another_server_instance_is_rejected(self, tmp_path: Path) -> None:
        first = _client(tmp_path / "a")
        _sign_in(first)
        second = _client(tmp_path / "b")
        second.cookies.set(DASHBOARD_COOKIE, first.cookies[DASHBOARD_COOKIE])
        assert second.get("/api/sessions").status_code == 401

    def test_logout_clears_the_cookie(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        _sign_in(client)
        resp = client.get("/dashboard/logout", follow_redirects=False)
        assert resp.status_code == 303 and resp.headers["location"] == "/dashboard/login"
        assert f'{DASHBOARD_COOKIE}=""' in resp.headers["set-cookie"]
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
