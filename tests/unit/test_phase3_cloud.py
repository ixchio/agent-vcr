"""Phase 3 tests: cloud store, Cloud API auth matrix, live end-to-end sync.

Part A runs through the real ASGI app (TestClient). Part B spins a real
uvicorn server on 127.0.0.1 and drives it with CloudClient + `vcr cloud`
over actual HTTP sockets — no mocks anywhere in this file.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import agent_vcr
from agent_vcr.cloud_store import CloudStore
from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder
from agent_vcr.server import VCRServer


def _record(tmp_path, session_id="cloud-sess", n=3):
    rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    rec.start_session(session_id)
    for i in range(n):
        rec.record_step(
            f"step_{i}", {"i": i}, {"result": i * 2},
            metadata=FrameMetadata(model="gpt-4o", cost_usd=0.001,
                                   tokens_used=50, latency_ms=10.0),
        )
    return rec.save()


def _bouquet(tmp_path, admin="admin-secret"):
    server = VCRServer(vcr_dir=str(tmp_path / "legacy"), data_dir=str(tmp_path / "cloud"),
                       admin_token=admin)
    client = TestClient(server.app, raise_server_exceptions=False)
    return server, client


class TestCloudStore:
    def test_project_crud_and_slugs(self, tmp_path):
        store = CloudStore(tmp_path)
        p = store.create_project("Refunds & More")
        assert p.id == "refunds-more"
        assert store.get_project(p.id).name == "Refunds & More"
        assert [x.id for x in store.list_projects()] == [p.id]
        with pytest.raises(FileExistsError):
            store.create_project("other", project_id=p.id)
        with pytest.raises(ValueError):
            store.create_project("x", project_id="../evil")
        with pytest.raises(KeyError):
            store.get_project("missing")

    def test_token_lifecycle(self, tmp_path):
        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        public, secret = store.mint_token("p", "ci", role="owner")
        assert secret.startswith("vcr_")
        assert "token_hash" not in public.model_dump()
        rec = store.verify_token(secret)
        assert rec is not None and rec.role == "owner" and rec.last_used_at
        assert store.verify_token("vcr_bogus") is None
        assert store.verify_token(None) is None
        assert len(store.list_tokens("p")) == 1
        assert store.revoke_token("p", public.id) is True
        assert store.verify_token(secret) is None  # revoked is dead
        assert store.revoke_token("p", public.id) is False
        # persistence across instances, hashes only at rest
        store2 = CloudStore(tmp_path)
        assert store2.verify_token(secret) is None
        raw = json.loads((tmp_path / "tokens.json").read_text())
        assert secret not in json.dumps(raw)

    def test_session_ingest_gate(self, tmp_path):
        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        src = _record(tmp_path / "rec", "sess-1")
        report = store.put_session("p", src.read_bytes())
        assert report == {"session_id": "sess-1", "frames": 3,
                          "sha256": report["sha256"], "deduplicated": False}
        dedup = store.put_session("p", src.read_bytes())
        assert dedup["deduplicated"] is True
        with pytest.raises(FileExistsError):
            store.put_session("p", b'{"type":"session","data":{"session_id":"sess-1"}}\n')
        with pytest.raises(ValueError):
            store.put_session("p", b"garbage")
        with pytest.raises(KeyError):
            store.put_session("nope", src.read_bytes())
        assert store.get_session_bytes("p", "sess-1") == src.read_bytes()
        with pytest.raises(KeyError):
            store.get_session_bytes("p", "missing")
        assert store.delete_session("p", "sess-1") is True
        assert store.delete_session("p", "sess-1") is False

    def test_traversal_guard(self, tmp_path):
        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        with pytest.raises(ValueError):
            store.get_session_bytes("p", "../../etc/passwd")


class TestCloudClientSafety:
    def test_invalid_download_does_not_replace_existing_file(self, tmp_path):
        from agent_vcr.cloud import CloudClient, CloudError

        class InvalidDownload(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(b"not a VCR recording")

            def log_message(self, format, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), InvalidDownload)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        target = tmp_path / "existing.vcr"
        target.write_bytes(b"known-good-local-recording")
        try:
            client = CloudClient(f"http://127.0.0.1:{server.server_port}")
            with pytest.raises(CloudError, match="not a valid session"):
                client.download_session("project", "session", target)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
        assert target.read_bytes() == b"known-good-local-recording"
        assert not list(tmp_path.glob(".existing.vcr.*.download"))


class TestCloudAPI:
    def _setup(self, tmp_path):
        server, client = _bouquet(tmp_path)
        r = client.post("/api/projects", json={"name": "Refunds"},
                        headers={"Authorization": "Bearer admin-secret"})
        assert r.status_code == 201, r.text
        pid = r.json()["id"]
        owner = client.post(f"/api/projects/{pid}/tokens",
                            json={"name": "owner", "role": "owner"},
                            headers={"Authorization": "Bearer admin-secret"}).json()
        member = client.post(f"/api/projects/{pid}/tokens",
                             json={"name": "member", "role": "member"},
                             headers={"Authorization": "Bearer admin-secret"}).json()
        src = _record(tmp_path / "rec", "sess-1")
        return client, pid, owner, member, src

    def test_health_open_and_project_flow(self, tmp_path):
        _, client = _bouquet(tmp_path, admin=None)
        assert client.get("/api/health").json()["status"] == "ok"
        r = client.post("/api/projects", json={"name": "Open"})
        assert r.status_code == 201  # open dev mode

    def test_admin_gate(self, tmp_path):
        _, client = _bouquet(tmp_path)
        assert client.get("/api/projects").status_code == 401
        assert client.get("/api/projects",
                          headers={"x-agent-vcr-token": "admin-secret"}).status_code == 200
        assert client.post("/api/projects", json={"name": "x"}).status_code == 401

    def test_role_matrix(self, tmp_path):
        client, pid, owner, member, src = self._setup(tmp_path)
        OWNER_H = {"Authorization": f"Bearer {owner['token']}"}
        MEMBER_H = {"Authorization": f"Bearer {member['token']}"}
        # anonymous on secured server
        assert client.get(f"/api/projects/{pid}/sessions").status_code == 401
        # member reads, cannot write
        assert client.get(f"/api/projects/{pid}/sessions", headers=MEMBER_H).status_code == 200
        with open(src, "rb") as f:
            denied = client.post(f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=MEMBER_H)
        assert denied.status_code == 403
        # owner writes
        with open(src, "rb") as f:
            up = client.post(f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=OWNER_H)
        assert up.status_code == 200, up.text
        assert up.json()["session_id"] == "sess-1"
        # cross-project token is scoped out
        other = client.post("/api/projects", json={"name": "Other"},
                            headers={"Authorization": "Bearer admin-secret"}).json()["id"]
        cross = client.get(f"/api/projects/{other}/sessions", headers=MEMBER_H)
        assert cross.status_code == 403
        # revoke kills the token
        members = client.get(f"/api/projects/{pid}/tokens", headers=OWNER_H).json()["tokens"]
        mid = next(t["id"] for t in members if t["name"] == "member")
        assert client.post(f"/api/projects/{pid}/tokens/{mid}/revoke", headers=OWNER_H).status_code == 200
        assert client.get(f"/api/projects/{pid}/sessions", headers=MEMBER_H).status_code == 401

    def test_upload_conflict_dedup_corrupt(self, tmp_path):
        client, pid, owner, _, src = self._setup(tmp_path)
        OWNER_H = {"Authorization": f"Bearer {owner['token']}"}
        with open(src, "rb") as f:
            assert client.post(f"/api/projects/{pid}/sessions/upload",
                               files={"file": f}, headers=OWNER_H).status_code == 200
        with open(src, "rb") as f:
            again = client.post(f"/api/projects/{pid}/sessions/upload",
                                files={"file": f}, headers=OWNER_H).json()
        assert again["deduplicated"] is True
        clash = b'{"type":"session","data":{"session_id":"sess-1"}}\n'
        r = client.post(f"/api/projects/{pid}/sessions/upload",
                        files={"file": ("x.vcr", clash)}, headers=OWNER_H)
        assert r.status_code == 409
        r = client.post(f"/api/projects/{pid}/sessions/upload",
                        files={"file": ("x.vcr", clash)},
                        data={"overwrite": "true"}, headers=OWNER_H)
        assert r.status_code == 200
        bad = client.post(f"/api/projects/{pid}/sessions/upload",
                          files={"file": ("x.vcr", b"not jsonl")}, headers=OWNER_H)
        assert bad.status_code == 422

    def test_pull_replay_delete(self, tmp_path):
        client, pid, owner, _, src = self._setup(tmp_path)
        OWNER_H = {"Authorization": f"Bearer {owner['token']}"}
        with open(src, "rb") as f:
            client.post(f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=OWNER_H)
        detail = client.get(f"/api/projects/{pid}/sessions/sess-1", headers=OWNER_H).json()
        assert detail["statistics"]["total_frames"] == 3
        dl = client.get(f"/api/projects/{pid}/sessions/sess-1/download", headers=OWNER_H)
        assert dl.status_code == 200 and dl.content == src.read_bytes()
        rep = client.post(f"/api/projects/{pid}/sessions/sess-1/replay",
                          json={"mode": "full"}, headers=OWNER_H).json()
        assert rep["ledger"]["steps_replayed"] == 3
        assert [o["result"] for o in rep["outputs"]] == [0, 2, 4]
        assert rep["ledger"]["saved"]["percent"] == 100.0
        assert client.post(f"/api/projects/{pid}/sessions/sess-1/replay",
                           json={"mode": "hybrid"}, headers=OWNER_H).status_code == 400
        assert client.delete(f"/api/projects/{pid}/sessions/sess-1", headers=OWNER_H).status_code == 200
        assert client.get(f"/api/projects/{pid}/sessions/sess-1", headers=OWNER_H).status_code == 404

    def test_listing_pagination_search(self, tmp_path):
        client, pid, owner, _, _ = self._setup(tmp_path)
        OWNER_H = {"Authorization": f"Bearer {owner['token']}"}
        for i in range(3):
            src = _record(tmp_path / f"r{i}", f"sess-{i}")
            with open(src, "rb") as f:
                client.post(f"/api/projects/{pid}/sessions/upload", files={"file": f}, headers=OWNER_H)
        page = client.get(f"/api/projects/{pid}/sessions?limit=2", headers=OWNER_H).json()
        assert page["total"] == 3 and len(page["sessions"]) == 2
        page2 = client.get(f"/api/projects/{pid}/sessions?limit=2&offset=2", headers=OWNER_H).json()
        assert len(page2["sessions"]) == 1
        found = client.get(f"/api/projects/{pid}/sessions?search=sess-1", headers=OWNER_H).json()
        assert found["total"] == 1

    def test_legacy_token_unlocks_cloud(self, tmp_path):
        server = VCRServer(vcr_dir=str(tmp_path / "legacy"), data_dir=str(tmp_path / "cloud"),
                           auth_token="legacy-secret")
        client = TestClient(server.app, raise_server_exceptions=False)
        H = {"x-agent-vcr-token": "legacy-secret"}
        assert client.post("/api/projects", json={"name": "L"} , headers=H).status_code == 201

    def test_public_api(self):
        assert agent_vcr.__version__ == "1.5.1"
        for name in ("CloudStore", "Project", "ApiTokenPublic", "CloudClient", "CloudError"):
            assert name in agent_vcr.__all__


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLiveEndToEnd:
    """Real uvicorn + real sockets + CloudClient + `vcr cloud` CLI."""

    @pytest.fixture
    def live(self, tmp_path):
        import uvicorn

        server = VCRServer(vcr_dir=str(tmp_path / "legacy"),
                           data_dir=str(tmp_path / "cloud"),
                           admin_token="live-admin")
        port = _free_port()
        config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
        uv = uvicorn.Server(config)
        thread = threading.Thread(target=uv.run, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                    if json.loads(r.read())["status"] == "ok":
                        break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("live server did not start")
        yield base
        uv.should_exit = True
        thread.join(timeout=10)

    def test_sync_push_pull_replay_over_http(self, live, tmp_path):
        from agent_vcr.cloud import CloudClient

        admin = CloudClient(live, "live-admin")
        pid = admin.create_project("Live")["id"]
        assert admin.health()["auth"] is True

        src = _record(tmp_path / "rec", "live-sess")
        # mint a scoped owner token via the admin API directly
        with urllib.request.urlopen(urllib.request.Request(
            live + f"/api/projects/{pid}/tokens",
            data=json.dumps({"name": "sync", "role": "owner"}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer live-admin"},
        )) as r:
            scoped = json.loads(r.read())["token"]

        user = CloudClient(live, scoped)
        report = user.upload_session(pid, src)
        assert report["session_id"] == "live-sess" and report["frames"] == 3
        assert user.list_sessions(pid)["total"] == 1
        pulled = user.download_session(pid, "live-sess", tmp_path / "pulled.vcr")
        assert pulled.read_bytes() == src.read_bytes()
        rep = user.replay(pid, "live-sess")
        assert [o["result"] for o in rep["outputs"]] == [0, 2, 4]

        # wrong token -> 401 over the wire
        from agent_vcr.cloud import CloudError

        with pytest.raises(CloudError) as exc:
            CloudClient(live, "vcr_bogus").list_sessions(pid)
        assert exc.value.status == 401

    def test_cli_cloud_against_live(self, live, tmp_path, monkeypatch, capsys):
        import os

        from agent_vcr.cli import main

        monkeypatch.setenv("AGENT_VCR_CONFIG_HOME", str(tmp_path / "cfg"))
        src = _record(tmp_path / "rec", "cli-sess")
        main(["cloud", "login", live, "--token", "live-admin"])
        capsys.readouterr()
        main(["cloud", "status"])
        assert json.loads(capsys.readouterr().out)["status"] == "ok"
        # create project via client, sync the rest via CLI profile
        from agent_vcr.cloud import CloudClient

        pid = CloudClient(live, "live-admin").create_project("CLI")["id"]
        main(["cloud", "push", str(src), "--project", pid])
        pushed = json.loads(capsys.readouterr().out)
        assert pushed["session_id"] == "cli-sess"
        main(["cloud", "pull", pid, "cli-sess", "-o", str(tmp_path / "out.vcr")])
        assert (tmp_path / "out.vcr").read_bytes() == src.read_bytes()
        assert os.path.exists(tmp_path / "cfg" / "credentials.json")


class TestBugFixes:
    def test_scoped_token_login_accepted_live(self, tmp_path, monkeypatch):
        """End-to-end over real HTTP: a project-scoped token gets 403 on
        /api/projects — login must accept that as proof of validity."""
        import urllib.request

        import uvicorn

        from agent_vcr.cloud import CloudClient

        server = VCRServer(vcr_dir=str(tmp_path / "legacy"),
                           data_dir=str(tmp_path / "cloud"),
                           admin_token="live-admin")
        port = _free_port()
        config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
        uv = uvicorn.Server(config)
        thread = threading.Thread(target=uv.run, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=2):
                    break
            except OSError:
                time.sleep(0.1)
        try:
            monkeypatch.setenv("AGENT_VCR_CONFIG_HOME", str(tmp_path / "cfg"))
            admin = CloudClient(base, "live-admin")
            pid = admin.create_project("Scoped")["id"]
            with urllib.request.urlopen(urllib.request.Request(
                base + f"/api/projects/{pid}/tokens",
                data=json.dumps({"name": "m", "role": "member"}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer live-admin"},
            )) as r:
                scoped = json.loads(r.read())["token"]
            # scoped token really is forbidden from the admin listing...
            from agent_vcr.cloud import CloudError

            with pytest.raises(CloudError) as exc:
                CloudClient(base, scoped).list_projects()
            assert exc.value.status == 403
            # ...yet login succeeds and persists the profile
            path = CloudClient(base, scoped).login(profile="scoped")
            assert json.loads(path.read_text())["profiles"]["scoped"]["token"] == scoped
            # stored profile round-trips
            assert CloudClient.from_profile("scoped").base_url == base
        finally:
            uv.should_exit = True
            thread.join(timeout=10)

    def test_cli_cloud_errors_exit_cleanly(self, tmp_path, monkeypatch, capsys):
        from agent_vcr.cli import main

        monkeypatch.setenv("AGENT_VCR_CONFIG_HOME", str(tmp_path / "empty-cfg"))
        with pytest.raises(SystemExit) as exc:
            main(["cloud", "pull", "p", "s"])
        assert exc.value.code == 1
        assert "Error" in capsys.readouterr().err

    def test_concurrent_token_verify_keeps_store_intact(self, tmp_path):
        import threading

        store = CloudStore(tmp_path)
        store.create_project("p", project_id="p")
        _, secret = store.mint_token("p", "race", role="owner")
        errors: list = []

        def hammer():
            try:
                for _ in range(25):
                    assert store.verify_token(secret) is not None
            except Exception as err:  # noqa: BLE001
                errors.append(err)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(store.list_tokens("p")) == 1
        json.loads((tmp_path / "tokens.json").read_text())  # still parses
