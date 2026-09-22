"""Cloud HTTP client — sync/push/pull/replay against VCR Cloud (Phase 3).

Stdlib only (``urllib``): no new hard dependencies. Timeouts on every
request; server errors surface as :class:`CloudError` with status codes.

Credentials live in ``$AGENT_VCR_CONFIG_HOME/credentials.json`` (or
``~/.config/agent-vcr/credentials.json``), mode ``0600``:

    vcr cloud login https://vcr.example.com --token-stdin < token.txt
    vcr cloud push ./session.vcr --project refunds
    vcr cloud pull refunds <session-id> -o ./restored.vcr
    vcr cloud replay refunds <session-id>
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class CloudError(Exception):
    """HTTP failure against VCR Cloud (carries the status code)."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"cloud error {status}: {detail}")
        self.status = status
        self.detail = detail


def credentials_path() -> Path:
    """Config-home override first (tests/CI), then XDG/default."""
    base = os.environ.get("AGENT_VCR_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config", "agent-vcr"
    )
    return Path(base) / "credentials.json"


def _load_profiles() -> dict[str, Any]:
    path = credentials_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("profiles", {})
    except (OSError, ValueError) as err:
        raise CloudError(0, f"Cannot read credentials {path}: {err}") from err


def _save_profiles(profiles: dict[str, Any]) -> Path:
    path = credentials_path()
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _atomic_write(
        path,
        (json.dumps({"profiles": profiles}, indent=2) + "\n").encode(),
        mode=0o600,
    )
    return path


def _atomic_write(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    """Durably replace a file without exposing a partial new version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(path, mode)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


class CloudClient:
    """Thin authenticated client for the VCR Cloud API."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # -- auth profiles -------------------------------------------------

    @classmethod
    def from_profile(cls, name: str = "default") -> CloudClient:
        profiles = _load_profiles()
        if name not in profiles:
            raise CloudError(0, f"No cloud profile {name!r} — run `vcr cloud login` first")
        entry = profiles[name]
        return cls(entry["base_url"], entry.get("token"))

    def login(self, profile: str = "default") -> Path:
        """Verify credentials against the server, then persist them.

        Admin/legacy tokens list projects; scoped project tokens get 403
        there — which still proves the token is valid (401 means bad).
        """
        health = self.health()
        if health.get("status") != "ok":
            raise CloudError(0, f"Unhealthy server: {health}")
        try:
            me = self._request("GET", "/api/projects")
        except CloudError as err:
            if err.status != 403:
                raise
            me = {"projects": [], "scoped": True}
        if not isinstance(me, dict) or "projects" not in me:
            raise CloudError(0, f"Unexpected server response: {me}")
        profiles = _load_profiles()
        profiles[profile] = {"base_url": self.base_url, "token": self.token}
        return _save_profiles(profiles)

    # -- transport -----------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"User-Agent": "agent-vcr-cloud/1.2"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data: bytes | None = None
        headers = self._headers()
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                if not payload:
                    return {}
                try:
                    return json.loads(payload.decode())
                except ValueError:
                    return {"raw": payload.decode(errors="replace")}
        except urllib.error.HTTPError as err:
            try:
                detail = json.loads(err.read().decode()).get("detail", err.reason)
            except Exception:
                detail = err.reason
            raise CloudError(err.code, str(detail)) from err
        except OSError as err:
            raise CloudError(0, f"Connection failed: {err}") from err

    # -- API -----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        result = self._request("GET", "/api/health")
        if not isinstance(result, dict):
            raise CloudError(0, f"Bad health response: {result}")
        return result

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/projects").get("projects", [])

    def create_project(
        self, name: str, project_id: str | None = None, description: str = ""
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/api/projects",
            {"name": name, "id": project_id, "description": description},
        )

    def list_sessions(
        self, project: str, limit: int = 100, offset: int = 0, search: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "GET", f"/api/projects/{project}/sessions",
            query={"limit": limit, "offset": offset, "search": search},
        )

    def upload_session(
        self, project: str, path: str | Path, *, overwrite: bool = False
    ) -> dict[str, Any]:
        """Multipart ingest of a ``.vcr`` file; returns the server report."""
        src = Path(path)
        data = src.read_bytes()
        boundary = "----vcrcloudformboundary7MA4YWxkTrZu0gW"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="file"; filename="{src.name}"\r\n'.encode()
        )
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        body.write(data)
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="overwrite"\r\n\r\n')
        body.write(b"true" if overwrite else b"false")
        body.write(f"\r\n--{boundary}--\r\n".encode())
        url = self.base_url + f"/api/projects/{project}/sessions/upload"
        headers = self._headers({"Content-Type": f"multipart/form-data; boundary={boundary}"})
        req = urllib.request.Request(url, data=body.getvalue(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
                if not isinstance(result, dict):
                    raise CloudError(0, f"Bad upload response: {result}")
                return result
        except urllib.error.HTTPError as err:
            try:
                detail = json.loads(err.read().decode()).get("detail", err.reason)
            except Exception:
                detail = err.reason
            raise CloudError(err.code, str(detail)) from err
        except OSError as err:
            raise CloudError(0, f"Connection failed: {err}") from err

    def download_session(
        self, project: str, session_id: str, dest: str | Path
    ) -> Path:
        """Pull + locally integrity-verify; raises on corrupt payload."""
        from agent_vcr.player import VCRPlayer

        url = self.base_url + f"/api/projects/{project}/sessions/{session_id}/download"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as err:
            try:
                detail = json.loads(err.read().decode()).get("detail", err.reason)
            except Exception:
                detail = err.reason
            raise CloudError(err.code, str(detail)) from err
        except OSError as err:
            raise CloudError(0, f"Connection failed: {err}") from err
        out = Path(dest)
        temporary: str | None = None
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                dir=str(out.parent), prefix=f".{out.name}.", suffix=".download"
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            player = VCRPlayer.load(temporary)
        except (ValueError, FileNotFoundError, OSError) as err:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
            raise CloudError(0, f"Downloaded payload is not a valid session: {err}") from err
        try:
            report = player.verify_integrity()
            if not report["ok"]:
                raise CloudError(0, f"Downloaded session failed integrity: {report['issues']}")
            os.replace(temporary, out)
            temporary = None
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
        return out

    def replay(
        self, project: str, session_id: str, mode: str = "full", task: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/api/projects/{project}/sessions/{session_id}/replay",
            {"mode": mode, "task": task},
        )
