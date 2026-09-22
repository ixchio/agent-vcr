"""Cloud object store — projects, scoped tokens, execution blobs (Phase 3).

On-disk layout under ``<root>/``::

    projects.json                 # project registry
    tokens.json                   # token records (sha256 hashes ONLY, never secrets)
    projects/<pid>/sessions/<sid>.vcr   # execution objects (Phase 0 JSONL)

Security properties (all enforced, all tested):
- Token secrets exist in plaintext exactly once: at mint time. At rest
  only sha256 hashes are stored; verification uses ``hmac.compare_digest``.
- ``project_id`` / ``session_id`` values are slug-validated AND resolved
  paths are confined with ``is_relative_to`` (defense in depth).
- Roles: ``owner`` (read+write) vs ``member`` (read-only).
- Ingest gate: uploads are parsed + ``verify_integrity`` checked BEFORE
  they become visible; corrupt payloads never land in the store.

No network, no mocks — pure local durability. The FastAPI layer
(``server.py``) and the HTTP client (``cloud.py``) build on this.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"^[a-zA-Z0-9_.-]+$")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per execution object

Role = Literal["owner", "member"]


class Project(BaseModel):
    """A tenant namespace: sessions, tokens, datasets live under it."""

    id: str
    name: str
    description: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ApiTokenRecord(BaseModel):
    """Persisted token metadata — hash only, never the secret."""

    id: str
    project_id: str
    name: str
    role: Role = "member"
    prefix: str
    token_hash: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used_at: str | None = None
    revoked: bool = False


class ApiTokenPublic(BaseModel):
    """Safe listing view (no hash)."""

    id: str
    project_id: str
    name: str
    role: Role
    prefix: str
    created_at: str
    last_used_at: str | None = None
    revoked: bool = False


def slugify(name: str) -> str:
    """Derive a safe id from a display name (lowercase, dashes)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot derive project id from {name!r}")
    return slug


def _check_slug(value: str, kind: str) -> str:
    if not _SLUG.fullmatch(value):
        raise ValueError(f"Invalid {kind} {value!r}: use letters, digits, _, -, .")
    return value


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as err:
        raise ValueError(f"Corrupt store file {path}: {err}") from err


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def _store_lock(root: Path) -> Iterator[None]:
    """Inter-process mutex for registry read-modify-write cycles.

    The server verifies tokens (and flushes ``last_used_at``) on every
    request, so concurrent workers would otherwise lose each other's
    updates. POSIX ``flock``; fail-open where unavailable. Body
    exceptions always propagate (only lock-acquisition may degrade).
    """
    try:
        import fcntl
    except ImportError:  # Windows: single-process assumption
        yield
        return
    try:
        fh = open(root / ".lock", "a+b")  # noqa: PTH123, SIM115
    except OSError:
        logger.debug("lock file unavailable, proceeding unlocked", exc_info=True)
        yield
        return
    with fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError:
            logger.debug("flock unavailable, proceeding unlocked", exc_info=True)
            locked = False
        try:
            yield
        finally:
            if locked:
                with contextlib.suppress(OSError):
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class CloudStore:
    """Project registry + token vault + execution object store."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "projects").mkdir(exist_ok=True)

    # -- projects ------------------------------------------------------

    def create_project(
        self, name: str, project_id: str | None = None, description: str = ""
    ) -> Project:
        pid = _check_slug(project_id, "project id") if project_id else slugify(name)
        with _store_lock(self.root):
            projects = self._projects()
            if any(p["id"] == pid for p in projects):
                raise FileExistsError(f"Project {pid!r} already exists")
            project = Project(id=pid, name=name, description=description)
            projects.append(project.model_dump())
            _atomic_json(self.root / "projects.json", {"projects": projects})
        (self.root / "projects" / pid / "sessions").mkdir(parents=True, exist_ok=True)
        logger.info("Created project %s", pid)
        return project

    def get_project(self, project_id: str) -> Project:
        _check_slug(project_id, "project id")
        for p in self._projects():
            if p["id"] == project_id:
                return Project(**p)
        raise KeyError(f"Project {project_id!r} not found")

    def list_projects(self) -> list[Project]:
        return [Project(**p) for p in self._projects()]

    def _projects(self) -> list[dict]:
        return _read_json(self.root / "projects.json", {"projects": []})["projects"]

    # -- tokens --------------------------------------------------------

    def mint_token(self, project_id: str, name: str, role: Role = "member") -> tuple[ApiTokenPublic, str]:
        """Mint a token; returns (public record, plaintext secret — shown once)."""
        self.get_project(project_id)  # validates existence
        if role not in ("owner", "member"):
            raise ValueError(f"Invalid role {role!r}")
        secret = f"vcr_{secrets.token_urlsafe(8)}_{secrets.token_urlsafe(32)}"
        record = ApiTokenRecord(
            id=uuid.uuid4().hex[:12],
            project_id=project_id,
            name=name,
            role=role,
            prefix=secret.split("_")[1],
            token_hash=hashlib.sha256(secret.encode()).hexdigest(),
        )
        with _store_lock(self.root):
            tokens = self._tokens()
            tokens.append(record.model_dump())
            _atomic_json(self.root / "tokens.json", {"tokens": tokens})
        public = ApiTokenPublic(**{k: v for k, v in record.model_dump().items() if k != "token_hash"})
        return public, secret

    def verify_token(self, supplied: str | None) -> ApiTokenRecord | None:
        """Constant-time token check; None when missing/unknown/revoked."""
        if not supplied:
            return None
        digest = hashlib.sha256(supplied.encode()).hexdigest()
        with _store_lock(self.root):
            tokens = self._tokens()
            for raw in tokens:
                if not hmac.compare_digest(raw.get("token_hash", ""), digest):
                    continue
                record = ApiTokenRecord(**raw)
                if record.revoked:
                    return None
                record.last_used_at = _utcnow()
                raw["last_used_at"] = record.last_used_at
                with contextlib.suppress(Exception):
                    _atomic_json(self.root / "tokens.json", {"tokens": tokens})
                return record
        return None

    def list_tokens(self, project_id: str) -> list[ApiTokenPublic]:
        self.get_project(project_id)
        return [
            ApiTokenPublic(**{k: v for k, v in t.items() if k != "token_hash"})
            for t in self._tokens()
            if t["project_id"] == project_id
        ]

    def revoke_token(self, project_id: str, token_id: str) -> bool:
        with _store_lock(self.root):
            tokens = self._tokens()
            for raw in tokens:
                if raw["id"] == token_id and raw["project_id"] == project_id:
                    if raw.get("revoked"):
                        return False
                    raw["revoked"] = True
                    _atomic_json(self.root / "tokens.json", {"tokens": tokens})
                    return True
        return False

    def has_credentials(self) -> bool:
        """True once any active token exists (server then requires auth)."""
        return any(not t.get("revoked", False) for t in self._tokens())

    def _tokens(self) -> list[dict]:
        return _read_json(self.root / "tokens.json", {"tokens": []})["tokens"]

    # -- sessions (execution objects) ----------------------------------

    def _sessions_dir(self, project_id: str) -> Path:
        self.get_project(project_id)
        d = self._confined(self.root / "projects" / project_id / "sessions")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _confined(self, path: Path) -> Path:
        """Resolve and confine under the store root (path-traversal guard)."""
        resolved = (self.root / path.relative_to(self.root)).resolve() if not path.is_absolute() else path.resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as err:
            raise ValueError(f"Path escapes store root: {path}") from err
        return resolved

    def put_session(self, project_id: str, data: bytes, *, overwrite: bool = False) -> dict[str, Any]:
        """Ingest gate: parse + integrity-verify BEFORE the object is visible.

        Returns ``{session_id, frames, sha256, deduplicated}``.
        Raises ``ValueError`` (corrupt), ``FileExistsError`` (conflict).
        """
        from agent_vcr.player import VCRPlayer

        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
        sessions = self._sessions_dir(project_id)

        fd, tmp = tempfile.mkstemp(dir=str(sessions), suffix=".upload")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            try:
                player = VCRPlayer.load(tmp)
            except (ValueError, FileNotFoundError) as err:
                raise ValueError(f"Corrupt session upload: {err}") from err
            integrity = player.verify_integrity()
            if not integrity["ok"]:
                raise ValueError(f"Upload failed integrity: {integrity['issues']}")
            sid = _check_slug(player.session.session_id, "session id")
            dest = self._confined(sessions / f"{sid}.vcr")
            sha = _sha256_bytes(data)
            if dest.exists():
                if _sha256_bytes(dest.read_bytes()) == sha:
                    return {
                        "session_id": sid,
                        "frames": len(player.frames),
                        "sha256": sha,
                        "deduplicated": True,
                    }
                if not overwrite:
                    raise FileExistsError(f"Session {sid!r} already exists (use overwrite)")
            os.replace(tmp, str(dest))
            return {
                "session_id": sid,
                "frames": len(player.frames),
                "sha256": sha,
                "deduplicated": False,
            }
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)

    def get_session_bytes(self, project_id: str, session_id: str) -> bytes:
        _check_slug(session_id, "session id")
        path = self._confined(self._sessions_dir(project_id) / f"{session_id}.vcr")
        if not path.exists():
            raise KeyError(f"Session {session_id!r} not found")
        return path.read_bytes()

    def delete_session(self, project_id: str, session_id: str) -> bool:
        _check_slug(session_id, "session id")
        path = self._confined(self._sessions_dir(project_id) / f"{session_id}.vcr")
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_session_headers(
        self,
        project_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> dict[str, Any]:
        """Fast listing: parse only the first JSONL line of each object."""
        sessions = self._sessions_dir(project_id)
        items: list[dict] = []
        skipped = 0
        for path in sorted(sessions.glob("*.vcr")):
            try:
                with open(path) as f:
                    first = f.readline().strip()
                record = json.loads(first)
                if record.get("type") != "session":
                    raise ValueError("missing session header")
                data = record.get("data", {})
                data.setdefault("session_id", path.stem)
                items.append(data)
            except (OSError, ValueError, json.JSONDecodeError):
                skipped += 1
        if search:
            q = search.lower()
            items = [
                s for s in items
                if q in s.get("session_id", "").lower()
                or any(q in t.lower() for t in s.get("tags", []))
            ]
        total = len(items)
        return {
            "sessions": items[offset:offset + limit],
            "total": total,
            "skipped": skipped,
        }
