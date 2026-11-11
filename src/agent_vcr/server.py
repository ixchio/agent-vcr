"""FastAPI server for VCR - serves recordings and handles live updates."""

from __future__ import annotations

import asyncio
import hmac
import http.client
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import sys
import urllib.parse
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from agent_vcr import __version__
from agent_vcr.cloud_store import MAX_UPLOAD_BYTES, CloudStore
from agent_vcr.hosted import (
    HostedAuthenticationError,
    HostedBillingError,
    HostedConfigurationError,
    HostedConflictError,
    HostedControlPlane,
    HostedDependencyError,
    HostedUser,
)
from agent_vcr.models import Frame, Session
from agent_vcr.player import VCRPlayer

_SERVER_IMPORT_ERROR: ImportError | None = None

try:
    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
        UploadFile,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError as exc:
    _SERVER_IMPORT_ERROR = exc

    class FileSystemEventHandler:  # type: ignore[no-redef]
        """Fallback base so importing agent_vcr.server can show a helpful error."""

        pass
logger = logging.getLogger(__name__)

_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Environment variables that provide defaults for the matching CLI flags.
AUTH_TOKEN_ENV = "AGENT_VCR_AUTH_TOKEN"
ADMIN_TOKEN_ENV = "AGENT_VCR_ADMIN_TOKEN"
ALLOW_UNAUTHENTICATED_ENV = "AGENT_VCR_ALLOW_UNAUTHENTICATED"
# Opt-in for workflow callbacks to loopback/private/link-local hosts (SSRF guard).
ALLOW_PRIVATE_CALLBACKS_ENV = "AGENT_VCR_ALLOW_PRIVATE_CALLBACKS"

# /api paths that never require a VCR token: the platform health check, and the
# hosted namespace, which authenticates with Supabase bearer tokens / Stripe
# signatures on its own.
_AUTH_EXEMPT_PATHS = frozenset({"/api/health"})
_AUTH_EXEMPT_PREFIXES = ("/api/hosted/",)
# The only /api namespace a project-scoped token may reach; each route under it
# re-checks that the token belongs to the requested project.
_PROJECT_SCOPED_PREFIX = "/api/projects/"
_IDENTITY_SCOPE_KEY = "agent_vcr.identity"

_UPLOAD_CHUNK_BYTES = 1024 * 1024
_UPLOAD_READ_TIMEOUT_S = 30.0
# Slack for multipart framing when pre-checking Content-Length against the limit.
_MULTIPART_OVERHEAD_BYTES = 64 * 1024

_STORE_TIMEOUT_S = 30.0
_HOSTED_AUTH_TIMEOUT_S = 7.0
_STORE_UPLOAD_TIMEOUT_S = 120.0
_REPLAY_TIMEOUT_S = 300.0
_WS_SEND_TIMEOUT_S = 5.0
_SHUTDOWN_GRACE_S = 10.0

_CALLBACK_RESOLVE_TIMEOUT_S = 5.0
_CALLBACK_POST_TIMEOUT_S = 10.0
_CALLBACK_MAX_URL_LENGTH = 2048

_T = TypeVar("_T")


def is_loopback_host(host: str) -> bool:
    """True when ``host`` only accepts connections from this machine."""
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def bind_safety_problem(host: str, auth_token: str | None, admin_token: str | None) -> str | None:
    """Explain why binding ``host`` without any token is unsafe, or None if it is fine."""
    if auth_token or admin_token or is_loopback_host(host):
        return None
    return (
        f"Refusing to serve the Agent VCR API on non-loopback host {host!r} without "
        f"authentication. Set {AUTH_TOKEN_ENV} and/or {ADMIN_TOKEN_ENV} (or pass "
        "--auth-token/--admin-token). To expose an unauthenticated server anyway, pass "
        f"--allow-unauthenticated or set {ALLOW_UNAUTHENTICATED_ENV}=1."
    )


def _env_token(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _tokens_equal(supplied: str, expected: str | None) -> bool:
    """Constant-time comparison that also tolerates non-ASCII input."""
    if not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _header_tokens(headers: Any) -> list[str]:
    """Token candidates from the ``x-agent-vcr-token`` and bearer headers."""
    candidates: list[str] = []
    header_token = headers.get("x-agent-vcr-token")
    if header_token:
        candidates.append(header_token.strip())
    authorization = headers.get("authorization")
    if authorization and authorization.lower().startswith("bearer "):
        candidates.append(authorization[7:].strip())
    return [c for c in candidates if c]


async def _read_upload_limited(file: Any, limit: int) -> bytes:
    """Read an upload in chunks, aborting with 413 as soon as ``limit`` is exceeded."""
    buffer = bytearray()
    while True:
        try:
            chunk = await asyncio.wait_for(
                file.read(_UPLOAD_CHUNK_BYTES), timeout=_UPLOAD_READ_TIMEOUT_S
            )
        except asyncio.TimeoutError as err:
            raise HTTPException(status_code=408, detail="Upload read timed out") from err
        if not chunk:
            return bytes(buffer)
        if len(buffer) + len(chunk) > limit:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
        buffer.extend(chunk)


@dataclass(frozen=True)
class CallbackTarget:
    """A validated workflow callback: where to connect and what to send as Host/SNI."""

    scheme: str
    host: str
    port: int
    path: str
    address: str

    @property
    def origin(self) -> str:
        """``scheme://host`` only: safe to log (no path, query or credentials)."""
        return f"{self.scheme}://{self.host}"


def callback_origin(url: str) -> str:
    """Best-effort ``scheme://host`` of a URL for logs; never the path or query."""
    try:
        parsed = urllib.parse.urlsplit(url)
        return f"{parsed.scheme}://{parsed.hostname or '?'}"
    except ValueError:
        return "<invalid url>"


def _is_public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # is_global excludes loopback, RFC 1918, link-local (169.254.169.254 cloud
    # metadata), CGNAT, unspecified and reserved ranges.
    return ip.is_global and not ip.is_multicast


def resolve_callback_url(url: str, *, allow_private: bool) -> CallbackTarget:
    """Validate a workflow ``callback_url`` and pin it to one resolved address.

    Blocking (DNS): call it from a worker thread. Raises ``ValueError`` with a
    message that is safe to return to the client.
    """
    if len(url) > _CALLBACK_MAX_URL_LENGTH:
        raise ValueError("callback_url is too long")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as err:
        raise ValueError("callback_url is not a valid URL") from err
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("callback_url must be an http:// or https:// URL")
    host = parsed.hostname
    if not host:
        raise ValueError("callback_url must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("callback_url must not contain credentials")
    if port is None:
        port = 443 if scheme == "https" else 80
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as err:
        raise ValueError(f"callback_url host {host!r} cannot be resolved") from err
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ValueError(f"callback_url host {host!r} cannot be resolved")
    if not allow_private and not all(_is_public_address(a) for a in addresses):
        raise ValueError(
            "callback_url resolves to a loopback, private, link-local or metadata address; "
            f"set {ALLOW_PRIVATE_CALLBACKS_ENV}=1 on the server to allow it"
        )
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return CallbackTarget(scheme=scheme, host=host, port=port, path=path, address=addresses[0])


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection to a pre-validated address (no second DNS lookup, no proxies)."""

    def __init__(self, target: CallbackTarget, timeout: float):
        super().__init__(target.host, target.port, timeout=timeout)
        self._pinned_address = target.address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a pre-validated address; the certificate is checked against the host name."""

    def __init__(self, target: CallbackTarget, timeout: float):
        context = ssl.create_default_context()
        super().__init__(target.host, target.port, timeout=timeout, context=context)
        self._pinned_address = target.address
        self._tls_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._pinned_address, self.port), self.timeout)
        self.sock = self._tls_context.wrap_socket(sock, server_hostname=self.host)


def post_callback(target: CallbackTarget, body: bytes, timeout: float) -> int:
    """POST JSON to a validated callback target; returns the HTTP status (no redirects)."""
    connection: http.client.HTTPConnection
    if target.scheme == "https":
        connection = _PinnedHTTPSConnection(target, timeout)
    else:
        connection = _PinnedHTTPConnection(target, timeout)
    try:
        connection.request(
            "POST",
            target.path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"agent-vcr/{__version__}",
            },
        )
        response = connection.getresponse()
        response.read(64 * 1024)
        return response.status
    finally:
        connection.close()


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[Session]
    total: int


class SessionDetailResponse(BaseModel):
    """Response for session details."""

    session: Session
    frames: list[Frame]
    statistics: dict


class WorkflowTriggerRequest(BaseModel):
    inputs: dict[str, Any] = {}
    secret: str = ""
    callback_url: str = ""

class ResumeRequest(BaseModel):
    """Request to resume execution from a frame."""

    from_frame: int
    state_overrides: dict = {}
    mode: str = "fork"
    new_session_id: str | None = None


class ReplayRequest(BaseModel):
    """Request to replay execution from a session."""

    mode: str = "full"
    task: str | None = None


class VCRFileWatcher(FileSystemEventHandler):
    """Watches .vcr files and broadcasts updates via WebSocket."""

    def __init__(self, vcr_dir: Path):
        self.vcr_dir = vcr_dir
        self.connections: list[WebSocket] = []
        # Created lazily inside the running loop (Python 3.9 binds a Lock to the
        # loop current at construction time, which is not uvicorn's loop).
        self._lock_obj: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Strong references so scheduled broadcasts cannot be garbage-collected.
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock_obj is None or self._lock_loop is not loop:
            self._lock_obj = asyncio.Lock()
            self._lock_loop = loop
        return self._lock_obj

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the running event loop for thread-safe async scheduling."""
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """Add a new WebSocket connection."""
        async with self._lock:
            self.connections.append(websocket)
            logger.info("WebSocket client connected (%d total)", len(self.connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)
                logger.info("WebSocket client disconnected (%d remaining)", len(self.connections))

    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to all connected clients.

        The connection list is snapshotted under the lock; sends happen outside
        it (with a timeout) so one slow client cannot block connects/disconnects.
        """
        async with self._lock:
            targets = list(self.connections)

        async def _send(conn: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(conn.send_json(message), timeout=_WS_SEND_TIMEOUT_S)
            except Exception:
                return conn
            return None

        results = await asyncio.gather(*(_send(conn) for conn in targets))
        for conn in results:
            if conn is not None:
                await self.disconnect(conn)

    def _spawn_broadcast(self, message: dict) -> None:
        task = asyncio.ensure_future(self.broadcast(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _schedule_broadcast(self, message: dict) -> None:
        """Thread-safe broadcast scheduling from watchdog threads."""
        if self._loop is None or self._loop.is_closed():
            logger.warning("No event loop available for broadcast")
            return
        self._loop.call_soon_threadsafe(self._spawn_broadcast, message)

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if not event.src_path.endswith(".vcr"):
            return

        filepath = Path(event.src_path)
        session_id = filepath.stem

        try:
            player = VCRPlayer.load(filepath)
            last_frame = player.frames[-1] if player.frames else None

            if last_frame:
                message = {
                    "type": "frame_update",
                    "session_id": session_id,
                    "frame": last_frame.model_dump(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._schedule_broadcast(message)
        except Exception as e:
            logger.debug("Error processing file modification for %s: %s", session_id, e)

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if not event.src_path.endswith(".vcr"):
            return

        filepath = Path(event.src_path)
        session_id = filepath.stem

        try:
            player = VCRPlayer.load(filepath)

            message = {
                "type": "session_created",
                "session_id": session_id,
                "session": player.session.model_dump(),
                "frame_count": len(player.frames),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._schedule_broadcast(message)
        except Exception as e:
            logger.debug("Error processing file creation for %s: %s", session_id, e)


class VCRServer:
    """FastAPI server for VCR dashboard."""

    def __init__(
        self,
        vcr_dir: str = ".vcr",
        host: str = "127.0.0.1",
        port: int = 8000,
        allowed_origins: list[str] | None = None,
        auth_token: str | None = None,
        data_dir: str | None = None,
        admin_token: str | None = None,
        workflow_handlers: str | None = None,
        hosted: HostedControlPlane | None = None,
        allow_unauthenticated: bool = False,
        allow_private_callbacks: bool | None = None,
    ):
        if _SERVER_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The dashboard server dependencies are not installed. "
                'Install them with: pip install "ai-agent-vcr[dashboard]"'
            ) from _SERVER_IMPORT_ERROR

        self.vcr_dir = Path(vcr_dir)
        self.host = host
        self.port = port
        self.auth_token = auth_token or None
        self.data_dir = data_dir
        self.admin_token = admin_token or None
        self.allow_unauthenticated = allow_unauthenticated
        self.workflow_handlers = workflow_handlers
        self.hosted = hosted
        self.allow_private_callbacks = (
            _env_flag(ALLOW_PRIVATE_CALLBACKS_ENV)
            if allow_private_callbacks is None
            else allow_private_callbacks
        )
        self.watcher = VCRFileWatcher(self.vcr_dir)
        self.observer: Any | None = None
        # Strong references to fire-and-forget work (workflow callbacks).
        self._background_tasks: set[asyncio.Task[None]] = set()

        self.vcr_dir.mkdir(parents=True, exist_ok=True)

        self.app = FastAPI(
            title="Agent VCR API",
            description="API for Agent VCR - The DVR for AI Agents",
            version=__version__,
            lifespan=self._lifespan,
        )

        if allowed_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        else:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=[
                    f"http://127.0.0.1:{self.port}",
                    f"http://localhost:{self.port}",
                ],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        self._install_auth_middleware()

        # Serve the built React dashboard if it exists
        dashboard_dir = Path(__file__).parent / "dashboard"
        if dashboard_dir.is_dir() and any(dashboard_dir.iterdir()):
            self.app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")
            self._has_dashboard = True
        else:
            self._has_dashboard = False

        self._setup_routes()
        self._setup_workflow_routes()
        self._setup_cloud_routes()
        self._setup_intel_routes()
        if self.hosted:
            self._setup_hosted_routes()

    def _cloud_store(self) -> CloudStore:
        return CloudStore(self.data_dir or self.vcr_dir)

    async def _store_io(
        self, func: Callable[[CloudStore], _T], *, timeout: float = _STORE_TIMEOUT_S
    ) -> _T:
        """Run blocking CloudStore work (file locks, JSON rewrites) off the event loop."""

        def _call() -> _T:
            return func(self._cloud_store())

        return await self._offload(_call, timeout=timeout)

    @staticmethod
    async def _offload(func: Callable[[], _T], *, timeout: float) -> _T:
        """Run blocking work in a thread with a hard timeout (503 when exceeded)."""
        try:
            return await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout)
        except asyncio.TimeoutError as err:
            raise HTTPException(status_code=503, detail="Request timed out") from err

    async def _session_path_or_404(self, pid: str, sid: str) -> Path:
        """Resolve an existing project session file off the loop, or raise 404."""

        def _resolve(store: CloudStore) -> Path:
            path = store.session_path(pid, sid)
            if not path.exists():
                raise KeyError(f"Session {sid!r} not found")
            return path

        try:
            return await self._store_io(_resolve)
        except (KeyError, ValueError) as err:
            raise HTTPException(status_code=404, detail="Session not found") from err

    async def _sessions_dir_or_404(self, pid: str) -> Path:
        try:
            return await self._store_io(lambda store: store.sessions_dir(pid))
        except (KeyError, ValueError) as err:
            raise HTTPException(status_code=404, detail="Project not found") from err

    def _spawn_background(self, coro: Any) -> asyncio.Task[None]:
        """Start fire-and-forget work, keeping a strong reference until it finishes."""
        task: asyncio.Task[None] = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _drain_background_tasks(self, timeout: float = _SHUTDOWN_GRACE_S) -> None:
        """Let in-flight callbacks finish briefly, then cancel the rest."""
        tasks = set(self._background_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    @property
    def auth_required(self) -> bool:
        """True when any token is configured; every /api route then needs a valid one."""
        return bool(self.auth_token or self.admin_token)

    def _install_auth_middleware(self) -> None:
        """Gate /api with the configured tokens and pre-check upload sizes."""

        @self.app.middleware("http")
        async def require_vcr_token(request: Request, call_next: Any) -> Any:
            path = request.url.path
            if request.method == "POST" and path.endswith("/sessions/upload"):
                length = request.headers.get("content-length", "")
                if length.isdigit() and int(length) > (
                    MAX_UPLOAD_BYTES + _MULTIPART_OVERHEAD_BYTES
                ):
                    return JSONResponse(
                        {"detail": f"Upload exceeds {MAX_UPLOAD_BYTES} bytes"}, status_code=413
                    )

            if not self.auth_required or not path.startswith("/api"):
                return await call_next(request)
            if path in _AUTH_EXEMPT_PATHS or path.startswith(_AUTH_EXEMPT_PREFIXES):
                return await call_next(request)

            try:
                identity = await asyncio.wait_for(
                    asyncio.to_thread(self._resolve_identity, _header_tokens(request.headers)),
                    timeout=10,
                )
            except asyncio.TimeoutError:
                return JSONResponse({"detail": "Token verification timed out"}, status_code=503)
            # Route handlers reuse this instead of verifying the token a second time.
            request.scope[_IDENTITY_SCOPE_KEY] = identity
            role = identity[0]
            if role is None:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            if role != "admin" and not path.startswith(_PROJECT_SCOPED_PREFIX):
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            return await call_next(request)

    def _is_admin_token(self, supplied: str) -> bool:
        # The legacy --auth-token is an admin-equivalent credential (backwards compatible).
        return _tokens_equal(supplied, self.admin_token) or _tokens_equal(
            supplied, self.auth_token
        )

    def _resolve_identity(self, candidates: list[str]) -> tuple[str | None, str | None]:
        """Map supplied tokens to ``(role, project_id)``; ``(None, None)`` when invalid."""
        for supplied in candidates:
            if self._is_admin_token(supplied):
                return "admin", None
        store = self._cloud_store() if candidates else None
        for supplied in candidates:
            assert store is not None
            record = store.verify_token(supplied)
            if record:
                return record.role, record.project_id
        return None, None

    async def _hosted_user(self, request: Request) -> HostedUser:
        """Authenticate an opt-in hosted endpoint with a Supabase access token."""
        if self.hosted is None:
            raise HTTPException(status_code=404, detail="Hosted control plane is disabled")
        authorization = request.headers.get("authorization", "")
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        token = authorization[7:].strip()
        try:
            return await asyncio.wait_for(
                self.hosted.auth.verify(token), timeout=_HOSTED_AUTH_TIMEOUT_S
            )
        except asyncio.TimeoutError as err:
            raise HTTPException(
                status_code=503, detail="Hosted authentication timed out"
            ) from err
        except HostedAuthenticationError as err:
            raise HTTPException(status_code=401, detail="Invalid bearer token") from err
        except HostedDependencyError as err:
            raise HTTPException(status_code=503, detail="Hosted authentication is unavailable") from err

    async def _hosted_call(self, callback: Any, *args: Any, timeout: float = 15.0) -> Any:
        """Run blocking hosted-provider I/O off the event loop with a hard timeout."""
        try:
            return await asyncio.wait_for(asyncio.to_thread(callback, *args), timeout=timeout)
        except asyncio.TimeoutError as err:
            raise HTTPException(status_code=503, detail="Hosted provider timed out") from err
        except HostedDependencyError as err:
            # Includes HostedUnavailableError: a database/IdP outage is a 503, not a
            # billing-provider rejection.
            raise HTTPException(status_code=503, detail="Hosted provider is unavailable") from err
        except HostedConflictError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except HostedBillingError as err:
            raise HTTPException(status_code=502, detail="Hosted billing provider rejected the request") from err

    def _session_file(self, session_id: str) -> Path:
        if not _SAFE_SESSION_ID.fullmatch(session_id):
            raise HTTPException(status_code=400, detail="Invalid session id")
        return self.vcr_dir / f"{session_id}.vcr"

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Manage server lifespan."""
        # Store the event loop so watchdog threads can schedule async work
        self.watcher.set_event_loop(asyncio.get_running_loop())

        self.observer = Observer()
        self.observer.schedule(self.watcher, str(self.vcr_dir), recursive=False)
        self.observer.start()
        logger.info("File watcher started on %s", self.vcr_dir)

        yield

        if self.observer:
            self.observer.stop()
            await asyncio.to_thread(self.observer.join, 10)
            logger.info("File watcher stopped")
        await self._drain_background_tasks()

    def _workflow_registry(self) -> Any:
        from agent_vcr.workflows.store import WorkflowRegistry
        return WorkflowRegistry(self.data_dir or self.vcr_dir)

    def _workflow_runs(self) -> Any:
        from agent_vcr.workflows.store import RunStore
        return RunStore(self.data_dir or self.vcr_dir)

    @property
    def _workflow_handlers(self) -> tuple[Any, dict[str, Any]]:
        if hasattr(self, "_wf_gateway"):
            return self._wf_gateway, self._wf_agents  # type: ignore[has-type]
        from agent_vcr.superagent.router import ToolGateway, ToolSpec
        gateway = ToolGateway()
        agents: dict[str, Any] = {}
        if self.workflow_handlers:
            import importlib.util
            from pathlib import Path
            p = Path(self.workflow_handlers)
            spec = importlib.util.spec_from_file_location("handlers", p)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load handlers from {p}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            approval_set = set(getattr(mod, "APPROVAL_REQUIRED", []))
            for name, handler in getattr(mod, "TOOLS", {}).items():
                if isinstance(handler, ToolSpec):
                    gateway.register(handler)
                else:
                    gateway.register(ToolSpec(
                        name=name, handler=handler,
                        parameters=getattr(handler, "PARAMS", {}),
                        requires_approval=name in approval_set))
            agents = dict(getattr(mod, "AGENTS", {}))
        self._wf_gateway, self._wf_agents = gateway, agents
        return gateway, agents

    async def _resolve_callback(self, url: str) -> CallbackTarget:
        """Validate ``callback_url`` off the loop; ``ValueError`` when it is not allowed."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    resolve_callback_url, url, allow_private=self.allow_private_callbacks
                ),
                timeout=_CALLBACK_RESOLVE_TIMEOUT_S,
            )
        except asyncio.TimeoutError as err:
            raise ValueError("callback_url host lookup timed out") from err

    async def _run_and_callback(
        self, engine: Any, definition: Any, inputs: dict[str, Any], secret: str, callback_url: str
    ) -> None:
        origin = callback_origin(callback_url)
        result: dict[str, Any]
        try:
            record = await asyncio.to_thread(engine.trigger, definition, inputs, secret)
            result = {
                "run_id": record.run_id,
                "status": record.status.value,
                "session_id": record.session_id,
            }
        except Exception as e:
            result = {"run_id": "unknown", "status": "failed", "error": str(e)}
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(engine.shutdown, close_gateway=False),
                    timeout=_STORE_TIMEOUT_S,
                )
            except Exception as err:
                logger.warning("Workflow engine shutdown failed: %s", type(err).__name__)
        data = json.dumps(result, default=str).encode("utf-8")
        try:
            # Re-validated right before sending: DNS may have changed since the
            # trigger request. The connection is pinned to the checked address.
            target = await self._resolve_callback(callback_url)
        except ValueError as err:
            logger.warning("Webhook callback to %s refused: %s", origin, err)
            return
        try:
            status = await asyncio.wait_for(
                asyncio.to_thread(post_callback, target, data, _CALLBACK_POST_TIMEOUT_S),
                timeout=_CALLBACK_POST_TIMEOUT_S + 5,
            )
        except asyncio.TimeoutError:
            logger.warning("Webhook callback to %s timed out", origin)
            return
        except Exception as err:
            logger.warning("Webhook callback to %s failed: %s", origin, type(err).__name__)
            return
        if not 200 <= status < 300:
            logger.warning("Webhook callback to %s returned HTTP %d", origin, status)

    def _setup_workflow_routes(self) -> None:
        # Workflows are server-global (not per project): every route below calls
        # _require_cloud without a project id, so only admin-level tokens (or the
        # tokenless loopback dev mode) can reach them; project tokens get 403.

        @self.app.post("/api/workflows", status_code=201)
        async def save_workflow(request: Request, body: dict[str, Any]) -> dict[str, Any]:
            self._require_cloud(request, write=True)
            from agent_vcr.workflows.models import WorkflowDef
            try:
                definition = WorkflowDef.model_validate(body)
                stored = await asyncio.to_thread(self._workflow_registry().save, definition)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) # noqa: B904
            return stored.model_dump()

        @self.app.get("/api/workflows")
        async def list_workflows(request: Request) -> dict[str, Any]:
            self._require_cloud(request)
            return {"workflows": await asyncio.to_thread(self._workflow_registry().list)}

        @self.app.get("/api/workflows/{name}")
        async def get_workflow(name: str, request: Request, version: int | None = None) -> dict[str, Any]:
            self._require_cloud(request)
            try:
                definition = await asyncio.to_thread(self._workflow_registry().get, name, version)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) # noqa: B904
            return definition.model_dump()

        @self.app.post("/api/workflows/{name}/trigger")
        async def trigger_workflow(name: str, request: Request, body: WorkflowTriggerRequest) -> Any:
            from fastapi.responses import JSONResponse

            from agent_vcr.superagent import CallbackApprover
            from agent_vcr.workflows.engine import WorkflowEngine
            from agent_vcr.workflows.models import StepKind

            self._require_cloud(request, write=True)
            try:
                definition = await asyncio.to_thread(self._workflow_registry().get, name, None)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) # noqa: B904

            if body.callback_url:
                try:
                    await self._resolve_callback(body.callback_url)
                except ValueError as err:
                    raise HTTPException(status_code=422, detail=str(err)) from err

            def _needs_handlers(step: Any) -> bool:
                if step.kind in (StepKind.TOOL, StepKind.AGENT):
                    return True
                nested: list[Any] = list(step.then_steps) + list(step.else_steps) + list(step.body)
                for branch in step.branches.values():
                    nested.extend(branch)
                return any(_needs_handlers(s) for s in nested)

            gateway, agents = self._workflow_handlers
            if not self.workflow_handlers and any(_needs_handlers(s) for s in definition.steps):
                raise HTTPException(status_code=409, detail="Server has no workflow handlers configured")

            engine = WorkflowEngine(
                gateway, agents=agents, vcr_dir=str(self.vcr_dir),
                run_store=self._workflow_runs(),
                approver=CallbackApprover(lambda req: False, name="server-deny"),
            )

            if body.callback_url:
                self._spawn_background(
                    self._run_and_callback(
                        engine, definition, body.inputs, body.secret, body.callback_url
                    )
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "run_id": "pending",
                        "status": "accepted",
                        "callback_url": body.callback_url,
                    },
                )

            try:
                record = await asyncio.to_thread(engine.trigger, definition, body.inputs, body.secret)
            except PermissionError as err:
                raise HTTPException(status_code=403, detail=str(err)) # noqa: B904
            except (ValueError, KeyError) as err:
                raise HTTPException(status_code=422, detail=str(err)) # noqa: B904
            finally:
                await asyncio.to_thread(engine.shutdown, close_gateway=False)
            return {"run_id": record.run_id, "status": record.status.value, "session_id": record.session_id}

        @self.app.get("/api/workflows/{name}/runs")
        async def list_workflow_runs(name: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request)
            return {"runs": await asyncio.to_thread(self._workflow_runs().list, name, None, 100)}

        @self.app.get("/api/workflow-runs/{run_id}")
        async def get_run(run_id: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request)
            try:
                record = await asyncio.to_thread(self._workflow_runs().load, run_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) # noqa: B904
            return record.model_dump()

        @self.app.post("/api/workflow-runs/{run_id}/resume")
        async def resume_run(run_id: str, request: Request) -> dict[str, Any]:
            from agent_vcr.superagent import CallbackApprover
            from agent_vcr.workflows.engine import WorkflowEngine
            self._require_cloud(request, write=True)
            gateway, agents = self._workflow_handlers
            engine = WorkflowEngine(
                gateway, agents=agents, vcr_dir=str(self.vcr_dir),
                run_store=self._workflow_runs(),
                approver=CallbackApprover(lambda req: False, name="server-deny"),
            )
            try:
                record = await asyncio.to_thread(engine.resume, run_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) # noqa: B904
            finally:
                await asyncio.to_thread(engine.shutdown, close_gateway=False)
            return {"run_id": record.run_id, "status": record.status.value}

    def _require_cloud(
        self, request: Request, project_id: str | None = None, write: bool = False
    ) -> None:
        if not self.auth_required:
            return  # local dev mode: loopback-only unless --allow-unauthenticated

        role, token_pid = self._cloud_identity(request)

        if role is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if role == "admin":
            return
        # Project tokens are only valid for their own project's routes.
        if project_id is None or token_pid != project_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        if write and role != "owner":
            raise HTTPException(status_code=403, detail="Forbidden")

    def _require_admin(self, request: Request, *, allow_open: bool) -> None:
        """Require the admin token. ``allow_open`` keeps tokenless local dev mode open."""
        if not self.auth_required:
            if allow_open:
                return
            raise HTTPException(
                status_code=403,
                detail=(
                    "This operation requires an admin token; start the server with "
                    f"--admin-token or {ADMIN_TOKEN_ENV}"
                ),
            )
        role, _ = self._cloud_identity(request)
        if role is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if role != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

    def _cloud_identity(self, request: Request) -> tuple[str | None, str | None]:
        cached = request.scope.get(_IDENTITY_SCOPE_KEY)
        if cached is not None:
            role, project_id = cached
            return role, project_id
        return self._resolve_identity(_header_tokens(request.headers))

    def _setup_cloud_routes(self) -> None:
        from fastapi import File, Form
        from fastapi.responses import Response

        def _not_found(kind: str, err: Exception) -> HTTPException:
            # KeyError (missing) and ValueError (invalid slug) both mean "no such thing".
            if isinstance(err, KeyError) and err.args:
                return HTTPException(status_code=404, detail=str(err.args[0]))
            return HTTPException(status_code=404, detail=f"{kind} not found")

        @self.app.get("/api/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "version": __version__, "auth": bool(self.admin_token or self.auth_token)}

        @self.app.get("/api/projects")
        async def list_projects(request: Request) -> dict[str, Any]:
            self._require_admin(request, allow_open=True)
            return {"projects": await self._store_io(lambda store: store.list_projects())}

        @self.app.post("/api/projects", status_code=201)
        async def create_project(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
            # Never anonymous: disabled entirely when no admin token is configured.
            self._require_admin(request, allow_open=False)
            body = body or {}
            name = body.get("name", "New Project")
            project_id = body.get("id") or None
            description = body.get("description") or ""
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(status_code=422, detail="Project name must be a non-empty string")
            if project_id is not None and not isinstance(project_id, str):
                raise HTTPException(status_code=422, detail="Project id must be a string")
            if not isinstance(description, str):
                raise HTTPException(status_code=422, detail="Project description must be a string")
            try:
                project = await self._store_io(
                    lambda store: store.create_project(name.strip(), project_id, description)
                )
            except FileExistsError as err:
                raise HTTPException(status_code=409, detail=str(err)) from err
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err)) from err
            return project.model_dump()

        @self.app.post("/api/projects/{pid}/tokens")
        async def mint_token(pid: str, request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
            # Never anonymous: disabled entirely when no admin token is configured.
            self._require_admin(request, allow_open=False)
            body = body or {}
            token_role = body.get("role", "member")
            name = body.get("name", "Token")
            if token_role not in ("owner", "member"):
                raise HTTPException(
                    status_code=422, detail="Token role must be 'owner' or 'member'"
                )
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(status_code=422, detail="Token name must be a non-empty string")
            try:
                record, token = await self._store_io(
                    lambda store: store.mint_token(pid, name=name, role=token_role)
                )
            except (KeyError, ValueError) as err:
                raise _not_found("Project", err) from err
            return {"id": record.id, "token": token}

        @self.app.get("/api/projects/{pid}/tokens")
        async def list_tokens(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid, write=True)
            try:
                tokens = await self._store_io(lambda store: store.list_tokens(pid))
            except (KeyError, ValueError) as err:
                raise _not_found("Project", err) from err
            return {"tokens": [t.model_dump() for t in tokens]}

        @self.app.post("/api/projects/{pid}/tokens/{tid}/revoke")
        async def revoke_token(pid: str, tid: str, request: Request) -> dict[str, str]:
            self._require_cloud(request, project_id=pid, write=True)

            def _revoke(store: CloudStore) -> bool:
                """True when the token exists (revoking twice is idempotent)."""
                if store.revoke_token(pid, tid):
                    return True
                return any(t.id == tid for t in store.list_tokens(pid))

            try:
                found = await self._store_io(_revoke)
            except (KeyError, ValueError) as err:
                raise _not_found("Project", err) from err
            if not found:
                raise HTTPException(status_code=404, detail=f"Token not found: {tid!r}")
            return {"status": "revoked"}

        @self.app.get("/api/projects/{pid}/sessions")
        async def list_project_sessions(pid: str, request: Request, limit: int = 50, offset: int = 0, search: str | None = None) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            try:
                return await self._store_io(
                    lambda store: store.list_session_headers(
                        pid, limit=max(limit, 0), offset=max(offset, 0), search=search
                    )
                )
            except (KeyError, ValueError) as err:
                raise _not_found("Project", err) from err

        @self.app.post("/api/projects/{pid}/sessions/upload")
        async def upload_session(pid: str, request: Request, file: UploadFile = File(...), overwrite: bool = Form(False)) -> dict[str, Any]:  # noqa: B008
            self._require_cloud(request, project_id=pid, write=True)
            content = await _read_upload_limited(file, MAX_UPLOAD_BYTES)
            try:
                return await self._store_io(
                    lambda store: store.put_session(pid, content, overwrite=overwrite),
                    timeout=_STORE_UPLOAD_TIMEOUT_S,
                )
            except KeyError as err:
                raise _not_found("Project", err) from err
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) # noqa: B904
            except FileExistsError as e:
                raise HTTPException(status_code=409, detail=str(e)) # noqa: B904

        @self.app.get("/api/projects/{pid}/sessions/{sid}")
        async def get_project_session(pid: str, sid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            path = await self._session_path_or_404(pid, sid)

            def _load() -> dict[str, Any]:
                player = VCRPlayer.load(path)
                statistics = {
                    "total_frames": len(player.frames),
                    "total_latency_ms": player.get_total_latency(),
                    "total_tokens": player.get_total_tokens(),
                    "total_cost_usd": player.get_total_cost(),
                    "nodes": player.list_nodes(),
                    "errors": len(player.get_errors()),
                }
                return {
                    "session": player.session.model_dump(),
                    "frames": [f.model_dump() for f in player.frames],
                    "statistics": statistics,
                }

            return await self._offload(_load, timeout=_STORE_TIMEOUT_S)

        @self.app.get("/api/projects/{pid}/sessions/{sid}/download")
        async def download_session(pid: str, sid: str, request: Request) -> Response:
            self._require_cloud(request, project_id=pid)
            try:
                content = await self._store_io(lambda store: store.get_session_bytes(pid, sid))
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=404, detail="Session not found") from e
            return Response(content=content, media_type="application/octet-stream")

        @self.app.delete("/api/projects/{pid}/sessions/{sid}")
        async def delete_session(pid: str, sid: str, request: Request) -> dict[str, str]:
            self._require_cloud(request, project_id=pid, write=True)
            try:
                deleted = await self._store_io(lambda store: store.delete_session(pid, sid))
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=404, detail="Session not found") from e
            if not deleted:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"status": "deleted"}

        @self.app.post("/api/projects/{pid}/sessions/{sid}/replay")
        async def replay_session(pid: str, sid: str, request: Request, body: ReplayRequest) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            if body.mode != "full":
                raise HTTPException(status_code=400, detail="Only full mode is supported")

            import tempfile

            from agent_vcr.golden_cache import GoldenRunCache

            path = await self._session_path_or_404(pid, sid)

            def _do_replay() -> dict[str, Any]:
                player = VCRPlayer.load(path)
                with tempfile.TemporaryDirectory() as td:
                    cache = GoldenRunCache(td)

                    class FakeRecorder:
                        def get_session(self): return player.session  # type: ignore
                        def get_frames(self): return player.frames    # type: ignore

                    task_name = body.task or "ghost_task"
                    cache.save_golden_run(task_name, FakeRecorder()) # type: ignore
                    outputs, ledger = cache.replay_with_mode(task=task_name, mode=body.mode)
                    return {"outputs": outputs, "ledger": ledger.summary()}

            return await self._offload(_do_replay, timeout=_REPLAY_TIMEOUT_S)

    def _setup_intel_routes(self) -> None:
        @self.app.get("/api/projects/{pid}/intel/failures")
        async def intel_failures(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.failures import cluster_sessions

            sessions_dir = await self._sessions_dir_or_404(pid)
            clusters = await self._offload(
                lambda: cluster_sessions(sessions_dir), timeout=_REPLAY_TIMEOUT_S
            )
            return {"clusters": [c.model_dump() for c in clusters]}

        @self.app.get("/api/projects/{pid}/intel/risk")
        async def intel_risk(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.risk import score_agent

            sessions_dir = await self._sessions_dir_or_404(pid)
            score = await self._offload(
                lambda: score_agent(sessions_dir), timeout=_REPLAY_TIMEOUT_S
            )
            return score.model_dump()

        @self.app.get("/api/projects/{pid}/intel/why/{session_id}")
        async def intel_why(pid: str, session_id: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.query import diagnose

            sessions_dir = await self._sessions_dir_or_404(pid)
            session_path = await self._session_path_or_404(pid, session_id)
            diag = await self._offload(
                lambda: diagnose(session_path, sessions_dir), timeout=_REPLAY_TIMEOUT_S
            )
            return diag.model_dump()

    def _setup_hosted_routes(self) -> None:
        """Expose optional Supabase-authenticated account and Stripe billing routes.

        These endpoints deliberately live under a separate namespace. They do
        not alter the backwards-compatible local/self-hosted API token model.
        """
        assert self.hosted is not None
        hosted = self.hosted

        @self.app.get("/api/hosted/me")
        async def hosted_me(request: Request) -> dict[str, Any]:
            user = await self._hosted_user(request)
            account = await self._hosted_call(hosted.accounts.ensure_account, user)
            return {
                "user": {"id": user.id, "email": user.email},
                "billing": {
                    "plan": account.plan,
                    "status": account.subscription_status,
                    "entitled": account.entitled,
                    "enabled": hosted.billing is not None,
                },
            }

        @self.app.post("/api/hosted/billing/checkout")
        async def hosted_checkout(request: Request) -> dict[str, str]:
            user = await self._hosted_user(request)
            if hosted.billing is None:
                raise HTTPException(status_code=503, detail="Stripe billing is not configured")
            account = await self._hosted_call(hosted.accounts.ensure_account, user)
            if account.has_live_subscription:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This account already has a subscription; use "
                        "POST /api/hosted/billing/portal to manage it"
                    ),
                )
            checkout_url = await self._hosted_call(hosted.billing.create_checkout, user, account)
            return {"checkout_url": checkout_url}

        @self.app.post("/api/hosted/billing/portal")
        async def hosted_billing_portal(request: Request) -> dict[str, str]:
            user = await self._hosted_user(request)
            if hosted.billing is None:
                raise HTTPException(status_code=503, detail="Stripe billing is not configured")
            account = await self._hosted_call(hosted.accounts.ensure_account, user)
            portal_url = await self._hosted_call(hosted.billing.create_portal, account)
            return {"portal_url": portal_url}

        @self.app.post("/api/hosted/billing/webhook")
        async def hosted_billing_webhook(request: Request) -> dict[str, bool]:
            if hosted.billing is None:
                raise HTTPException(status_code=404, detail="Stripe billing is disabled")
            content_length = request.headers.get("content-length")
            if content_length and (not content_length.isdigit() or int(content_length) > 1_000_000):
                raise HTTPException(status_code=413, detail="Webhook payload is too large")
            try:
                payload = await asyncio.wait_for(request.body(), timeout=10)
            except asyncio.TimeoutError as err:
                raise HTTPException(status_code=408, detail="Webhook body timed out") from err
            if len(payload) > 1_000_000:
                raise HTTPException(status_code=413, detail="Webhook payload is too large")
            try:
                event = await asyncio.wait_for(
                    asyncio.to_thread(
                        hosted.billing.verify_webhook,
                        payload,
                        request.headers.get("stripe-signature"),
                    ),
                    timeout=15,
                )
            except asyncio.TimeoutError as err:
                raise HTTPException(status_code=503, detail="Stripe webhook verification timed out") from err
            except HostedDependencyError as err:
                raise HTTPException(status_code=503, detail="Stripe webhook verification is unavailable") from err
            except HostedBillingError as err:
                raise HTTPException(status_code=400, detail="Invalid Stripe webhook") from err
            try:
                processed = await asyncio.wait_for(
                    asyncio.to_thread(hosted.accounts.apply_stripe_event, event), timeout=15
                )
            except asyncio.TimeoutError as err:
                # 5xx makes Stripe redeliver; the event ledger keeps that idempotent.
                raise HTTPException(
                    status_code=503, detail="Hosted account store timed out"
                ) from err
            except HostedDependencyError as err:
                raise HTTPException(
                    status_code=503, detail="Hosted account store is unavailable"
                ) from err
            except HostedBillingError as err:
                raise HTTPException(status_code=400, detail="Malformed Stripe event") from err
            return {"received": True, "processed": processed}

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root() -> Any:
            if self._has_dashboard:
                return RedirectResponse(url="/dashboard/")
            return {
                "name": "Agent VCR API",
                "version": __version__,
                "dashboard": "Not built — run: cd app/vcr-dashboard && npm run build",
                "endpoints": {
                    "sessions": "/api/sessions",
                    "session_detail": "/api/sessions/{session_id}",
                    "websocket": "/ws/live",
                },
            }

        @self.app.get("/api/sessions", response_model=SessionListResponse)
        async def list_sessions(
            search: str | None = None,
            tags: str | None = None,
        ) -> SessionListResponse:
            """List all recorded sessions."""
            sessions = []
            manifest_path = self.vcr_dir / "manifest.json"

            if manifest_path.exists():
                content = await asyncio.to_thread(manifest_path.read_text)
                manifest = json.loads(content)
                for session_data in manifest.get("sessions", []):
                    sessions.append(Session(**session_data))

            for vcr_file in self.vcr_dir.glob("*.vcr"):
                session_id = vcr_file.stem
                if not any(s.session_id == session_id for s in sessions):
                    try:
                        player = await asyncio.to_thread(VCRPlayer.load, vcr_file)
                        sessions.append(player.session)
                    except Exception:
                        continue

            sessions.sort(key=lambda s: s.created_at, reverse=True)

            # Apply search / tag filters
            if search:
                q = search.lower()
                sessions = [s for s in sessions if q in s.session_id.lower() or
                            any(q in t.lower() for t in s.tags) or
                            q in str(s.metadata).lower()]
            if tags:
                wanted = {t.strip() for t in tags.split(",") if t.strip()}
                sessions = [s for s in sessions if wanted.issubset(set(s.tags))]

            return SessionListResponse(sessions=sessions, total=len(sessions))

        @self.app.get("/api/tags")
        async def list_tags() -> dict[str, Any]:
            """List all unique tags across all sessions."""
            all_tags: set[str] = set()
            for vcr_file in self.vcr_dir.glob("*.vcr"):
                try:
                    player = await asyncio.to_thread(VCRPlayer.load, vcr_file)
                    all_tags.update(player.session.tags)
                except Exception:
                    continue
            return {"tags": sorted(all_tags)}

        @self.app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
        async def get_session(session_id: str) -> SessionDetailResponse:
            """Get details of a specific session."""
            vcr_file = self._session_file(session_id)

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = await asyncio.to_thread(VCRPlayer.load, vcr_file)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")  # noqa: B904

            statistics = {
                "total_frames": len(player.frames),
                "total_latency_ms": player.get_total_latency(),
                "total_tokens": player.get_total_tokens(),
                "total_cost_usd": player.get_total_cost(),
                "nodes": player.list_nodes(),
                "errors": len(player.get_errors()),
            }

            return SessionDetailResponse(
                session=player.session,
                frames=player.frames,
                statistics=statistics,
            )

        @self.app.get("/api/sessions/{session_id}/frames/{frame_index}")
        async def get_frame(session_id: str, frame_index: int) -> dict[str, Any]:
            """Get a specific frame from a session."""
            vcr_file = self._session_file(session_id)

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = await asyncio.to_thread(VCRPlayer.load, vcr_file)
                frame = player.get_frame(frame_index)
                return {
                    "frame": frame.model_dump(),
                    "input_state": player.get_input_state(frame_index),
                    "output_state": player.get_output_state(frame_index),
                }
            except IndexError:
                raise HTTPException(status_code=404, detail="Frame not found")  # noqa: B904
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))  # noqa: B904

        @self.app.post("/api/sessions/{session_id}/resume")
        async def resume_session(session_id: str, request: ResumeRequest) -> dict[str, Any]:
            """Resume execution from a specific frame."""
            vcr_file = self._session_file(session_id)

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = await asyncio.to_thread(VCRPlayer.load, vcr_file)
                state = player.goto_frame(request.from_frame)

                if request.state_overrides:
                    state.update(request.state_overrides)

                return {
                    "session_id": session_id,
                    "from_frame": request.from_frame,
                    "state": state,
                    "mode": request.mode,
                    "message": "To complete resume, use the Python SDK with this state",
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))  # noqa: B904

        @self.app.get("/api/sessions/{session_id}/export")
        async def export_session(session_id: str, export_format: str = "json") -> dict[str, Any]:
            """Export a session in various formats."""
            vcr_file = self._session_file(session_id)

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = await asyncio.to_thread(VCRPlayer.load, vcr_file)

                if export_format == "json":
                    return player.to_dict()
                elif export_format == "mermaid":
                    return {"mermaid": self._to_mermaid(player)}
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported format: {export_format}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))  # noqa: B904

        @self.app.post("/api/push")
        async def push_frame(frame_data: dict[str, Any]) -> dict[str, str]:
            """Push a frame directly for real-time WebSocket broadcast.

            Recorders can call this endpoint to broadcast frames immediately
            without relying on file-watcher based detection.
            """
            message = {
                "type": "frame_push",
                "session_id": frame_data.get("session_id", "unknown"),
                "frame": frame_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.watcher.broadcast(message)
            return {"status": "pushed"}

        @self.app.websocket("/ws/live")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for live updates."""
            if self.auth_required:
                # Browsers cannot set headers on a WebSocket handshake, so the
                # ?token= query parameter is accepted here (and only here).
                candidates = _header_tokens(websocket.headers)
                query_token = websocket.query_params.get("token")
                if query_token:
                    candidates.append(query_token)
                # The live feed spans every local recording: admin-level tokens only.
                if not any(self._is_admin_token(c) for c in candidates):
                    await websocket.close(code=1008)
                    return

            await websocket.accept()
            await self.watcher.connect(websocket)

            try:
                while True:
                    data = await websocket.receive_json()

                    if data.get("action") == "subscribe":
                        session_id = data.get("session_id")
                        await websocket.send_json({
                            "type": "subscribed",
                            "session_id": session_id,
                        })
                    elif data.get("action") == "ping":
                        await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.debug("WebSocket client loop stopped unexpectedly", exc_info=True)
            finally:
                await self.watcher.disconnect(websocket)

    def _to_mermaid(self, player: VCRPlayer) -> str:
        """Convert session to Mermaid diagram."""
        lines = ["graph TD"]

        prev_node = None
        for i, frame in enumerate(player.frames):
            node_id = f"N{i}"
            node_label = frame.node_name.replace('"', '\\"')

            if frame.frame_type.value == "error":
                lines.append(f'    {node_id}["{node_label}"]:::error')
            else:
                lines.append(f'    {node_id}["{node_label}"]')

            if prev_node:
                lines.append(f"    {prev_node} --> {node_id}")

            prev_node = node_id

        lines.append("    classDef error fill:#f96,stroke:#333,stroke-width:2px")

        return "\n".join(lines)

    def check_bind_safety(self) -> None:
        """Refuse an unauthenticated non-loopback bind unless explicitly allowed."""
        problem = bind_safety_problem(self.host, self.auth_token, self.admin_token)
        if problem is None:
            return
        if not self.allow_unauthenticated:
            raise RuntimeError(problem)
        warning = (
            "WARNING: Agent VCR is serving an UNAUTHENTICATED API on "
            f"{self.host}:{self.port}. Anyone who can reach this address can read, "
            "upload and delete recordings and trigger workflows. "
            f"Set {AUTH_TOKEN_ENV} / {ADMIN_TOKEN_ENV} to secure it."
        )
        logger.warning(warning)
        print(f"\n{'!' * 78}\n{warning}\n{'!' * 78}\n", file=sys.stderr)

    def run(self) -> None:
        """Run the server."""
        import uvicorn

        self.check_bind_safety()
        logger.info("Starting VCR server on %s:%d", self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Agent VCR Server")
    parser.add_argument("--vcr-dir", default=".vcr", help="Directory for .vcr files")
    parser.add_argument("--data-dir", default=None, help="Directory for cloud project data")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--auth-token",
        default=_env_token(AUTH_TOKEN_ENV),
        help=f"Token required for API and WebSocket access (default: ${AUTH_TOKEN_ENV})",
    )
    parser.add_argument(
        "--admin-token",
        default=_env_token(ADMIN_TOKEN_ENV),
        help=(
            "Administrator token for the cloud API; required to create projects and "
            f"mint tokens (default: ${ADMIN_TOKEN_ENV})"
        ),
    )
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        default=_env_flag(ALLOW_UNAUTHENTICATED_ENV),
        help=(
            "Allow binding a non-loopback host without any token (insecure; "
            f"default: ${ALLOW_UNAUTHENTICATED_ENV}=1)"
        ),
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=None,
        help="Allowed browser origin; may be supplied more than once",
    )
    parser.add_argument(
        "--workflow-handlers",
        default=None,
        help="Python module file exporting workflow TOOLS and AGENTS",
    )
    parser.add_argument(
        "--hosted",
        action="store_true",
        help="Enable Supabase-authenticated hosted account and billing endpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.auth_token = (args.auth_token or "").strip() or None
    args.admin_token = (args.admin_token or "").strip() or None

    problem = bind_safety_problem(args.host, args.auth_token, args.admin_token)
    if problem and not args.allow_unauthenticated:
        parser.error(problem)

    hosted = None
    if args.hosted:
        try:
            from agent_vcr.hosted import HostedConfig

            hosted = HostedControlPlane(HostedConfig.from_environment())
        except HostedConfigurationError as err:
            parser.error(str(err))

    server = VCRServer(
        vcr_dir=args.vcr_dir,
        host=args.host,
        port=args.port,
        auth_token=args.auth_token,
        data_dir=args.data_dir,
        admin_token=args.admin_token,
        allowed_origins=args.allowed_origin,
        workflow_handlers=args.workflow_handlers,
        hosted=hosted,
        allow_unauthenticated=args.allow_unauthenticated,
    )
    server.run()


if __name__ == "__main__":
    main()
