"""FastAPI server for VCR - serves recordings and handles live updates."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_vcr import __version__
from agent_vcr.hosted import (
    HostedAuthenticationError,
    HostedBillingError,
    HostedConfigurationError,
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
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

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
        """Broadcast a message to all connected clients."""
        disconnected = []

        for conn in self.connections:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)

        for conn in disconnected:
            await self.disconnect(conn)

    def _schedule_broadcast(self, message: dict) -> None:
        """Thread-safe broadcast scheduling from watchdog threads."""
        if self._loop is None or self._loop.is_closed():
            logger.warning("No event loop available for broadcast")
            return
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future, self.broadcast(message)
        )

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
    ):
        if _SERVER_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The dashboard server dependencies are not installed. "
                'Install them with: pip install "ai-agent-vcr[dashboard]"'
            ) from _SERVER_IMPORT_ERROR

        self.vcr_dir = Path(vcr_dir)
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.data_dir = data_dir
        self.admin_token = admin_token
        self.workflow_handlers = workflow_handlers
        self.hosted = hosted
        self.watcher = VCRFileWatcher(self.vcr_dir)
        self.observer: Any | None = None

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

        if self.auth_token:
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

    def _cloud_store(self) -> Any:
        from agent_vcr.cloud_store import CloudStore
        return CloudStore(self.data_dir or self.vcr_dir)

    def _install_auth_middleware(self) -> None:
        """Require a token for API endpoints when auth_token is configured."""

        @self.app.middleware("http")
        async def require_vcr_token(request: Request, call_next: Any) -> Any:
            if request.url.path.startswith("/api") and not self._request_authorized(request):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return await call_next(request)

    def _request_authorized(self, request: Request) -> bool:
        supplied = request.headers.get("x-agent-vcr-token") or request.query_params.get("token")
        authorization = request.headers.get("authorization")
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        return supplied == self.auth_token

    async def _hosted_user(self, request: Request) -> HostedUser:
        """Authenticate an opt-in hosted endpoint with a Supabase access token."""
        if self.hosted is None:
            raise HTTPException(status_code=404, detail="Hosted control plane is disabled")
        authorization = request.headers.get("authorization", "")
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Bearer token required")
        token = authorization[7:].strip()
        try:
            return await asyncio.wait_for(self.hosted.auth.verify(token), timeout=7)
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
            raise HTTPException(status_code=503, detail="Hosted provider is unavailable") from err
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
            self.observer.join()
            logger.info("File watcher stopped")

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

    async def _run_and_callback(self, engine: Any, definition: Any, inputs: dict[str, Any], secret: str, callback_url: str) -> None:
        import asyncio
        import json
        import logging
        import urllib.request
        logger = logging.getLogger(__name__)
        result: dict[str, Any] = {}
        try:
            record = await asyncio.to_thread(engine.trigger, definition, inputs, secret)
            result = {"run_id": record.run_id, "status": record.status.value, "session_id": record.session_id}
        except Exception as e:
            result = {"run_id": "unknown", "status": "failed", "error": str(e)}
        finally:
            await asyncio.to_thread(engine.shutdown, close_gateway=False)
        data = json.dumps(result).encode("utf-8")
        req = urllib.request.Request(callback_url, data=data, headers={"Content-Type": "application/json"})
        def _post() -> None:
            try:
                urllib.request.urlopen(req, timeout=10)
            except Exception as err:
                logger.warning("Webhook callback to %s failed: %s", callback_url, err)
        try:
            await asyncio.wait_for(asyncio.to_thread(_post), timeout=15)
        except asyncio.TimeoutError:
            logger.warning("Webhook callback to %s timed out", callback_url)

    def _setup_workflow_routes(self) -> None:
        import asyncio

        from fastapi import HTTPException

        def _admin(request: Request) -> None:
            kind, _ = self._cloud_identity(request)
            if kind not in ("admin", "legacy"):
                raise HTTPException(status_code=403, detail="Admin access required")

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
                asyncio.create_task(self._run_and_callback(engine, definition, body.inputs, body.secret, body.callback_url))
                return JSONResponse(status_code=202, content={"run_id": "pending", "status": "accepted", "callback_url": body.callback_url})

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

    def _require_cloud(self, request: Request, project_id: str | None = None, write: bool = False) -> None:
        if not self.admin_token:
            return

        from fastapi import HTTPException
        role, token_pid = self._cloud_identity(request)

        if role is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if role == "admin":
            return
        if project_id and token_pid != project_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        if write and role != "owner":
            raise HTTPException(status_code=403, detail="Forbidden")

    def _cloud_identity(self, request: Request) -> tuple[str | None, str | None]:
        supplied = request.headers.get("x-agent-vcr-token") or request.query_params.get("token")
        authorization = request.headers.get("authorization")
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:]

        if not supplied:
            return None, None

        if supplied == self.admin_token or supplied == self.auth_token:
            return "admin", None

        store = self._cloud_store()
        record = store.verify_token(supplied)
        if record:
            return record.role, record.project_id

        return None, None

    def _setup_cloud_routes(self) -> None:
        import asyncio

        from fastapi import File, Form, HTTPException
        from fastapi.responses import Response

        @self.app.get("/api/health")
        async def health() -> dict[str, Any]:
            return {"status": "ok", "version": __version__, "auth": bool(self.admin_token or self.auth_token)}

        @self.app.get("/api/projects")
        async def list_projects(request: Request) -> dict[str, Any]:
            if self.admin_token:
                role, _ = self._cloud_identity(request)
                if role is None:
                    raise HTTPException(status_code=401, detail="Unauthorized")
                if role != "admin":
                    raise HTTPException(status_code=403, detail="Admin access required")
            return {"projects": await asyncio.to_thread(self._cloud_store().list_projects)}

        @self.app.post("/api/projects", status_code=201)
        async def create_project(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
            if self.admin_token:
                role, _ = self._cloud_identity(request)
                if role is None:
                    raise HTTPException(status_code=401, detail="Unauthorized")
                if role != "admin":
                    raise HTTPException(status_code=403, detail="Admin access required")
            name = body.get("name", "New Project") if body else "New Project"
            project = await asyncio.to_thread(self._cloud_store().create_project, name)
            return project.model_dump()

        @self.app.post("/api/projects/{pid}/tokens")
        async def mint_token(pid: str, request: Request, body: dict[str, str] | None = None) -> dict[str, Any]:
            if self.admin_token:
                role, _ = self._cloud_identity(request)
                if role is None:
                    raise HTTPException(status_code=401, detail="Unauthorized")
                if role != "admin":
                    raise HTTPException(status_code=403, detail="Admin access required")

            token_role = body.get("role", "member") if body else "member"
            name = body.get("name", "Token") if body else "Token"

            # Note: store.mint_token takes (project_id, name, role)
            record, token = await asyncio.to_thread(self._cloud_store().mint_token, pid, name=name, role=token_role)
            return {"id": record.id, "token": token}

        @self.app.get("/api/projects/{pid}/tokens")
        async def list_tokens(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid, write=True)
            tokens = await asyncio.to_thread(self._cloud_store().list_tokens, pid)
            return {"tokens": [t.model_dump() for t in tokens]}

        @self.app.post("/api/projects/{pid}/tokens/{tid}/revoke")
        async def revoke_token(pid: str, tid: str, request: Request) -> dict[str, str]:
            self._require_cloud(request, project_id=pid, write=True)
            await asyncio.to_thread(self._cloud_store().revoke_token, pid, tid)
            return {"status": "revoked"}

        @self.app.get("/api/projects/{pid}/sessions")
        async def list_project_sessions(pid: str, request: Request, limit: int = 50, offset: int = 0, search: str | None = None) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            return await asyncio.to_thread(self._cloud_store().list_session_headers, pid, limit=limit, offset=offset, search=search)

        @self.app.post("/api/projects/{pid}/sessions/upload")
        async def upload_session(pid: str, request: Request, file: UploadFile = File(...), overwrite: bool = Form(False)) -> dict[str, Any]:  # noqa: B008
            self._require_cloud(request, project_id=pid, write=True)
            content = await file.read()
            try:
                return await asyncio.to_thread(self._cloud_store().put_session, pid, content, overwrite=overwrite)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) # noqa: B904
            except FileExistsError as e:
                raise HTTPException(status_code=409, detail=str(e)) # noqa: B904

        @self.app.get("/api/projects/{pid}/sessions/{sid}")
        async def get_project_session(pid: str, sid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            store = self._cloud_store()
            path = store._confined(store._sessions_dir(pid) / f"{sid}.vcr")
            if not path.exists():
                raise HTTPException(status_code=404, detail="Session not found")
            from agent_vcr.player import VCRPlayer
            player = await asyncio.to_thread(VCRPlayer.load, path)
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

        @self.app.get("/api/projects/{pid}/sessions/{sid}/download")
        async def download_session(pid: str, sid: str, request: Request) -> Response:
            self._require_cloud(request, project_id=pid)
            try:
                content = await asyncio.to_thread(self._cloud_store().get_session_bytes, pid, sid)
                return Response(content=content, media_type="application/octet-stream")
            except KeyError as e:
                raise HTTPException(status_code=404, detail="Session not found") from e

        @self.app.delete("/api/projects/{pid}/sessions/{sid}")
        async def delete_session(pid: str, sid: str, request: Request) -> dict[str, str]:
            self._require_cloud(request, project_id=pid, write=True)
            try:
                await asyncio.to_thread(self._cloud_store().delete_session, pid, sid)
                return {"status": "deleted"}
            except KeyError as e:
                raise HTTPException(status_code=404, detail="Session not found") from e

        @self.app.post("/api/projects/{pid}/sessions/{sid}/replay")
        async def replay_session(pid: str, sid: str, request: Request, body: ReplayRequest) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            if body.mode != "full":
                raise HTTPException(status_code=400, detail="Only full mode is supported")

            store = self._cloud_store()
            path = store._confined(store._sessions_dir(pid) / f"{sid}.vcr")
            if not path.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            import tempfile

            from agent_vcr.golden_cache import GoldenRunCache
            from agent_vcr.player import VCRPlayer

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

            return await asyncio.to_thread(_do_replay)

    def _setup_intel_routes(self) -> None:
        import asyncio

        from fastapi import HTTPException

        @self.app.get("/api/projects/{pid}/intel/failures")
        async def intel_failures(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.failures import cluster_sessions
            sessions_dir = Path(self.data_dir or self.vcr_dir) / "projects" / pid / "sessions"
            clusters = await asyncio.to_thread(cluster_sessions, sessions_dir)
            return {"clusters": [c.model_dump() for c in clusters]}

        @self.app.get("/api/projects/{pid}/intel/risk")
        async def intel_risk(pid: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.risk import score_agent
            sessions_dir = Path(self.data_dir or self.vcr_dir) / "projects" / pid / "sessions"
            score = await asyncio.to_thread(score_agent, sessions_dir)
            return score.model_dump()

        @self.app.get("/api/projects/{pid}/intel/why/{session_id}")
        async def intel_why(pid: str, session_id: str, request: Request) -> dict[str, Any]:
            self._require_cloud(request, project_id=pid)
            from agent_vcr.intelligence.query import diagnose
            sessions_dir = Path(self.data_dir or self.vcr_dir) / "projects" / pid / "sessions"
            session_path = sessions_dir / f"{session_id}.vcr"
            if not session_path.exists():
                raise HTTPException(status_code=404, detail="Session not found")
            diag = await asyncio.to_thread(diagnose, session_path, sessions_dir)
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
            processed = await self._hosted_call(hosted.accounts.apply_stripe_event, event)
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
            if self.auth_token:
                supplied = (
                    websocket.headers.get("x-agent-vcr-token")
                    or websocket.query_params.get("token")
                )
                authorization = websocket.headers.get("authorization")
                if authorization and authorization.lower().startswith("bearer "):
                    supplied = authorization[7:]
                if supplied != self.auth_token:
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
                await self.watcher.disconnect(websocket)
            except Exception:
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

    def run(self) -> None:
        """Run the server."""
        import uvicorn

        logger.info("Starting VCR server on %s:%d", self.host, self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent VCR Server")
    parser.add_argument("--vcr-dir", default=".vcr", help="Directory for .vcr files")
    parser.add_argument("--data-dir", default=None, help="Directory for cloud project data")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Optional token required for API and WebSocket access",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="Optional administrator token for the local cloud API",
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

    args = parser.parse_args()
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
    )
    server.run()


if __name__ == "__main__":
    main()
