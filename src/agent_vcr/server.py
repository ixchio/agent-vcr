"""FastAPI server for VCR - serves recordings and handles live updates."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from agent_vcr.models import Frame, Session
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[Session]
    total: int


class SessionDetailResponse(BaseModel):
    """Response for session details."""

    session: Session
    frames: list[Frame]
    statistics: dict


class ResumeRequest(BaseModel):
    """Request to resume execution from a frame."""

    from_frame: int
    state_overrides: dict = {}
    mode: str = "fork"
    new_session_id: str | None = None


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

    def __init__(self, vcr_dir: str = ".vcr", host: str = "0.0.0.0", port: int = 8000):
        self.vcr_dir = Path(vcr_dir)
        self.host = host
        self.port = port
        self.watcher = VCRFileWatcher(self.vcr_dir)
        self.observer: Any | None = None

        self.vcr_dir.mkdir(parents=True, exist_ok=True)

        self.app = FastAPI(
            title="Agent VCR API",
            description="API for Agent VCR - The DVR for AI Agents",
            version="0.1.0",
            lifespan=self._lifespan,
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

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

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "name": "Agent VCR API",
                "version": "0.1.0",
                "endpoints": {
                    "sessions": "/api/sessions",
                    "session_detail": "/api/sessions/{session_id}",
                    "websocket": "/ws/live",
                },
            }

        @self.app.get("/api/sessions", response_model=SessionListResponse)
        async def list_sessions() -> SessionListResponse:
            """List all recorded sessions."""
            sessions = []
            manifest_path = self.vcr_dir / "manifest.json"

            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    for session_data in manifest.get("sessions", []):
                        sessions.append(Session(**session_data))

            for vcr_file in self.vcr_dir.glob("*.vcr"):
                session_id = vcr_file.stem
                if not any(s.session_id == session_id for s in sessions):
                    try:
                        player = VCRPlayer.load(vcr_file)
                        sessions.append(player.session)
                    except Exception:
                        continue

            sessions.sort(key=lambda s: s.created_at, reverse=True)

            return SessionListResponse(sessions=sessions, total=len(sessions))

        @self.app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
        async def get_session(session_id: str) -> SessionDetailResponse:
            """Get details of a specific session."""
            vcr_file = self.vcr_dir / f"{session_id}.vcr"

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = VCRPlayer.load(vcr_file)
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
            vcr_file = self.vcr_dir / f"{session_id}.vcr"

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = VCRPlayer.load(vcr_file)
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
            vcr_file = self.vcr_dir / f"{session_id}.vcr"

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = VCRPlayer.load(vcr_file)
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
        async def export_session(session_id: str, format: str = "json") -> dict[str, Any]:
            """Export a session in various formats."""
            vcr_file = self.vcr_dir / f"{session_id}.vcr"

            if not vcr_file.exists():
                raise HTTPException(status_code=404, detail="Session not found")

            try:
                player = VCRPlayer.load(vcr_file)

                if format == "json":
                    return player.to_dict()
                elif format == "mermaid":
                    return {"mermaid": self._to_mermaid(player)}
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))  # noqa: B904

        @self.app.websocket("/ws/live")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for live updates."""
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
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    args = parser.parse_args()

    server = VCRServer(vcr_dir=args.vcr_dir, host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
