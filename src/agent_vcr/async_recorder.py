"""Async VCR Recorder - Records agent execution using async I/O."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from agent_vcr.models import (
    Frame,
    FrameMetadata,
    FrameType,
    Session,
    StateSerializer,
    VCRCache,
)

logger = logging.getLogger(__name__)


class AsyncVCRRecorder:
    """Async version of VCRRecorder for recording agent execution.

    Uses asyncio.Lock for thread safety and aiofiles for non-blocking I/O.
    """

    def __init__(
        self,
        output_dir: str | Path = ".vcr_recordings",
        auto_save: bool = True,
        buffer_size: int = 100,
        diff_mode: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self.buffer_size = buffer_size
        self.diff_mode = diff_mode

        self._session: Session | None = None
        self._frames: list[Frame] = []
        self._previous_state: dict | None = None
        self._lock = asyncio.Lock()
        self._cache = VCRCache()

    async def start_session(
        self,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        forked_from_frame: int | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Session:
        """Start a new recording session."""
        async with self._lock:
            self._session = Session(
                session_id=session_id or self._generate_session_id(),
                parent_session_id=parent_session_id,
                forked_from_frame=forked_from_frame,
                metadata=metadata or {},
                tags=tags or [],
            )
            self._frames = []
            self._previous_state = None
            self._cache.add_session(self._session)

            logger.info("Started async session: %s", self._session.session_id)

            if self.auto_save:
                await self._write_session_header()

            return self._session

    async def record_step(
        self,
        node_name: str,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
        metadata: FrameMetadata | None = None,
        frame_type: FrameType = FrameType.NODE_EXECUTION,
    ) -> Frame:
        """Record a single execution step."""
        async with self._lock:
            if self._session is None:
                raise RuntimeError("No active session. Call start_session() first.")

            serialized_input = StateSerializer.serialize(input_state)
            serialized_output = StateSerializer.serialize(output_state)

            state_diff = None
            if self.diff_mode and self._previous_state is not None:
                state_diff = self._compute_diff(self._previous_state, serialized_output)

            if metadata is None:
                metadata = FrameMetadata()

            frame = Frame(
                session_id=self._session.session_id,
                node_name=node_name,
                input_state=serialized_input,
                output_state=serialized_output,
                frame_type=frame_type,
                metadata=metadata,
                state_diff=state_diff,
            )

            self._frames.append(frame)
            self._previous_state = serialized_output

            # Update session statistics
            self._session.frame_count = len(self._frames)
            if metadata.tokens_used:
                self._session.total_tokens += metadata.tokens_used
            if metadata.cost_usd:
                self._session.total_cost_usd += metadata.cost_usd

            if self.auto_save and len(self._frames) % self.buffer_size == 0:
                logger.debug("Auto-flushing %d frames to disk", len(self._frames))
                await self._flush_frames()

            return frame

    async def record_llm_call(
        self,
        model: str,
        messages: list[dict],
        response: Any,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> Frame:
        """Record an LLM call."""
        metadata = FrameMetadata(
            model=model,
            tokens_used=tokens_input + tokens_output,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        return await self.record_step(
            node_name=f"llm_{model}",
            input_state={"messages": messages},
            output_state={"response": str(response)},
            metadata=metadata,
            frame_type=FrameType.LLM_CALL,
        )

    async def record_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_ms: float = 0.0,
    ) -> Frame:
        """Record a tool call."""
        metadata = FrameMetadata(latency_ms=latency_ms)
        return await self.record_step(
            node_name=f"tool_{tool_name}",
            input_state={"tool_input": StateSerializer.serialize(tool_input)},
            output_state={"tool_output": StateSerializer.serialize(tool_output)},
            metadata=metadata,
            frame_type=FrameType.TOOL_CALL,
        )

    async def record_error(
        self,
        node_name: str,
        input_state: dict[str, Any],
        error: Exception,
        latency_ms: float = 0.0,
    ) -> Frame:
        """Record an error during execution."""
        metadata = FrameMetadata(
            error_type=type(error).__name__,
            error_message=str(error),
            latency_ms=latency_ms,
        )
        return await self.record_step(
            node_name=node_name,
            input_state=input_state,
            output_state={"error": str(error)},
            metadata=metadata,
            frame_type=FrameType.ERROR,
        )

    async def save(self) -> Path:
        """Flush all pending frames to disk."""
        async with self._lock:
            if self._session is None:
                raise RuntimeError("No active session.")

            path = self._get_session_path()
            if not path.exists() or path.stat().st_size == 0:
                await self._write_session_header()

            await self._flush_frames()
            await self._update_session_manifest()

            logger.info("Saved async session %s to %s", self._session.session_id, path)
            return path

    def get_session(self) -> Session | None:
        return self._session

    def get_frames(self) -> list[Frame]:
        return list(self._frames)

    async def fork(self, from_frame: int) -> AsyncVCRRecorder:
        """Fork a new recorder from a specific frame."""
        if self._session is None:
            raise RuntimeError("No active session to fork from.")

        new_recorder = AsyncVCRRecorder(
            output_dir=self.output_dir,
            auto_save=self.auto_save,
            buffer_size=self.buffer_size,
            diff_mode=self.diff_mode,
        )

        await new_recorder.start_session(
            parent_session_id=self._session.session_id,
            forked_from_frame=from_frame,
        )

        return new_recorder

    # --- Private methods ---

    def _generate_session_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _get_session_path(self) -> Path:
        if self._session is None:
            raise RuntimeError("No active session.")
        return self.output_dir / f"{self._session.session_id}.vcr"

    async def _write_session_header(self) -> None:
        if self._session is None:
            return
        path = self._get_session_path()
        record = {
            "type": "session",
            "data": json.loads(self._session.model_dump_json()),
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(record) + "\n")

    async def _flush_frames(self) -> None:
        if not self._frames:
            return
        path = self._get_session_path()
        async with aiofiles.open(path, "a") as f:
            for frame in self._frames:
                record = {
                    "type": "frame",
                    "data": json.loads(frame.model_dump_json()),
                }
                await f.write(json.dumps(record) + "\n")
        self._frames = []

    async def _update_session_manifest(self) -> None:
        if self._session is None:
            return

        manifest_path = self.output_dir / "manifest.json"
        manifest: dict = {}

        if manifest_path.exists():
            try:
                async with aiofiles.open(manifest_path) as f:
                    content = await f.read()
                    manifest = json.loads(content)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read manifest, starting fresh: %s", e)
                manifest = {}

        sessions = manifest.get("sessions", [])
        session_data = json.loads(self._session.model_dump_json())

        existing_idx = next(
            (i for i, s in enumerate(sessions) if s["session_id"] == self._session.session_id),
            None,
        )

        if existing_idx is not None:
            sessions[existing_idx] = session_data
        else:
            sessions.append(session_data)

        manifest["sessions"] = sessions
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_dir), suffix=".tmp", prefix="manifest_"
        )
        try:
            async with aiofiles.open(tmp_path, "w") as f:
                await f.write(json.dumps(manifest, indent=2))
            os.close(fd)
            os.replace(tmp_path, str(manifest_path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _compute_diff(self, before: dict, after: dict) -> list[dict]:
        """Compute the diff between two states."""
        diffs = []
        all_keys = set(before.keys()) | set(after.keys())

        for key in all_keys:
            if key not in before:
                diffs.append({"op": "add", "path": f"/{key}", "value": after[key]})
            elif key not in after:
                diffs.append({"op": "remove", "path": f"/{key}"})
            elif before[key] != after[key]:
                diffs.append({"op": "replace", "path": f"/{key}", "value": after[key]})

        return diffs
