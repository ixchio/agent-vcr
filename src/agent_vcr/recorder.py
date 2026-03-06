"""VCR Recorder - Records agent execution to .vcr files."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import contextlib  # noqa: E402

from agent_vcr.models import (  # noqa: E402
    Frame,
    FrameMetadata,
    FrameType,
    Session,
    StateSerializer,
    VCRCache,
)


class VCRRecorder:
    """Records agent execution sessions to .vcr files."""

    def __init__(
        self,
        output_dir: str = ".vcr",
        buffer_size: int = 100,
        auto_save: bool = True,
        diff_mode: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.auto_save = auto_save
        self.diff_mode = diff_mode

        self._session: Session | None = None
        self._frames: list[Frame] = []
        self._previous_state: dict | None = None
        self._lock = threading.RLock()
        self._cache = VCRCache()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        forked_from_frame: int | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Session:
        """Start a new recording session."""
        with self._lock:
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

            logger.info("Started session: %s", self._session.session_id)

            if self.auto_save:
                self._write_session_header()

            return self._session

    def record_step(
        self,
        node_name: str,
        input_state: dict[str, Any],
        output_state: dict[str, Any],
        metadata: FrameMetadata | None = None,
        frame_type: FrameType = FrameType.NODE_EXECUTION,
        parent_frame_id: str | None = None,
    ) -> Frame:
        """Record a single execution step."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("No active session. Call start_session() first.")

            start_time = time.perf_counter()

            serialized_input = StateSerializer.serialize(input_state)
            serialized_output = StateSerializer.serialize(output_state)

            state_diff = None
            if self.diff_mode and self._previous_state is not None:
                state_diff = self._compute_diff(
                    self._previous_state, serialized_output
                )

            if isinstance(metadata, dict):
                metadata = FrameMetadata(**metadata)
            elif metadata is None:
                metadata = FrameMetadata()

            metadata.latency_ms = (time.perf_counter() - start_time) * 1000

            frame = Frame(
                session_id=self._session.session_id,
                parent_frame_id=parent_frame_id,
                frame_type=frame_type,
                node_name=node_name,
                input_state=serialized_input if not self.diff_mode else {},
                output_state=serialized_output if not self.diff_mode else {},
                metadata=metadata,
                state_diff=state_diff,
            )

            self._frames.append(frame)
            self._cache.add_frame(self._session.session_id, frame)
            self._previous_state = serialized_output

            self._session.frame_count = len(self._frames)
            if metadata and metadata.tokens_used:
                self._session.total_tokens += metadata.tokens_used
            if metadata and metadata.cost_usd:
                self._session.total_cost_usd += metadata.cost_usd

            if self.auto_save and len(self._frames) % self.buffer_size == 0:
                logger.debug("Auto-flushing %d frames to disk", len(self._frames))
                self._flush_frames()

            return frame

    def record_llm_call(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        cost_usd: float | None = None,
    ) -> Frame:
        """Record an LLM API call."""
        metadata = FrameMetadata(
            model=model,
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_used=tokens_input + tokens_output,
            cost_usd=cost_usd,
        )

        return self.record_step(
            node_name=f"llm:{model}",
            input_state={"messages": messages},
            output_state={"response": response},
            metadata=metadata,
            frame_type=FrameType.LLM_CALL,
        )

    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        latency_ms: float,
        error: str | None = None,
    ) -> Frame:
        """Record a tool execution."""
        metadata = FrameMetadata(
            latency_ms=latency_ms,
            error_message=error,
            error_type=type(error).__name__ if error else None,
        )

        return self.record_step(
            node_name=f"tool:{tool_name}",
            input_state=tool_input,
            output_state={"result": tool_output, "error": error},
            metadata=metadata,
            frame_type=FrameType.TOOL_CALL,
        )

    def record_error(
        self,
        node_name: str,
        input_state: dict,
        error: Exception,
        latency_ms: float,
    ) -> Frame:
        """Record an error during execution."""
        metadata = FrameMetadata(
            latency_ms=latency_ms,
            error_message=str(error),
            error_type=type(error).__name__,
        )

        return self.record_step(
            node_name=node_name,
            input_state=input_state,
            output_state={"error": str(error)},
            metadata=metadata,
            frame_type=FrameType.ERROR,
        )

    def save(self) -> Path:
        """Flush all pending frames to disk."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("No active session.")

            path = self._get_session_path()
            if not path.exists() or path.stat().st_size == 0:
                self._write_session_header()

            self._flush_frames()
            self._update_session_manifest()

            logger.info("Saved session %s to %s", self._session.session_id, path)
            return path

    def get_session(self) -> Session | None:
        """Get the current session."""
        return self._session

    def get_frames(self) -> list[Frame]:
        """Get all recorded frames."""
        return self._frames.copy()

    @property
    def frames(self) -> list[Frame]:
        """Get all recorded frames as a property."""
        return self._frames.copy()

    def fork(
        self,
        from_frame: int,
        new_session_id: str | None = None,
        state_overrides: dict | None = None,
    ) -> VCRRecorder:
        """Create a forked recorder starting from a specific frame."""
        if from_frame >= len(self._frames):
            raise ValueError(f"Frame {from_frame} not found")

        target_frame = self._frames[from_frame]
        forked_state = target_frame.output_state.copy()
        if state_overrides:
            forked_state.update(state_overrides)

        new_recorder = VCRRecorder(
            output_dir=str(self.output_dir),
            buffer_size=self.buffer_size,
            auto_save=self.auto_save,
            diff_mode=self.diff_mode,
        )

        new_recorder.start_session(
            session_id=new_session_id,
            parent_session_id=self._session.session_id if self._session else None,
            forked_from_frame=from_frame,
            metadata={"forked_from": self._session.session_id if self._session else None},
        )

        return new_recorder

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = os.urandom(4).hex()
        return f"{timestamp}_{unique_id}"

    def _get_session_path(self) -> Path:
        """Get the file path for the current session."""
        if self._session is None:
            raise RuntimeError("No active session.")
        return self.output_dir / f"{self._session.session_id}.vcr"

    def _write_session_header(self) -> None:
        """Write the session header to the file."""
        if self._session is None:
            return

        path = self._get_session_path()
        with open(path, "w") as f:
            header = {
                "type": "session",
                "data": self._session.model_dump(),
            }
            f.write(json.dumps(header, separators=(",", ":")) + "\n")

    def _flush_frames(self) -> None:
        """Write buffered frames to disk."""
        if not self._frames:
            return

        path = self._get_session_path()
        with open(path, "a") as f:
            for frame in self._frames:
                line = {
                    "type": "frame",
                    "data": frame.model_dump(),
                }
                f.write(json.dumps(line, separators=(",", ":")) + "\n")

        self._frames = []

    def _update_session_manifest(self) -> None:
        """Update the session manifest file using atomic write to prevent corruption."""
        if self._session is None:
            return

        manifest_path = self.output_dir / "manifest.json"
        manifest: dict = {}

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read manifest, starting fresh: %s", e)
                manifest = {}

        sessions = manifest.get("sessions", [])
        session_data = self._session.model_dump()

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

        # Atomic write: write to temp file, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_dir), suffix=".tmp", prefix="manifest_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_path, str(manifest_path))
        except Exception:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _compute_diff(self, before: dict, after: dict) -> list[dict]:
        """Compute the diff between two states."""
        diff: list[dict] = []

        for key in after:
            if key not in before:
                diff.append({"op": "add", "path": f"/{key}", "value": after[key]})
            elif before[key] != after[key]:
                diff.append({"op": "replace", "path": f"/{key}", "value": after[key]})

        for key in before:
            if key not in after:
                diff.append({"op": "remove", "path": f"/{key}"})

        return diff
