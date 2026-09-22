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
    StateDiff,
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
        on_frame_recorded: Any | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.auto_save = auto_save
        self.diff_mode = diff_mode
        self._on_frame_recorded = on_frame_recorded

        self._session: Session | None = None
        self._frames: list[Frame] = []
        self._previous_state: dict | None = None
        self._lock = threading.RLock()
        self._cache = VCRCache()
        self._header_frame_count: int | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> VCRRecorder:
        """Support ``with`` usage — auto-saves on exit."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Auto-save session on context manager exit."""
        if self._session is not None:
            try:
                self.save()
            except Exception:
                logger.warning("Failed to auto-save session on exit", exc_info=True)

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
                # In diff mode, omit input_state from the frame when it
                # duplicates the previous output_state (the common case).
                # The player can reconstruct it from the prior frame.
                serialized_input = {}

            if isinstance(metadata, dict):
                metadata = FrameMetadata(**metadata)
            elif metadata is None:
                metadata = FrameMetadata()

            # Only auto-set latency if the caller did not already supply one
            if metadata.latency_ms == 0.0:
                metadata.latency_ms = (time.perf_counter() - start_time) * 1000

            frame = Frame(
                session_id=self._session.session_id,
                parent_frame_id=parent_frame_id,
                frame_type=frame_type,
                node_name=node_name,
                input_state=serialized_input,
                output_state=serialized_output,
                metadata=metadata,
                state_diff=state_diff,
            )

            self._frames.append(frame)
            self._cache.add_frame(self._session.session_id, frame)
            self._previous_state = serialized_output

            self._session.frame_count = len(
                self._cache.get_frames(self._session.session_id)
            )
            self._session.updated_at = datetime.now(timezone.utc)
            if metadata and metadata.tokens_used:
                self._session.total_tokens += metadata.tokens_used
            if metadata and metadata.cost_usd:
                self._session.total_cost_usd += metadata.cost_usd

            if self.auto_save and len(self._frames) % self.buffer_size == 0:
                logger.debug("Auto-flushing %d frames to disk", len(self._frames))
                self._flush_frames()

            # Fire live-push callback if registered
            if self._on_frame_recorded is not None:
                try:
                    self._on_frame_recorded(frame)
                except Exception:
                    logger.debug("on_frame_recorded callback error", exc_info=True)

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
            error_type="ToolError" if error else None,
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
        model: str | None = None,
    ) -> Frame:
        """Record an error during execution.

        Args:
            model: Optional model attribution (feeds routing evidence).
        """
        metadata = FrameMetadata(
            latency_ms=latency_ms,
            error_message=str(error),
            error_type=type(error).__name__,
            model=model,
        )

        return self.record_step(
            node_name=node_name,
            input_state=input_state,
            output_state={"error": str(error)},
            metadata=metadata,
            frame_type=FrameType.ERROR,
        )

    def record_checkpoint(
        self,
        state: dict[str, Any],
        label: str = "checkpoint",
        node_name: str | None = None,
        metadata: FrameMetadata | None = None,
    ) -> Frame:
        """Record an explicit named checkpoint (Phase 0 checkpoint API).

        Checkpoints are addressable restore points: ``player.goto_checkpoint``
        / ``list_checkpoints`` resolve them by ``label`` stored in
        ``metadata.custom["checkpoint_label"]``.

        Args:
            state: Snapshot to persist as both input and output state.
            label: StableCheckpoint name (e.g. ``"before_payment"``).
            node_name: Optional frame name; defaults to ``checkpoint:<label>``.
            metadata: Optional metadata (custom dict is preserved).
        """
        meta = metadata or FrameMetadata()
        custom = dict(meta.custom)
        custom["checkpoint_label"] = label
        meta.custom = custom
        snapshot = StateSerializer.deserialize(StateSerializer.serialize(state))
        return self.record_step(
            node_name=node_name or f"checkpoint:{label}",
            input_state=snapshot,
            output_state=snapshot,
            metadata=meta,
            frame_type=FrameType.CHECKPOINT,
        )

    def create_checkpoint(self, *args: Any, **kwargs: Any) -> Frame:
        """Alias for :meth:`record_checkpoint` (ergonomics)."""
        return self.record_checkpoint(*args, **kwargs)

    def save(self) -> Path:
        """Flush all pending frames to disk (header kept accurate)."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("No active session.")

            path = self._get_session_path()
            if not path.exists() or path.stat().st_size == 0:
                self._write_session_header()

            self._flush_frames()
            if self._header_frame_count != self._session.frame_count:
                self._rewrite_file()
            self._update_session_manifest()

            logger.info("Saved session %s to %s", self._session.session_id, path)
            return path

    def attach_session(self, session_id: str) -> Session:
        """Attach to an existing ``.vcr`` file for append (crash-resume path).

        Loads the session header + all frames into the in-memory cache so
        totals continue monotonically; the file itself is never truncated —
        :meth:`save` appends because the header already exists. Raises
        ``FileNotFoundError`` when no recording exists for the id.
        """
        from agent_vcr.player import VCRPlayer  # local: player imports recorder

        with self._lock:
            if self._session is not None:
                raise RuntimeError("Recorder already has an active session")
            path = self.output_dir / f"{session_id}.vcr"
            if not path.exists():
                raise FileNotFoundError(f"VCR file not found: {path}")
            player = VCRPlayer.load(path)
            if player.session.session_id != session_id:
                raise ValueError(f"Session id mismatch in {path}")
            self._session = player.session
            self._frames = []
            self._previous_state = None
            self._cache = VCRCache()
            self._cache.add_session(self._session)
            for frame in player.frames:
                self._cache.add_frame(self._session.session_id, frame)
            self._header_frame_count = len(player.frames)
            try:
                last_output = player.frames[-1].output_state if player.frames else None
                self._previous_state = last_output
            except IndexError:
                self._previous_state = None
            return self._session

    @staticmethod
    def rebuild_manifest(output_dir: str | Path) -> dict:
        """Rebuild ``manifest.json`` by scanning all ``*.vcr`` files.

        Recovery path for a corrupt/deleted manifest (Phase 0 durability).
        Each file's first ``session`` line is trusted; files without a
        header are skipped. Writes atomically and returns the manifest.
        """
        from pathlib import Path as _Path

        out = _Path(output_dir)
        sessions: list[dict] = []
        skipped: list[str] = []
        for path in sorted(out.glob("*.vcr")):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if record.get("type") == "session":
                            data = record.get("data", {})
                            data.setdefault("session_id", path.stem)
                            sessions.append(data)
                            break
                    else:
                        skipped.append(path.name)
            except OSError:
                skipped.append(path.name)
        manifest = {
            "sessions": sessions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rebuilt": True,
            "skipped_files": skipped,
        }
        manifest_path = out / "manifest.json"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(out), suffix=".tmp", prefix="manifest_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_path, str(manifest_path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return manifest

    def get_frames(self) -> list[Frame]:
        """Get all recorded frames (including already-flushed ones)."""
        if self._session:
            return self._cache.get_frames(self._session.session_id)
        return self._frames.copy()

    def get_session(self) -> Session | None:
        """Get the current session."""
        return self._session

    @property
    def frames(self) -> list[Frame]:
        """Get all recorded frames as a property."""
        return self.get_frames()

    def fork(
        self,
        from_frame: int,
        new_session_id: str | None = None,
        state_overrides: dict | None = None,
    ) -> VCRRecorder:
        """Create a forked recorder starting from a specific frame."""
        all_frames = self.get_frames()
        if from_frame < 0 or from_frame >= len(all_frames):
            raise ValueError(f"Frame {from_frame} not found (have {len(all_frames)} frames)")

        target_frame = all_frames[from_frame]
        base_state = StateSerializer.deserialize(target_frame.output_state)
        forked_state = dict(base_state)
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

        new_recorder.record_step(
            node_name=f"fork:{target_frame.node_name}",
            input_state=base_state,
            output_state=forked_state,
            metadata=FrameMetadata(
                custom={
                    "source": "fork",
                    "forked_from_frame": from_frame,
                    "forked_from_frame_id": target_frame.frame_id,
                }
            ),
            frame_type=FrameType.CHECKPOINT,
            parent_frame_id=target_frame.frame_id,
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
        """Write the session header to the file (durable: flush + fsync)."""
        if self._session is None:
            return

        path = self._get_session_path()
        with open(path, "w") as f:
            header = {
                "type": "session",
                "schema": "execution/v1",
                "data": self._session.model_dump(),
            }
            f.write(json.dumps(header, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if self._session is not None:
            self._header_frame_count = self._session.frame_count

    def _flush_frames(self) -> None:
        """Write buffered frames to disk.

        Uses fsync to ensure frames are durable on disk before clearing
        the in-memory buffer.  Without this, a crash between write() and
        the OS flushing kernel buffers would silently lose frames.
        """
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
            f.flush()
            os.fsync(f.fileno())

        self._frames = []

    def _rewrite_file(self) -> None:
        """Rewrite header + all frames atomically (fixes header staleness).

        The JSONL header embeds ``frame_count`` at write time, so any
        frames appended afterwards (buffered saves, crash-resume attach)
        leave it stale. Rewriting from the in-memory cache restores the
        ``verify_integrity`` contract. Runs only when counts diverge.
        """
        if self._session is None:
            return
        path = self._get_session_path()
        frames = self._cache.get_frames(self._session.session_id)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_dir), suffix=".tmp", prefix="vcr_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                header = {
                    "type": "session",
                    "schema": "execution/v1",
                    "data": self._session.model_dump(),
                }
                f.write(json.dumps(header, separators=(",", ":")) + "\n")
                for frame in frames:
                    line = {"type": "frame", "data": frame.model_dump()}
                    f.write(json.dumps(line, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
            self._header_frame_count = self._session.frame_count
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

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
        """Compute the diff between two states (canonical JSON-patch view)."""
        return StateDiff.to_json_patch(before, after)
