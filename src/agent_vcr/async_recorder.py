"""Async VCR Recorder - Records agent execution using async I/O."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from agent_vcr.models import (
    INPUT_FROM_PREVIOUS_KEY,
    Frame,
    FrameMetadata,
    FrameType,
    Session,
    StateDiff,
    StateSerializer,
    VCRCache,
    json_dumps_safe,
)
from agent_vcr.recorder import (
    _IF_EXISTS_CHOICES,
    _handle_existing_recording,
    _validate_session_id,
    iter_frames_on_disk,
    rewrite_session_file,
    rewrite_session_file_from_disk,
    update_manifest,
)

logger = logging.getLogger(__name__)

#: Upper bound for reading evicted frames back from disk / rewriting a session file.
_DISK_IO_TIMEOUT_S = 300.0


class AsyncVCRRecorder:
    """Async version of VCRRecorder for recording agent execution.

    Uses asyncio.Lock for thread safety and aiofiles for non-blocking I/O.
    """

    def __init__(
        self,
        output_dir: str | Path = ".vcr",
        auto_save: bool = True,
        buffer_size: int = 100,
        diff_mode: bool = False,
        on_frame_recorded: Any | None = None,
        max_cached_frames: int | None = None,
    ):
        """
        Args:
            max_cached_frames: Bounded memory, as in :class:`~agent_vcr.recorder.VCRRecorder`:
                keep at most this many already-written frames in RAM; older ones are read
                back from the ``.vcr`` file when needed. ``None`` (default) keeps all.
        """
        if max_cached_frames is not None and max_cached_frames < 0:
            raise ValueError("max_cached_frames must be >= 0 or None")
        self.max_cached_frames = max_cached_frames
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save
        self.buffer_size = buffer_size
        self.diff_mode = diff_mode
        self._on_frame_recorded = on_frame_recorded

        self._session: Session | None = None
        self._frames: list[Frame] = []
        self._previous_state: dict | None = None
        self._lock = asyncio.Lock()
        self._cache = VCRCache()
        self._header_frame_count: int | None = None
        # Frames of the current session evicted from the cache (on disk only).
        self._evicted = 0

    async def __aenter__(self) -> AsyncVCRRecorder:
        """Support ``async with`` usage — auto-saves on exit."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Auto-save session on async context manager exit."""
        if self._session is not None:
            try:
                await self.save()
            except Exception:
                logger.warning("Failed to auto-save async session on exit", exc_info=True)

    async def start_session(
        self,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        forked_from_frame: int | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        if_exists: str = "rotate",
    ) -> Session:
        """Start a new recording session (``if_exists`` as in the sync recorder)."""
        if if_exists not in _IF_EXISTS_CHOICES:
            raise ValueError(f"if_exists must be one of {sorted(_IF_EXISTS_CHOICES)}")
        async with self._lock:
            if self._session is not None and self._frames:
                logger.warning(
                    "start_session() called with %d unsaved frames in session %s; flushing",
                    len(self._frames),
                    self._session.session_id,
                )
                await self._flush_frames()
            sid = session_id or self._generate_session_id()
            _validate_session_id(sid)
            await asyncio.to_thread(
                _handle_existing_recording, self.output_dir / f"{sid}.vcr", if_exists
            )
            self._session = Session(
                session_id=sid,
                parent_session_id=parent_session_id,
                forked_from_frame=forked_from_frame,
                metadata=metadata or {},
                tags=tags or [],
            )
            self._frames = []
            self._previous_state = None
            # A fresh cache: frames of earlier sessions are on disk and must not pile up.
            self._cache = VCRCache()
            self._evicted = 0
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
        parent_frame_id: str | None = None,
    ) -> Frame:
        """Record a single execution step."""
        async with self._lock:
            if self._session is None:
                raise RuntimeError("No active session. Call start_session() first.")

            serialized_input = StateSerializer.serialize(input_state)
            serialized_output = StateSerializer.serialize(output_state)

            if isinstance(metadata, dict):
                metadata = FrameMetadata(**metadata)
            elif metadata is None:
                metadata = FrameMetadata()
            else:
                metadata = metadata.model_copy(update={"custom": dict(metadata.custom)})

            state_diff = None
            if self.diff_mode and self._previous_state is not None:
                state_diff = self._compute_diff(self._previous_state, serialized_output)
                # Only elide input that duplicates the previous output (see sync recorder).
                elided = serialized_input == self._previous_state
                metadata.custom[INPUT_FROM_PREVIOUS_KEY] = elided
                if elided:
                    serialized_input = {}

            frame = Frame(
                session_id=self._session.session_id,
                parent_frame_id=parent_frame_id,
                node_name=node_name,
                input_state=serialized_input,
                output_state=serialized_output,
                frame_type=frame_type,
                metadata=metadata,
                state_diff=state_diff,
            )

            self._frames.append(frame)
            self._cache.add_frame(self._session.session_id, frame)
            self._previous_state = serialized_output

            # Update session statistics
            self._session.frame_count = self._evicted + len(
                self._cache.get_frames(self._session.session_id)
            )
            self._session.updated_at = datetime.now(timezone.utc)
            if metadata.tokens_used:
                self._session.total_tokens += metadata.tokens_used
            if metadata.cost_usd:
                self._session.total_cost_usd += metadata.cost_usd

            if self.auto_save and len(self._frames) % self.buffer_size == 0:
                logger.debug("Auto-flushing %d frames to disk", len(self._frames))
                await self._flush_frames()

        # Fire the live-push callback *outside* the lock: asyncio.Lock is not
        # reentrant, so a callback that records (or a slow websocket push)
        # would otherwise deadlock or stall every other recorder call.
        if self._on_frame_recorded is not None:
            try:
                result = self._on_frame_recorded(frame)
                # Support both sync and async callbacks
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("on_frame_recorded callback error", exc_info=True)

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
            node_name=f"llm:{model}",
            input_state={"messages": messages},
            output_state={"response": response},
            metadata=metadata,
            frame_type=FrameType.LLM_CALL,
        )

    async def record_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> Frame:
        """Record a tool call."""
        metadata = FrameMetadata(
            latency_ms=latency_ms,
            error_message=error,
            error_type="ToolError" if error else None,
        )
        input_state = tool_input if isinstance(tool_input, dict) else {"value": tool_input}
        return await self.record_step(
            node_name=f"tool:{tool_name}",
            input_state=input_state,
            output_state={"result": tool_output, "error": error},
            metadata=metadata,
            frame_type=FrameType.TOOL_CALL,
        )

    async def record_error(
        self,
        node_name: str,
        input_state: dict[str, Any],
        error: Exception,
        latency_ms: float = 0.0,
        model: str | None = None,
    ) -> Frame:
        """Record an error during execution (model attribution optional)."""
        metadata = FrameMetadata(
            error_type=type(error).__name__,
            error_message=str(error),
            latency_ms=latency_ms,
            model=model,
        )
        return await self.record_step(
            node_name=node_name,
            input_state=input_state,
            output_state={"error": str(error)},
            metadata=metadata,
            frame_type=FrameType.ERROR,
        )

    async def record_checkpoint(
        self,
        state: dict[str, Any],
        label: str = "checkpoint",
        node_name: str | None = None,
        metadata: FrameMetadata | None = None,
    ) -> Frame:
        """Record an explicit named checkpoint (async mirror of sync API)."""
        meta = metadata or FrameMetadata()
        custom = dict(meta.custom)
        custom["checkpoint_label"] = label
        meta.custom = custom
        snapshot = StateSerializer.deserialize(StateSerializer.serialize(state))
        return await self.record_step(
            node_name=node_name or f"checkpoint:{label}",
            input_state=snapshot,
            output_state=snapshot,
            metadata=meta,
            frame_type=FrameType.CHECKPOINT,
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
            if self._header_frame_count != self._session.frame_count:
                # Header totals (frame_count, tokens, cost) were written at
                # session start; rewrite atomically so they match the frames.
                # Frames are streamed from the file, not from memory.
                await asyncio.wait_for(
                    asyncio.to_thread(self._rewrite_file, path, self._session.model_copy(),
                                      self._evicted,
                                      list(self._cache.get_frames(self._session.session_id))),
                    timeout=_DISK_IO_TIMEOUT_S,
                )
                self._header_frame_count = self._session.frame_count
            await self._update_session_manifest()

            logger.info("Saved async session %s to %s", self._session.session_id, path)
            return path

    def get_session(self) -> Session | None:
        return self._session

    def get_frames(self) -> list[Frame]:
        """Get all recorded frames (including already-flushed ones).

        With ``max_cached_frames`` set, evicted frames are read back from the ``.vcr``
        file with blocking I/O; in a coroutine prefer ``await load_frames()``.
        """
        if not self._session:
            return list(self._frames)
        cached = self._cache.get_frames(self._session.session_id)
        if not self._evicted:
            return cached
        on_disk = list(iter_frames_on_disk(self._get_session_path(), 0, self._evicted))
        return on_disk + list(cached)

    async def load_frames(self) -> list[Frame]:
        """All frames, reading evicted ones from disk off the event loop."""
        if not self._session or not self._evicted:
            return list(self.get_frames())
        evicted = self._evicted
        cached = list(self._cache.get_frames(self._session.session_id))
        path = self._get_session_path()
        on_disk = await asyncio.wait_for(
            asyncio.to_thread(lambda: list(iter_frames_on_disk(path, 0, evicted))),
            timeout=_DISK_IO_TIMEOUT_S,
        )
        return on_disk + cached

    @property
    def cached_frame_count(self) -> int:
        """Number of frames of the current session held in memory."""
        if not self._session:
            return len(self._frames)
        return len(self._cache.get_frames(self._session.session_id))

    async def _frame_at(self, index: int) -> Frame:
        """Frame ``index`` of the current session, from the cache or the file."""
        assert self._session is not None
        if index >= self._evicted:
            return self._cache.get_frames(self._session.session_id)[index - self._evicted]
        path = self._get_session_path()
        found = await asyncio.wait_for(
            asyncio.to_thread(lambda: list(iter_frames_on_disk(path, index, index + 1))),
            timeout=_DISK_IO_TIMEOUT_S,
        )
        if not found:
            raise ValueError(f"Frame {index} not found in {path}")
        return found[0]

    @property
    def frames(self) -> list[Frame]:
        """Get all recorded frames as a property."""
        return self.get_frames()

    async def fork(
        self,
        from_frame: int,
        new_session_id: str | None = None,
        state_overrides: dict | None = None,
    ) -> AsyncVCRRecorder:
        """Fork a new recorder from a specific frame."""
        if self._session is None:
            raise RuntimeError("No active session to fork from.")

        total = self._session.frame_count
        if from_frame < 0 or from_frame >= total:
            raise ValueError(f"Frame {from_frame} not found (have {total} frames)")

        target_frame = await self._frame_at(from_frame)
        base_state = StateSerializer.deserialize(target_frame.output_state)
        forked_state = dict(base_state)
        if state_overrides:
            forked_state.update(state_overrides)

        new_recorder = AsyncVCRRecorder(
            output_dir=self.output_dir,
            auto_save=self.auto_save,
            buffer_size=self.buffer_size,
            diff_mode=self.diff_mode,
            max_cached_frames=self.max_cached_frames,
        )

        await new_recorder.start_session(
            session_id=new_session_id,
            parent_session_id=self._session.session_id,
            forked_from_frame=from_frame,
            metadata={"forked_from": self._session.session_id},
        )
        await new_recorder.record_step(
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

    # --- Private methods ---

    def _generate_session_id(self) -> str:
        import os
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = os.urandom(4).hex()
        return f"{timestamp}_{unique_id}"

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
            "schema": "execution/v1",
            "data": self._session.model_dump(),
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json_dumps_safe(record) + "\n")
            await f.flush()
        await asyncio.to_thread(self._fsync_path, path)
        self._header_frame_count = self._session.frame_count

    async def _flush_frames(self) -> None:
        """Write buffered frames to disk.

        Performs an fsync after writing to ensure frames are durable on
        disk before clearing the in-memory buffer.
        """
        if not self._frames:
            return
        path = self._get_session_path()
        # Serialize the whole batch before writing so a failure can't leave
        # a partial batch on disk that gets re-appended on the next flush.
        payload = "".join(
            json_dumps_safe({"type": "frame", "data": frame.model_dump()}) + "\n"
            for frame in self._frames
        )
        async with aiofiles.open(path, "a") as f:
            await f.write(payload)
            await f.flush()
        # fsync must use the real fd — aiofiles doesn't expose fileno(),
        # so we open briefly just for the sync.  os.replace-style atomic
        # writes aren't practical for append-mode JSONL.
        await asyncio.to_thread(self._fsync_path, path)
        self._frames = []
        self._evict_flushed()

    def _evict_flushed(self) -> None:
        """Drop the oldest written frames beyond ``max_cached_frames`` from memory."""
        if self.max_cached_frames is None or self._session is None:
            return
        cached = self._cache.get_frames(self._session.session_id)
        excess = min(len(cached) - self.max_cached_frames, len(cached) - len(self._frames))
        if excess > 0:
            del cached[:excess]
            self._evicted += excess

    @staticmethod
    def _rewrite_file(path: Path, session: Session, evicted: int,
                      cached: list[Frame]) -> None:
        """Stream-rewrite ``path`` (see ``VCRRecorder._rewrite_file``); runs in a thread."""
        copied = rewrite_session_file_from_disk(path, session, session.frame_count)
        if copied == session.frame_count:
            return
        if not evicted:
            rewrite_session_file(path, session, cached)
        else:
            logger.warning(
                "%s has %d frames, expected %d; evicted frames cannot be restored",
                path, copied, session.frame_count,
            )

    @staticmethod
    def _fsync_path(path: Path) -> None:
        """Open a file briefly to fsync it (used after async writes)."""
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    async def _update_session_manifest(self) -> None:
        """Update manifest.json off the event loop (locked + atomic, shared with sync)."""
        if self._session is None:
            return
        await asyncio.to_thread(
            update_manifest, self.output_dir, self._session.model_dump(mode="json")
        )

    def _compute_diff(self, before: dict, after: dict) -> list[dict]:
        """Compute the diff between two states (canonical JSON-patch view)."""
        return StateDiff.to_json_patch(before, after)
