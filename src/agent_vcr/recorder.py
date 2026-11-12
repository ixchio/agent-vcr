"""VCR Recorder - Records agent execution to .vcr files."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import contextlib  # noqa: E402

from agent_vcr.models import (  # noqa: E402
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

_IF_EXISTS_CHOICES = frozenset({"rotate", "overwrite", "error"})


def _validate_session_id(session_id: str) -> None:
    """Reject ids that would escape the output directory."""
    if (
        not session_id
        or session_id in {".", ".."}
        or any(ch in session_id for ch in ("/", "\\", "\0"))
    ):
        raise ValueError(f"Invalid session_id {session_id!r}: must be a plain file name")


def _handle_existing_recording(path: Path, if_exists: str) -> None:
    """Apply the ``if_exists`` policy before a header would truncate ``path``."""
    try:
        has_data = path.stat().st_size > 0
    except FileNotFoundError:
        return
    if not has_data or if_exists == "overwrite":
        return
    if if_exists == "error":
        raise FileExistsError(
            f"Recording already exists: {path}. Use attach_session() to append, "
            "or start_session(if_exists='overwrite'|'rotate')."
        )
    backup = path.with_name(path.name + ".prev")
    os.replace(path, backup)
    logger.warning("Session file %s already existed; previous run moved to %s", path, backup)


@contextlib.contextmanager
def _manifest_lock(output_dir: Path) -> Iterator[None]:
    """Exclusive cross-process lock for manifest read-modify-write (POSIX flock).

    Without it, parallel recorders (multiple agents, pytest-xdist) each read
    the manifest, add their session and replace it — losing each other's
    entries. On platforms without ``fcntl`` this degrades to no locking.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        yield
        return
    with open(output_dir / ".manifest.lock", "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def update_manifest(output_dir: Path, session_data: dict[str, Any]) -> None:
    """Insert/replace one session entry in ``manifest.json`` (locked + atomic)."""
    manifest_path = output_dir / "manifest.json"
    with _manifest_lock(output_dir):
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    loaded = json.load(f)
                manifest = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read manifest, starting fresh: %s", e)
        sessions = [s for s in manifest.get("sessions", []) if isinstance(s, dict)]
        idx = next(
            (i for i, s in enumerate(sessions) if s.get("session_id") == session_data["session_id"]),
            None,
        )
        if idx is None:
            sessions.append(session_data)
        else:
            sessions[idx] = session_data  # keep position
        manifest["sessions"] = sessions
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

        fd, tmp_path = tempfile.mkstemp(dir=str(output_dir), suffix=".tmp", prefix="manifest_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp_path, str(manifest_path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise


def stored_form(*values: Any) -> tuple[Any, ...]:
    """``values`` as they come back from a ``.vcr`` file (a JSON round trip).

    Bounded recorders mix cached frames with frames read back from disk; passing the cached
    ones through this makes both identical (e.g. ``{1: x}`` is stored as ``{"1": x}``).
    """
    return tuple(json.loads(json_dumps_safe(value)) for value in values)


def iter_frames_on_disk(path: Path, start: int = 0, stop: int | None = None) -> Iterator[Frame]:
    """Stream the valid frames ``[start, stop)`` of a ``.vcr`` file (player skip rules).

    Used by bounded-memory recorders to read back frames they evicted from RAM.
    """
    if stop is not None and stop <= start:
        return
    index = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                if not isinstance(record, dict) or record.get("type") != "frame":
                    continue
                if index < start:
                    # Still validate: the player's index space only counts valid frames.
                    Frame(**record.get("data", {}))
                    index += 1
                    continue
                frame = Frame(**record.get("data", {}))
            except Exception:
                continue  # malformed / invalid line: the player skips it too
            yield frame
            index += 1
            if stop is not None and index >= stop:
                return


def rewrite_session_file_from_disk(
    path: Path, session: Session, expected_frames: int | None = None
) -> int:
    """Atomically rewrite ``path`` with a fresh header, streaming its frames from disk.

    Memory stays O(1) in the number of frames: frame lines are copied verbatim, so a
    bounded recorder never has to hold the whole session. Malformed lines and stale
    headers are dropped. When ``expected_frames`` is given and a fast copy (JSON check
    only) disagrees with it, the copy is redone keeping only frames that validate, as the
    player would load them. Returns the number of frames written.
    """
    copied = _stream_rewrite(path, session, validate=False)
    if expected_frames is not None and copied != expected_frames:
        copied = _stream_rewrite(path, session, validate=True)
    return copied


def _stream_rewrite(path: Path, session: Session, *, validate: bool) -> int:
    header = {"type": "session", "schema": "execution/v1", "data": session.model_dump()}
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix="vcr_")
    copied = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out, open(path, encoding="utf-8") as src:
            out.write(json_dumps_safe(header) + "\n")
            for line in src:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                    if not isinstance(record, dict) or record.get("type") != "frame":
                        continue
                    if validate:
                        Frame(**record.get("data", {}))
                except Exception:
                    continue
                out.write(text + "\n")
                copied += 1
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return copied


def rewrite_session_file(path: Path, session: Session, frames: list[Frame]) -> None:
    """Atomically rewrite a ``.vcr`` file from a session header + frames."""
    header = {"type": "session", "schema": "execution/v1", "data": session.model_dump()}
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix="vcr_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json_dumps_safe(header) + "\n")
            for frame in frames:
                f.write(json_dumps_safe({"type": "frame", "data": frame.model_dump()}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class VCRRecorder:
    """Records agent execution sessions to .vcr files."""

    def __init__(
        self,
        output_dir: str = ".vcr",
        buffer_size: int = 100,
        auto_save: bool = True,
        diff_mode: bool = False,
        on_frame_recorded: Any | None = None,
        max_cached_frames: int | None = None,
    ):
        """
        Args:
            max_cached_frames: Keep at most this many already-written frames in RAM
                (plus the unflushed buffer, up to ``buffer_size``). Older frames are
                evicted once they are on disk and read back from the ``.vcr`` file when
                :meth:`get_frames`, :meth:`fork` or a golden-cache save need them.
                ``None`` (default) keeps every frame in memory, as before.
        """
        if max_cached_frames is not None and max_cached_frames < 0:
            raise ValueError("max_cached_frames must be >= 0 or None")
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.max_cached_frames = max_cached_frames
        self.auto_save = auto_save
        self.diff_mode = diff_mode
        self._on_frame_recorded = on_frame_recorded

        self._session: Session | None = None
        self._frames: list[Frame] = []
        self._previous_state: dict | None = None
        self._lock = threading.RLock()
        self._cache = VCRCache()
        self._header_frame_count: int | None = None
        # Frames of the current session evicted from the cache (they are on disk only);
        # the cache holds frames [_evicted, frame_count).
        self._evicted = 0

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
        if_exists: str = "rotate",
    ) -> Session:
        """Start a new recording session.

        Args:
            if_exists: What to do when ``<session_id>.vcr`` already has data:
                ``"rotate"`` (default) moves it to ``<session_id>.vcr.prev``
                so the previous run is never silently destroyed;
                ``"overwrite"`` replaces it; ``"error"`` raises
                ``FileExistsError``. Use :meth:`attach_session` to append.
        """
        if if_exists not in _IF_EXISTS_CHOICES:
            raise ValueError(f"if_exists must be one of {sorted(_IF_EXISTS_CHOICES)}")
        with self._lock:
            if self._session is not None and self._frames:
                logger.warning(
                    "start_session() called with %d unsaved frames in session %s; saving them",
                    len(self._frames),
                    self._session.session_id,
                )
                self.save()
            sid = session_id or self._generate_session_id()
            _validate_session_id(sid)
            _handle_existing_recording(self.output_dir / f"{sid}.vcr", if_exists)
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

            if isinstance(metadata, dict):
                metadata = FrameMetadata(**metadata)
            elif metadata is None:
                metadata = FrameMetadata()
            else:
                # Never mutate the caller's object (e.g. a replayed frame's metadata).
                metadata = metadata.model_copy(update={"custom": dict(metadata.custom)})
            if self.max_cached_frames is not None:
                serialized_input, serialized_output, metadata.custom = stored_form(
                    serialized_input, serialized_output, metadata.custom
                )

            state_diff = None
            if self.diff_mode and self._previous_state is not None:
                state_diff = self._compute_diff(
                    self._previous_state, serialized_output
                )
                # Omit input_state only when it duplicates the previous
                # output_state; the explicit marker tells the player whether
                # to reconstruct it from the prior frame. Inputs that differ
                # (LLM prompts, tool args) are always stored.
                elided = serialized_input == self._previous_state
                metadata.custom[INPUT_FROM_PREVIOUS_KEY] = elided
                if elided:
                    serialized_input = {}

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

            self._session.frame_count = self._evicted + len(
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
            _validate_session_id(session_id)
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
            # Everything loaded here is already on disk, so a bounded cache keeps the tail.
            self._evicted = (
                0 if self.max_cached_frames is None
                else max(0, len(player.frames) - self.max_cached_frames)
            )
            for frame in player.frames[self._evicted:]:
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
        """Get all recorded frames (including already-flushed ones).

        With ``max_cached_frames`` set, evicted frames are read back from the ``.vcr``
        file, so this costs a file read once eviction has started, and the result is a
        snapshot list (eviction trims the cache in place). Unbounded, it is the live list
        of the session's frames, which only grows.
        """
        with self._lock:
            if not self._session:
                return self._frames.copy()
            cached = self._cache.get_frames(self._session.session_id)
            if self.max_cached_frames is None:
                return cached
            if not self._evicted:
                return list(cached)
            on_disk = list(iter_frames_on_disk(self._get_session_path(), 0, self._evicted))
            return on_disk + list(cached)

    @property
    def cached_frame_count(self) -> int:
        """Number of frames of the current session held in memory."""
        if not self._session:
            return len(self._frames)
        return len(self._cache.get_frames(self._session.session_id))

    def _frame_at(self, index: int) -> Frame:
        """Frame ``index`` of the current session, from the cache or the file."""
        with self._lock:
            if self._session is None:
                return self._frames[index]
            if index >= self._evicted:
                return self._cache.get_frames(self._session.session_id)[index - self._evicted]
            for frame in iter_frames_on_disk(self._get_session_path(), index, index + 1):
                return frame
            raise ValueError(f"Frame {index} not found in {self._get_session_path()}")

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
        in_place: bool = False,
    ) -> VCRRecorder:
        """Fork a new session that starts from frame ``from_frame``'s **output** state.

        (``VCRPlayer.resume(from_frame=n)`` instead re-runs frame *n* from its
        *input* state; use this method to continue *after* a frame.)

        Args:
            in_place: If True, save the current session and continue recording
                the fork on *this* recorder object (returned for chaining), so
                framework wrappers holding a reference keep recording into the
                new branch. Otherwise a new recorder is returned and this one
                is left untouched.
        """
        total = self._session.frame_count if self._session else len(self._frames)
        if from_frame < 0 or from_frame >= total:
            raise ValueError(f"Frame {from_frame} not found (have {total} frames)")

        target_frame = self._frame_at(from_frame)
        base_state = StateSerializer.deserialize(target_frame.output_state)
        forked_state = dict(base_state)
        if state_overrides:
            forked_state.update(state_overrides)
        parent_id = self._session.session_id if self._session else None

        if in_place:
            if self._session is not None:
                self.save()
            new_recorder = self
        else:
            new_recorder = VCRRecorder(
                output_dir=str(self.output_dir),
                buffer_size=self.buffer_size,
                auto_save=self.auto_save,
                diff_mode=self.diff_mode,
                on_frame_recorded=self._on_frame_recorded,
                max_cached_frames=self.max_cached_frames,
            )

        new_recorder.start_session(
            session_id=new_session_id,
            parent_session_id=parent_id,
            forked_from_frame=from_frame,
            metadata={"forked_from": parent_id},
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
        header = {
            "type": "session",
            "schema": "execution/v1",
            "data": self._session.model_dump(),
        }
        payload = json_dumps_safe(header) + "\n"
        fd, tmp_path = tempfile.mkstemp(dir=str(self.output_dir), suffix=".tmp", prefix="vcr_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
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
        # Serialize everything first: a failure here must not leave a
        # partially-written batch that would be re-appended on the next flush.
        payload = "".join(
            json_dumps_safe({"type": "frame", "data": frame.model_dump()}) + "\n"
            for frame in self._frames
        )
        with open(path, "a") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

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

    def _rewrite_file(self) -> None:
        """Rewrite header + all frames atomically (fixes header staleness).

        The JSONL header embeds ``frame_count`` at write time, so any
        frames appended afterwards (buffered saves, crash-resume attach)
        leave it stale. Frames are streamed from the file itself (``save()``
        flushes first), so memory does not grow with the session. If the
        file lost frames (deleted underneath us) while the whole session is
        still cached, it is rebuilt from the cache. Runs only when counts diverge.
        """
        if self._session is None:
            return
        path = self._get_session_path()
        expected = self._session.frame_count
        copied = rewrite_session_file_from_disk(path, self._session, expected)
        if copied != expected:
            if not self._evicted:
                rewrite_session_file(
                    path, self._session, self._cache.get_frames(self._session.session_id)
                )
            else:
                logger.warning(
                    "%s has %d frames, expected %d; evicted frames cannot be restored",
                    path, copied, expected,
                )
        self._header_frame_count = self._session.frame_count

    def _update_session_manifest(self) -> None:
        """Update the session manifest (atomic write, cross-process locked)."""
        if self._session is not None:
            update_manifest(self.output_dir, self._session.model_dump(mode="json"))

    def _compute_diff(self, before: dict, after: dict) -> list[dict]:
        """Compute the diff between two states (canonical JSON-patch view)."""
        return StateDiff.to_json_patch(before, after)
