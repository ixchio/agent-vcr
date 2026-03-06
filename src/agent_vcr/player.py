"""VCR Player - Playback and time-travel for recorded sessions."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from agent_vcr.models import (  # noqa: E402
    Frame,
    ResumeConfig,
    ResumeMode,
    Session,
    StateSerializer,
)
from agent_vcr.recorder import VCRRecorder  # noqa: E402


class VCRPlayer:
    """Plays back recorded VCR sessions with time-travel capabilities."""

    def __init__(self, session: Session, frames: list[Frame]):
        self.session = session
        self.frames = frames
        self._current_index = 0

    @classmethod
    def load(cls, filepath: str | Path) -> VCRPlayer:
        """Load a VCR file and reconstruct the session."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"VCR file not found: {filepath}")

        logger.info("Loading VCR file: %s", filepath)
        session: Session | None = None
        frames: list[Frame] = []

        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    record_type = record.get("type")
                    data = record.get("data", {})

                    if record_type == "session":
                        session = Session(**data)
                    elif record_type == "frame":
                        frame = Frame(**data)
                        frames.append(frame)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d in %s", line_num, filepath)
                    continue

        if session is None:
            raise ValueError(f"No session header found in {filepath}")

        frames.sort(key=lambda f: f.timestamp)
        logger.info("Loaded session %s with %d frames", session.session_id, len(frames))

        return cls(session, frames)

    @classmethod
    def load_by_id(cls, session_id: str, vcr_dir: str = ".vcr") -> VCRPlayer:
        """Load a session by its ID."""
        filepath = Path(vcr_dir) / f"{session_id}.vcr"
        return cls.load(filepath)

    def goto_frame(self, index: int) -> dict[str, Any]:
        """Jump to a specific frame and return its output state."""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range (0-{len(self.frames)-1})")

        self._current_index = index
        frame = self.frames[index]

        return StateSerializer.deserialize(frame.output_state)

    def goto_time(self, timestamp: str | datetime) -> dict[str, Any]:
        """Jump to the frame closest to a timestamp.

        Args:
            timestamp: ISO 8601 timestamp string or datetime object.

        Returns:
            The output state at the closest frame.
        """
        target_time = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

        # Ensure timezone-aware comparison
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        target_idx = 0
        min_diff = float("inf")

        for i, frame in enumerate(self.frames):
            frame_time = frame.timestamp
            if frame_time.tzinfo is None:
                frame_time = frame_time.replace(tzinfo=timezone.utc)
            diff = abs((frame_time - target_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                target_idx = i

        logger.debug("goto_time: closest frame is %d (diff=%.3fs)", target_idx, min_diff)
        return self.goto_frame(target_idx)

    def get_frame(self, index: int) -> Frame:
        """Get a frame by index."""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range")
        return self.frames[index]

    def get_current_state(self) -> dict[str, Any]:
        """Get the state at the current position."""
        if not self.frames:
            return {}
        return StateSerializer.deserialize(self.frames[self._current_index].output_state)

    def get_input_state(self, index: int) -> dict[str, Any]:
        """Get the input state for a frame."""
        frame = self.get_frame(index)
        return StateSerializer.deserialize(frame.input_state)

    def get_output_state(self, index: int) -> dict[str, Any]:
        """Get the output state for a frame."""
        frame = self.get_frame(index)
        return StateSerializer.deserialize(frame.output_state)

    def get_state_at_node(self, node_name: str) -> dict[str, Any] | None:
        """Get the state after a specific node execution."""
        for frame in self.frames:
            if frame.node_name == node_name:
                return StateSerializer.deserialize(frame.output_state)
        return None

    def list_nodes(self) -> list[str]:
        """List all node names in the session."""
        return list(dict.fromkeys(frame.node_name for frame in self.frames))

    def get_node_executions(self, node_name: str) -> list[Frame]:
        """Get all executions of a specific node."""
        return [frame for frame in self.frames if frame.node_name == node_name]

    def get_errors(self) -> list[Frame]:
        """Get all error frames."""
        from agent_vcr.models import FrameType
        return [frame for frame in self.frames if frame.frame_type == FrameType.ERROR]

    def get_total_latency(self) -> float:
        """Get total execution latency in milliseconds."""
        return sum(frame.metadata.latency_ms for frame in self.frames)

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(
            frame.metadata.tokens_used or 0
            for frame in self.frames
        )

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        return sum(
            frame.metadata.cost_usd or 0
            for frame in self.frames
        )

    def resume(
        self,
        agent_callable: Callable[[dict[str, Any]], dict[str, Any]],
        config: ResumeConfig | None = None,
        recorder: VCRRecorder | None = None,
    ) -> str:
        """Resume execution from a specific frame."""
        config = config or ResumeConfig(from_frame=self._current_index)

        if config.from_frame < 0 or config.from_frame >= len(self.frames):
            raise IndexError(f"Frame index {config.from_frame} out of range")

        logger.info(
            "Resuming from frame %d (mode=%s, overrides=%d keys)",
            config.from_frame,
            config.mode.value,
            len(config.state_overrides),
        )

        target_frame = self.frames[config.from_frame]
        base_state = StateSerializer.deserialize(target_frame.output_state)

        if config.state_overrides:
            base_state.update(config.state_overrides)

        if recorder is None:
            recorder = VCRRecorder()

        new_session = recorder.start_session(
            session_id=config.new_session_id,
            parent_session_id=self.session.session_id,
            forked_from_frame=config.from_frame,
        )

        if config.mode == ResumeMode.MOCK:
            self._execute_with_mocks(
                agent_callable, base_state, config.inject_mocks, recorder
            )
        elif config.mode == ResumeMode.REPLAY:
            self._execute_replay(
                agent_callable, config.from_frame, recorder
            )
        else:
            self._execute_fresh(agent_callable, base_state, recorder)

        recorder.save()

        return new_session.session_id

    def compare_frames(self, frame_a: int, frame_b: int) -> dict[str, Any]:
        """Compare two frames and return differences."""
        state_a = self.get_output_state(frame_a)
        state_b = self.get_output_state(frame_b)

        return self._compute_state_diff(state_a, state_b)

    def export_state(self, frame_index: int | None = None) -> dict[str, Any]:
        """Export state at a frame as a clean dictionary."""
        idx = frame_index if frame_index is not None else self._current_index
        return self.get_output_state(idx)

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire session to a dictionary."""
        return {
            "session": self.session.model_dump(),
            "frames": [frame.model_dump() for frame in self.frames],
            "statistics": {
                "total_frames": len(self.frames),
                "total_latency_ms": self.get_total_latency(),
                "total_tokens": self.get_total_tokens(),
                "total_cost_usd": self.get_total_cost(),
                "nodes": self.list_nodes(),
            },
        }

    def _execute_fresh(
        self,
        agent_callable: Callable[[dict[str, Any]], dict[str, Any]],
        initial_state: dict[str, Any],
        recorder: VCRRecorder,
    ) -> dict[str, Any]:
        """Execute agent with fresh state."""
        start_time = time.perf_counter()

        try:
            result = agent_callable(initial_state)
            latency_ms = (time.perf_counter() - start_time) * 1000

            recorder.record_step(
                node_name="resumed_execution",
                input_state=initial_state,
                output_state=result,
                metadata={"latency_ms": latency_ms},
            )

            return result
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            recorder.record_error(
                node_name="resumed_execution",
                input_state=initial_state,
                error=e,
                latency_ms=latency_ms,
            )
            raise

    def _execute_with_mocks(
        self,
        agent_callable: Callable[[dict[str, Any]], dict[str, Any]],
        initial_state: dict[str, Any],
        mocks: dict[str, Any],
        recorder: VCRRecorder,
    ) -> dict[str, Any]:
        """Execute agent with mocked dependencies."""
        state = {**initial_state, "__mocks__": mocks}
        return self._execute_fresh(agent_callable, state, recorder)

    def _execute_replay(
        self,
        agent_callable: Callable[[dict[str, Any]], dict[str, Any]],
        from_frame: int,
        recorder: VCRRecorder,
    ) -> dict[str, Any]:
        """Replay execution up to a frame, then continue."""
        current_state = StateSerializer.deserialize(self.frames[0].input_state)

        for i in range(from_frame):
            frame = self.frames[i]
            current_state = StateSerializer.deserialize(frame.output_state)

            recorder.record_step(
                node_name=frame.node_name,
                input_state=frame.input_state,
                output_state=frame.output_state,
                metadata=frame.metadata,
            )

        return self._execute_fresh(agent_callable, current_state, recorder)

    def _compute_state_diff(
        self, state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute differences between two states."""
        diff: dict[str, Any] = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {},
        }

        all_keys = set(state_a.keys()) | set(state_b.keys())

        for key in all_keys:
            if key not in state_a:
                diff["added"][key] = state_b[key]
            elif key not in state_b:
                diff["removed"][key] = state_a[key]
            elif state_a[key] != state_b[key]:
                diff["modified"][key] = {
                    "before": state_a[key],
                    "after": state_b[key],
                }
            else:
                diff["unchanged"][key] = state_a[key]

        return diff
