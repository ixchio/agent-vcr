"""Async VCR Player - Non-blocking playback and time-travel for recorded sessions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from agent_vcr.models import (
    Frame,
    Session,
    StateSerializer,
)

logger = logging.getLogger(__name__)


class AsyncVCRPlayer:
    """Async version of VCRPlayer for non-blocking load and playback."""

    def __init__(
        self,
        session: Session,
        frames: list[Frame],
        *,
        source_path: str | Path | None = None,
        skipped_lines: int = 0,
    ):
        self.session = session
        self.frames = frames
        self._current_index = 0
        self.source_path = Path(source_path) if source_path is not None else None
        self.skipped_lines = skipped_lines

    @classmethod
    async def load(cls, filepath: str | Path, *, strict: bool = False) -> AsyncVCRPlayer:
        """Load a VCR file asynchronously (strict mode mirrors sync player)."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"VCR file not found: {filepath}")

        logger.info("Async loading VCR file: %s", filepath)
        session: Session | None = None
        frames: list[Frame] = []
        skipped = 0

        async with aiofiles.open(filepath) as f:
            line_num = 0
            async for line in f:
                line_num += 1
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
                except (json.JSONDecodeError, Exception) as err:
                    if strict:
                        raise ValueError(
                            f"Malformed record at line {line_num} in {filepath}"
                        ) from err
                    logger.warning("Skipping malformed line %d in %s", line_num, filepath)
                    skipped += 1
                    continue

        if session is None:
            raise ValueError(f"No session header found in {filepath}")

        frames.sort(key=lambda f: f.timestamp)
        logger.info("Async loaded session %s with %d frames", session.session_id, len(frames))

        return cls(session, frames, source_path=filepath, skipped_lines=skipped)

    def goto_frame(self, index: int) -> dict[str, Any]:
        """Jump to a specific frame and return its output state."""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range [0, {len(self.frames)})")

        self._current_index = index
        frame = self.frames[index]
        return StateSerializer.deserialize(frame.output_state)

    def goto_time(self, timestamp: str | datetime) -> dict[str, Any]:
        """Jump to the frame closest to a timestamp."""
        target_time = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

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

        return self.goto_frame(target_idx)

    def get_frame(self, index: int) -> Frame:
        """Get a frame by index."""
        if index < 0 or index >= len(self.frames):
            raise IndexError(f"Frame index {index} out of range")
        return self.frames[index]

    def list_nodes(self) -> list[str]:
        """List unique node names in order of first appearance."""
        seen = set()
        nodes = []
        for frame in self.frames:
            if frame.node_name not in seen:
                seen.add(frame.node_name)
                nodes.append(frame.node_name)
        return nodes

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List explicit checkpoints (async mirror of sync API)."""
        from agent_vcr.models import FrameType

        out: list[dict[str, Any]] = []
        for i, frame in enumerate(self.frames):
            if frame.frame_type != FrameType.CHECKPOINT:
                continue
            label = (frame.metadata.custom or {}).get("checkpoint_label")
            if label is None and frame.node_name.startswith("checkpoint:"):
                label = frame.node_name.split("checkpoint:", 1)[1]
            out.append(
                {
                    "index": i,
                    "label": label,
                    "node_name": frame.node_name,
                    "frame_id": frame.frame_id,
                }
            )
        return out

    def goto_checkpoint(self, label: str) -> dict[str, Any]:
        """Jump to a named checkpoint and return its output state."""
        for cp in self.list_checkpoints():
            if cp["label"] == label:
                return self.goto_frame(cp["index"])
        raise KeyError(f"Checkpoint {label!r} not found")

    def verify_integrity(self) -> dict[str, Any]:
        """Verify session durability invariants (async mirror of sync API)."""
        issues: list[str] = []
        for i, frame in enumerate(self.frames):
            if frame.session_id != self.session.session_id:
                issues.append(f"frame {i} session mismatch: {frame.session_id!r}")
            if i > 0 and frame.timestamp < self.frames[i - 1].timestamp:
                issues.append(f"frame {i} timestamp goes backwards")
        if self.session.frame_count and self.session.frame_count != len(self.frames):
            issues.append(
                f"frame_count {self.session.frame_count} != loaded {len(self.frames)}"
            )
        return {
            "ok": not issues,
            "issues": issues,
            "frame_count": len(self.frames),
            "skipped_lines": self.skipped_lines,
            "session_id": self.session.session_id,
        }

    def get_total_latency(self) -> float:
        """Get total latency across all frames."""
        total = 0.0
        for f in self.frames:
            if f.metadata and f.metadata.latency_ms:
                total += f.metadata.latency_ms
        return total

    def get_total_tokens(self) -> int:
        """Get total tokens across all frames."""
        total = 0
        for f in self.frames:
            if f.metadata and f.metadata.tokens_used:
                total += f.metadata.tokens_used
        return total

    def get_total_cost(self) -> float:
        """Get total cost across all frames."""
        total = 0.0
        for f in self.frames:
            if f.metadata and f.metadata.cost_usd:
                total += f.metadata.cost_usd
        return total
