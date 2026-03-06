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

    def __init__(self, session: Session, frames: list[Frame]):
        self.session = session
        self.frames = frames
        self._current_index = 0

    @classmethod
    async def load(cls, filepath: str | Path) -> AsyncVCRPlayer:
        """Load a VCR file asynchronously."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"VCR file not found: {filepath}")

        logger.info("Async loading VCR file: %s", filepath)
        session: Session | None = None
        frames: list[Frame] = []

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
                except (json.JSONDecodeError, Exception):
                    logger.warning("Skipping malformed line %d in %s", line_num, filepath)
                    continue

        if session is None:
            raise ValueError(f"No session header found in {filepath}")

        frames.sort(key=lambda f: f.timestamp)
        logger.info("Async loaded session %s with %d frames", session.session_id, len(frames))

        return cls(session, frames)

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
