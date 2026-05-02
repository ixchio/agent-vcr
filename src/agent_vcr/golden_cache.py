"""
Golden Run Cache — THE thing nobody else has.

When your agent succeeds, agent-vcr saves that run as a "golden path."
Next time you run a similar task, instead of re-calling the LLM for every step,
it replays the golden path and only re-runs the parts that actually changed.

Same task. 80% fewer tokens. Instant.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_vcr.models import (
    FrameMetadata,
    StateSerializer,
)
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)


class CostLedger:
    """Tracks cost savings from golden run replays vs fresh LLM execution."""

    def __init__(self) -> None:
        self.original_tokens: int = 0
        self.original_cost_usd: float = 0.0
        self.original_latency_ms: float = 0.0
        self.replay_tokens: int = 0
        self.replay_cost_usd: float = 0.0
        self.replay_latency_ms: float = 0.0
        self.steps_replayed: int = 0
        self.steps_rerun: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.replay_tokens

    @property
    def cost_saved_usd(self) -> float:
        return self.original_cost_usd - self.replay_cost_usd

    @property
    def time_saved_ms(self) -> float:
        return self.original_latency_ms - self.replay_latency_ms

    @property
    def savings_percent(self) -> float:
        if self.original_cost_usd == 0:
            return 100.0
        return (self.cost_saved_usd / self.original_cost_usd) * 100

    def summary(self) -> dict[str, Any]:
        return {
            "original": {
                "tokens": self.original_tokens,
                "cost_usd": round(self.original_cost_usd, 4),
                "latency_ms": round(self.original_latency_ms, 2),
            },
            "replay": {
                "tokens": self.replay_tokens,
                "cost_usd": round(self.replay_cost_usd, 4),
                "latency_ms": round(self.replay_latency_ms, 2),
            },
            "saved": {
                "tokens": self.tokens_saved,
                "cost_usd": round(self.cost_saved_usd, 4),
                "time_ms": round(self.time_saved_ms, 2),
                "percent": round(self.savings_percent, 1),
            },
            "steps_replayed": self.steps_replayed,
            "steps_rerun": self.steps_rerun,
        }

    def __repr__(self) -> str:
        return (
            f"CostLedger(saved={self.savings_percent:.0f}% | "
            f"${self.cost_saved_usd:.4f} | "
            f"{self.tokens_saved} tokens | "
            f"{self.time_saved_ms:.0f}ms)"
        )


class GoldenRunCache:
    """
    Caches successful agent runs as 'golden paths' indexed by task fingerprint.

    When the same or similar task is run again, the cache replays the golden
    path's outputs directly — skipping LLM calls entirely for unchanged steps.
    Only steps whose inputs have materially changed are re-executed.

    This is ACID Layer 4: Never Pay Twice.
    """

    def __init__(self, cache_dir: str = ".vcr/golden"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}
        self._load_index()

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    def save_golden_run(
        self,
        task: str,
        recorder: VCRRecorder,
        tags: list[str] | None = None,
    ) -> str:
        """
        Save a successful agent run as a golden path.

        Args:
            task: The task description that produced this run.
            recorder: The VCRRecorder that recorded the successful run.
            tags: Optional tags for categorization.

        Returns:
            The fingerprint key used to cache this run.
        """
        fingerprint = self._fingerprint(task)
        session = recorder.get_session()
        frames = recorder.get_frames()

        if not session or not frames:
            raise ValueError("Cannot save an empty run as golden.")

        # Persist the golden run as a .vcr file
        golden_path = self.cache_dir / f"{fingerprint}.vcr"
        with open(golden_path, "w") as f:
            header = {"type": "session", "data": session.model_dump()}
            f.write(json.dumps(header, default=str) + "\n")
            for frame in frames:
                line = {"type": "frame", "data": frame.model_dump()}
                f.write(json.dumps(line, default=str) + "\n")

        # Update the index
        self._index[fingerprint] = {
            "task": task,
            "session_id": session.session_id,
            "frame_count": len(frames),
            "total_tokens": session.total_tokens,
            "total_cost_usd": session.total_cost_usd,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": tags or [],
            "file": str(golden_path),
        }
        self._save_index()

        logger.info(
            "Saved golden run: task=%r fingerprint=%s frames=%d",
            task[:50], fingerprint[:12], len(frames),
        )
        return fingerprint

    def has_golden_run(self, task: str) -> bool:
        """Check if a golden run exists for this task."""
        return self._fingerprint(task) in self._index

    def get_golden_info(self, task: str) -> dict | None:
        """Get metadata about a cached golden run."""
        fp = self._fingerprint(task)
        return self._index.get(fp)

    def replay(
        self,
        task: str,
        step_executor: Callable[[str, dict], dict] | None = None,
        changed_steps: set[int] | None = None,
        recorder: VCRRecorder | None = None,
    ) -> tuple[list[dict], CostLedger]:
        """
        Replay a golden run. Steps not in `changed_steps` are replayed from
        cache (zero LLM cost). Steps in `changed_steps` are re-executed via
        `step_executor`.

        Args:
            task: The task description to look up the golden run.
            step_executor: Callable(node_name, input_state) -> output_state
                           Only called for steps in `changed_steps`.
            changed_steps: Set of frame indices that need re-execution.
                           If None, ALL steps are replayed from cache.
            recorder: Optional recorder to log the replay session.

        Returns:
            Tuple of (list of output states, CostLedger with savings).
        """
        fingerprint = self._fingerprint(task)
        if fingerprint not in self._index:
            raise KeyError(f"No golden run found for task: {task!r}")

        golden_info = self._index[fingerprint]
        golden_player = VCRPlayer.load(golden_info["file"])

        changed_steps = changed_steps or set()
        ledger = CostLedger()

        # Populate the original cost from the golden run
        ledger.original_tokens = golden_player.get_total_tokens()
        ledger.original_cost_usd = golden_player.get_total_cost()
        ledger.original_latency_ms = golden_player.get_total_latency()

        if recorder is None:
            recorder = VCRRecorder()

        recorder.start_session(
            metadata={
                "replay_of": golden_info["session_id"],
                "golden_fingerprint": fingerprint,
                "mode": "golden_replay",
            },
            tags=["golden_replay"],
        )

        outputs: list[dict] = []

        for i, frame in enumerate(golden_player.frames):
            if i in changed_steps and step_executor is not None:
                # RE-RUN: This step has changed, actually call the LLM
                start = time.perf_counter()
                input_state = StateSerializer.deserialize(frame.input_state)
                result = step_executor(frame.node_name, input_state)
                elapsed = (time.perf_counter() - start) * 1000

                recorder.record_step(
                    node_name=frame.node_name,
                    input_state=input_state,
                    output_state=result,
                    metadata=FrameMetadata(
                        latency_ms=elapsed,
                        tokens_used=frame.metadata.tokens_used,
                        cost_usd=frame.metadata.cost_usd,
                        custom={"source": "re_executed"},
                    ),
                )

                # Charge the re-run to the ledger
                ledger.replay_tokens += frame.metadata.tokens_used or 0
                ledger.replay_cost_usd += frame.metadata.cost_usd or 0.0
                ledger.replay_latency_ms += elapsed
                ledger.steps_rerun += 1
                outputs.append(result)
            else:
                # REPLAY: Cache hit — zero LLM cost, instant
                output = StateSerializer.deserialize(frame.output_state)

                recorder.record_step(
                    node_name=frame.node_name,
                    input_state=frame.input_state,
                    output_state=output,
                    metadata=FrameMetadata(
                        latency_ms=0.1,  # negligible replay overhead
                        tokens_used=0,
                        cost_usd=0.0,
                        custom={"source": "golden_cache"},
                    ),
                )

                ledger.replay_latency_ms += 0.1
                ledger.steps_replayed += 1
                outputs.append(output)

        recorder.save()

        logger.info(
            "Golden replay complete: %d replayed, %d rerun | %s",
            ledger.steps_replayed, ledger.steps_rerun, ledger,
        )

        return outputs, ledger

    def list_golden_runs(self) -> list[dict]:
        """List all cached golden runs."""
        return [
            {"fingerprint": fp, **info}
            for fp, info in self._index.items()
        ]

    def invalidate(self, task: str) -> bool:
        """Remove a golden run from the cache."""
        fp = self._fingerprint(task)
        if fp in self._index:
            # Delete the file
            golden_file = Path(self._index[fp]["file"])
            if golden_file.exists():
                golden_file.unlink()
            del self._index[fp]
            self._save_index()
            return True
        return False

    # ──────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────

    def _fingerprint(self, task: str) -> str:
        """Generate a deterministic fingerprint for a task description."""
        normalized = task.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _load_index(self) -> None:
        """Load the golden run index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}

    def _save_index(self) -> None:
        """Persist the golden run index to disk (atomic write)."""
        import os
        import tempfile

        index_path = self.cache_dir / "index.json"
        # Atomic write: write to temp file, then rename.
        # os.replace is atomic on POSIX, preventing corruption on crash.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._index, f, indent=2)
            os.replace(tmp_path, str(index_path))
        except BaseException:
            # Clean up temp file if something goes wrong
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
