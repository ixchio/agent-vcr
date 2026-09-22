"""Model routing on observed evidence (Phase 6, §28).

:class:`ModelRegistry` aggregates per-model stats (calls, success rate,
avg cost/latency) from recorded sessions — the same frames CI already
produces. :func:`route` picks deterministically:

1. honor an explicit ``prefer`` list order (first viable wins);
2. else the cheapest model with ``success_rate >= min_success``;
3. ``fallbacks`` chain when the pick later fails (returned for the
   caller to walk — the router never executes).

No live calls, no API keys: routing is pure policy over evidence.
Pair with LiteLLM at the call site (``litellm.completion(model=choice)``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_vcr.intelligence.models import ModelStats, RouteDecision
from agent_vcr.models import FrameType
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Observed model performance, built from session files."""

    def __init__(self) -> None:
        self._stats: dict[str, ModelStats] = {}

    def ingest_player(self, player: VCRPlayer) -> None:
        """Attribute every model-tagged frame (LLM calls and agent steps)."""
        for frame in player.frames:
            model = (frame.metadata.model or "") if frame.metadata else ""
            if not model:
                continue
            stats = self._stats.setdefault(model, ModelStats(model=model))
            stats.calls += 1
            if frame.frame_type != FrameType.ERROR:
                stats.success_calls += 1
            stats.total_tokens += frame.metadata.tokens_used or 0
            stats.total_cost_usd += frame.metadata.cost_usd or 0.0
            stats.total_latency_ms += frame.metadata.latency_ms or 0.0

    def ingest_dir(self, sessions_dir: str | Path, limit: int = 500) -> ModelRegistry:
        paths = sorted(Path(sessions_dir).glob("*.vcr"))[:limit]
        for path in paths:
            try:
                self.ingest_player(VCRPlayer.load(path))
            except (ValueError, OSError) as err:
                logger.debug("Skipping unloadable session %s: %s", path.name, err)
        return self

    def stats(self) -> dict[str, ModelStats]:
        return dict(self._stats)

    def to_dict(self) -> dict[str, Any]:
        return {name: {**s.model_dump(),
                       "success_rate": s.success_rate, "avg_cost": s.avg_cost}
                for name, s in self._stats.items()}


def route(registry: ModelRegistry, *, prefer: list[str] | None = None,
          min_success: float = 0.8, max_avg_cost: float | None = None,
          fallbacks: list[str] | None = None) -> RouteDecision:
    """Pick a model from observed stats (deterministic, auditable)."""
    stats = registry.stats()
    if not stats:
        raise ValueError("No model observations — record sessions first")
    viable = {name: s for name, s in stats.items() if s.success_rate >= min_success}
    if max_avg_cost is not None:
        viable = {n: s for n, s in viable.items() if s.avg_cost <= max_avg_cost}
    pool = viable or dict(stats)  # constraints matched nothing: fall back openly
    fallback_note = "" if viable else " (constraints matched nothing; relaxed)"
    for name in prefer or []:
        if name in pool:
            return RouteDecision(model=name, candidates=sorted(pool),
                                 reason=f"explicit preference{fallback_note}")
    cheapest = min(pool.values(), key=lambda s: (s.avg_cost, -s.success_rate, s.model))
    return RouteDecision(
        model=cheapest.model, candidates=sorted(pool),
        reason=(f"cheapest viable: ${cheapest.avg_cost:.4f}/call at "
                f"{cheapest.success_rate:.0%} success{fallback_note}"),
    )
