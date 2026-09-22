"""Cost optimization on recorded evidence (Phase 6, §27 + §62).

Two analyses, both offline over ``*.vcr`` files:

- :func:`breakdown` — spend by model and by step (top offenders first).
- :func:`duplicate_calls` — identical ``(tool, canonical-input)`` calls
  repeated across a session set. Every repeat after the first is
  ghost-replayable in principle; the estimated savings assume zero-cost
  cache hits for those repeats (the Phase 1 guarantee).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from agent_vcr.intelligence.models import CostBreakdown
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def breakdown(sessions_dir: str | Path, limit: int = 200, top_steps: int = 10) -> CostBreakdown:
    """Aggregate spend across a session directory."""
    directory = Path(sessions_dir)
    result = CostBreakdown(subject=str(directory))
    steps: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.vcr")):
        if len(steps) > limit * 50:
            break
        try:
            player = VCRPlayer.load(path)
        except (ValueError, OSError) as err:
            logger.debug("Skipping unloadable session %s: %s", path.name, err)
            continue
        result.total_cost_usd += player.get_total_cost()
        result.total_tokens += player.get_total_tokens()
        for i, frame in enumerate(player.frames):
            cost = (frame.metadata.cost_usd or 0.0) if frame.metadata else 0.0
            model = (frame.metadata.model or "untagged") if frame.metadata else "untagged"
            result.by_model[model] = result.by_model.get(model, 0.0) + cost
            steps.append({"session": player.session.session_id, "frame": i,
                          "node": frame.node_name, "model": model,
                          "cost_usd": round(cost, 6)})
            if frame.node_name.startswith("tool:"):
                key = hashlib.sha256(
                    f"{frame.node_name}\x00{_canonical(frame.input_state)}".encode()
                ).hexdigest()[:16]
                entry = seen.setdefault(key, {"node": frame.node_name,
                                              "inputs": frame.input_state,
                                              "count": 0, "cost_usd": 0.0,
                                              "sessions": []})
                entry["count"] += 1
                entry["cost_usd"] = round(entry["cost_usd"] + cost, 6)
                if player.session.session_id not in entry["sessions"]:
                    entry["sessions"].append(player.session.session_id)
    steps.sort(key=lambda s: s["cost_usd"], reverse=True)
    result.by_step = steps[:top_steps]
    result.by_model = {k: round(v, 6) for k, v in
                       sorted(result.by_model.items(), key=lambda kv: kv[1], reverse=True)}
    duplicates = [e for e in seen.values() if e["count"] > 1]
    duplicates.sort(key=lambda e: e["cost_usd"], reverse=True)
    result.duplicate_calls = duplicates
    # repeats after the first are ghost-cacheable at ~zero marginal cost
    result.estimated_ghost_savings_usd = round(sum(
        e["cost_usd"] * (e["count"] - 1) / e["count"] for e in duplicates
        if e["count"] > 0), 6)
    result.total_cost_usd = round(result.total_cost_usd, 6)
    return result
