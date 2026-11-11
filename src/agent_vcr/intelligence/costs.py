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


def _sub_session_of(frame: Any) -> str:
    """Session id a frame's cost aggregates (nested sub-agent call), if any."""
    custom = frame.metadata.custom if frame.metadata else {}
    sub = custom.get("sub_session_id") if isinstance(custom, dict) else None
    if not sub and isinstance(frame.output_state, dict):
        sub = frame.output_state.get("_sub_session_id")
    return sub if isinstance(sub, str) else ""


def breakdown(sessions_dir: str | Path, limit: int = 200, top_steps: int = 10) -> CostBreakdown:
    """Aggregate spend across a session directory.

    A nested sub-agent's spend is recorded twice: per step in its own
    session, and as one aggregate frame in the parent (tagged with
    ``sub_session_id``). When the sub-session is part of the analyzed set,
    the parent's aggregate frame is counted as $0 (``rolled_up_into``) so
    spend is not double-counted.
    """
    directory = Path(sessions_dir)
    result = CostBreakdown(subject=str(directory))
    players: list[VCRPlayer] = []
    for path in sorted(directory.glob("*.vcr")):
        try:
            players.append(VCRPlayer.load(path))
        except (ValueError, OSError) as err:
            logger.debug("Skipping unloadable session %s: %s", path.name, err)
    session_ids = {p.session.session_id for p in players}
    steps: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for player in players:
        if len(steps) > limit * 50:
            break
        sid = player.session.session_id
        session_cost = player.get_total_cost()
        session_tokens = player.get_total_tokens()
        for i, frame in enumerate(player.frames):
            cost = (frame.metadata.cost_usd or 0.0) if frame.metadata else 0.0
            model = (frame.metadata.model or "untagged") if frame.metadata else "untagged"
            step: dict[str, Any] = {"session": sid, "frame": i,
                                    "node": frame.node_name, "model": model}
            sub = _sub_session_of(frame)
            if sub and sub != sid and sub in session_ids:
                session_cost -= cost
                session_tokens -= (frame.metadata.tokens_used or 0) if frame.metadata else 0
                cost = 0.0
                step["rolled_up_into"] = sub
            step["cost_usd"] = round(cost, 6)
            result.by_model[model] = result.by_model.get(model, 0.0) + cost
            steps.append(step)
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
                if sid not in entry["sessions"]:
                    entry["sessions"].append(sid)
        result.total_cost_usd += max(0.0, session_cost)
        result.total_tokens += max(0, session_tokens)
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
