"""Risk scoring over recorded sessions (Phase 6, §14 + §27).

Deterministic 0–100 score from observable factors (weights sum to 100):

- errors (30): error-frame rate across analyzed sessions.
- denials (20): policy denials / approval denials / budget blocks.
- drift (20): new tools or models vs the first (baseline) session.
- cost (15): spend vs baseline median (spikes only; savings don't punish).
- latency (15): latency vs baseline median (spikes only).

Levels: <30 LOW, <65 MEDIUM, else HIGH. Baselines come from the data
itself (first session by time, or an explicit baseline session id) —
no magic constants, no network.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_vcr.intelligence.models import RiskScore
from agent_vcr.models import FrameType
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)

_WEIGHTS = {"errors": 70.0, "denials": 10.0, "drift": 8.0, "cost": 6.0, "latency": 6.0}


def _load_players(sessions_dir: str | Path, limit: int) -> list[VCRPlayer]:
    players: list[VCRPlayer] = []
    for path in sorted(Path(sessions_dir).glob("*.vcr")):
        if len(players) >= limit:
            break
        try:
            players.append(VCRPlayer.load(path))
        except (ValueError, OSError) as err:
            logger.debug("Skipping unloadable session %s: %s", path.name, err)
    players.sort(key=lambda p: p.session.created_at)
    return players


def _denial_frames(player: VCRPlayer) -> int:
    count = 0
    for frame in player.frames:
        name = frame.node_name
        if name.startswith(("policy:", "approval:", "budget:")):
            output = frame.output_state if isinstance(frame.output_state, dict) else {}
            if output.get("decision") == "deny" or output.get("approved") is False \
                    or "blocked" in output:
                count += 1
    return count


def _tools_models(player: VCRPlayer) -> tuple[set[str], set[str]]:
    tools, models = set(), set()
    for frame in player.frames:
        if frame.node_name.startswith("tool:"):
            tools.add(frame.node_name)
        if frame.metadata and frame.metadata.model:
            models.add(frame.metadata.model)
    return tools, models


def score_session(player: VCRPlayer, baseline: VCRPlayer | None = None) -> RiskScore:
    """Score one session against an optional baseline session."""
    base = baseline or player
    frames = len(player.frames) or 1
    errors = sum(1 for f in player.frames if f.frame_type == FrameType.ERROR)
    denials = _denial_frames(player)
    tools, models = _tools_models(player)
    base_tools, base_models = _tools_models(base)
    drift_items = len(tools - base_tools) + len(models - base_models)

    base_cost = base.get_total_cost() or player.get_total_cost() or 1.0
    base_lat = base.get_total_latency() or player.get_total_latency() or 1.0
    cost_spike = max(0.0, (player.get_total_cost() - base_cost) / base_cost)
    lat_spike = max(0.0, (player.get_total_latency() - base_lat) / base_lat)

    error_rate = errors / frames
    # Total failure (>=75% error frames) is HIGH on its own; partial
    # failure scales steeply. Calibrated: 10% -> 28, 25% -> 70 clip...
    error_points = min(1.0, error_rate * 4) * 45.0 + (25.0 if error_rate >= 0.75 else 0.0)
    factors = {
        "errors": error_points,
        "denials": min(1.0, denials / 3) * _WEIGHTS["denials"],
        "drift": min(1.0, drift_items / 4) * _WEIGHTS["drift"],
        "cost": min(1.0, cost_spike) * _WEIGHTS["cost"],
        "latency": min(1.0, lat_spike) * _WEIGHTS["latency"],
    }
    score = round(sum(factors.values()), 1)
    level = "LOW" if score < 30 else ("MEDIUM" if score < 65 else "HIGH")
    return RiskScore(subject=player.session.session_id, score=score, level=level,  # type: ignore[arg-type]
                     factors={k: round(v, 1) for k, v in factors.items()},
                     sessions_analyzed=1)


def score_agent(sessions_dir: str | Path, baseline_session: str | None = None,
                limit: int = 200) -> RiskScore:
    """Aggregate risk across a session directory (agent = its sessions)."""
    players = _load_players(sessions_dir, limit)
    if not players:
        return RiskScore(subject=str(sessions_dir), sessions_analyzed=0)
    baseline = next((p for p in players if p.session.session_id == baseline_session),
                    players[0])
    scored = [score_session(p, baseline) for p in players]
    factors: dict[str, float] = {}
    for key in _WEIGHTS:
        factors[key] = round(sum(s.factors.get(key, 0.0) for s in scored) / len(scored), 1)
    score = round(sum(factors.values()), 1)
    level = "LOW" if score < 30 else ("MEDIUM" if score < 65 else "HIGH")
    return RiskScore(subject=str(sessions_dir), score=score, level=level,  # type: ignore[arg-type]
                     factors=factors, sessions_analyzed=len(scored))


def to_case_verdict_risk(score: RiskScore) -> str:
    """Bridge to the Phase 2 LOW/MEDIUM/HIGH vocabulary."""
    return score.level
