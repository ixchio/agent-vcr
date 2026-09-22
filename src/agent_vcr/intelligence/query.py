"""'Why did my agent fail?' — unified diagnosis (Phase 6, §49).

:func:`diagnose` fuses the intelligence layer for one session: the
error timeline, the dominant failure's cluster key (linking to fleet
history), a risk score against sibling sessions, and a ranked recovery
plan. One call, one structured answer, zero LLM cost.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_vcr.intelligence.failures import cluster_sessions
from agent_vcr.intelligence.models import Diagnosis
from agent_vcr.intelligence.recovery import classify_player, failure_signature, plan_recovery
from agent_vcr.intelligence.risk import score_agent, score_session
from agent_vcr.models import FrameType
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)


def diagnose(session_path: str | Path, siblings_dir: str | Path | None = None,
             run_id: str = "") -> Diagnosis:
    """Diagnose one recorded session, optionally against sibling sessions."""
    player = VCRPlayer.load(session_path)
    timeline: list[dict[str, Any]] = []
    for i, frame in enumerate(player.frames):
        output = frame.output_state if isinstance(frame.output_state, dict) else {}
        flagged = (
            frame.frame_type == FrameType.ERROR
            or bool(frame.metadata and frame.metadata.error_type)
            or (frame.node_name.startswith("policy:") and output.get("decision") == "deny")
            or (frame.node_name.startswith("approval:") and output.get("approved") is False)
            or (frame.node_name.startswith("verify:") and output.get("passed") is False)
        )
        if flagged:
            timeline.append({
                "frame": i,
                "node": frame.node_name,
                "type": frame.frame_type.value,
                "message": (frame.metadata.error_message or "") if frame.metadata else "",
                "at": frame.timestamp.isoformat(),
            })
    kind, detail = classify_player(player)
    verdict = "ok" if kind.value == "unknown" and not timeline else (
        "failed" if timeline else "degraded")

    siblings = Path(siblings_dir) if siblings_dir else Path(session_path).parent
    baseline = None
    try:
        candidates = sorted(siblings.glob("*.vcr"))
        if candidates:
            baseline = VCRPlayer.load(candidates[0])
    except (ValueError, OSError):
        baseline = None
    risk = score_session(player, baseline)
    try:
        agent_risk = score_agent(siblings)
        risk.sessions_analyzed = max(risk.sessions_analyzed, agent_risk.sessions_analyzed)
    except Exception:  # noqa: BLE001 — sibling scoring is best-effort context
        logger.debug("Sibling risk scoring failed", exc_info=True)

    signature = failure_signature(player)
    cluster_key = ""
    if signature:
        for cluster in cluster_sessions(siblings, limit=500):
            if cluster.key == signature:
                cluster_key = cluster.key
                break
    has_checkpoint = any(n.startswith("checkpoint:") for n in player.list_nodes())
    recovery = plan_recovery(player, run_id=run_id, has_checkpoint=has_checkpoint)

    likely = detail or "no failure signal"
    if cluster_key:
        likely += f" (seen across fleet — cluster {cluster_key[:8]})"
    return Diagnosis(session_id=player.session.session_id, verdict=verdict,  # type: ignore[arg-type]
                     timeline=timeline, likely_cause=likely, cluster_key=cluster_key,
                     risk=risk, recovery=recovery)
