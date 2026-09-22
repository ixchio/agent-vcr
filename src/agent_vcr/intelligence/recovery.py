"""Recovery engine — classify failures, plan ranked recovery (Phase 6, §13+§29).

Classification is deterministic pattern matching over the recorded
trace (error types, timeout signals, denial frames, verification
frames) — no LLM needed to know *what kind* of failure happened.

Planning ranks actions by blast radius. Only provably side-effect-free
actions are marked ``auto=True`` (retry of idempotent reads, resume of
an interrupted workflow run). Mutating actions (fork/rollback/skip)
are proposed with reasons for a human or policy to approve.

:func:`auto_recover_workflow` executes the safe subset against a real
:class:`WorkflowEngine`: interrupted runs are resumed; nothing else is
touched without approval.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_vcr.intelligence.failures import cluster_key, normalize_message
from agent_vcr.intelligence.models import FailureKind, RecoveryAction, RecoveryPlan
from agent_vcr.models import FrameType
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)

_TRANSIENT = re.compile(r"timeout|timed out|rate.?limit|429|503|502|504|connection|"
                        r"temporar|transient|econnreset|broken pipe|too many requests", re.I)
_MISUSE = re.compile(r"invalid params|validation|unknown tool|unknown agent|missing .*arg|"
                     r"schema|unexpected keyword|takes .* argument", re.I)


def classify_player(player: VCRPlayer) -> tuple[FailureKind, str]:
    """Classify a recorded session's failure (pure, deterministic)."""
    denials = approvals = 0
    last_error = ""
    for frame in player.frames:
        name = frame.node_name
        output = frame.output_state if isinstance(frame.output_state, dict) else {}
        if name.startswith("policy:") and output.get("decision") == "deny":
            denials += 1
        if name.startswith("budget:"):
            denials += 1
        if name.startswith("approval:") and output.get("approved") is False:
            approvals += 1
        if frame.frame_type == FrameType.ERROR:
            last_error = frame.metadata.error_message or last_error
        if name.startswith("verify:") and output.get("passed") is False:
            last_error = f"verification failed: {output.get('reasons')}"
    if approvals:
        return FailureKind.APPROVAL, f"{approvals} approval denial(s)"
    if denials:
        return FailureKind.POLICY, f"{denials} policy/budget denial(s)"
    if last_error:
        if _TRANSIENT.search(last_error):
            return FailureKind.TRANSIENT, last_error
        if _MISUSE.search(last_error):
            return FailureKind.TOOL_MISUSE, last_error
        return FailureKind.OUTPUT, last_error
    if player.get_errors():
        return FailureKind.OUTPUT, "error frames present"
    return FailureKind.UNKNOWN, "no failure signal — session looks healthy"


def plan_recovery(player: VCRPlayer, run_id: str = "",
                  has_checkpoint: bool = False) -> RecoveryPlan:
    """Ranked recovery actions for a session (human approves non-auto)."""
    kind, detail = classify_player(player)
    sid = player.session.session_id
    actions: list[RecoveryAction] = []
    if kind == FailureKind.TRANSIENT:
        actions.append(RecoveryAction(action="retry", target=sid,
                                      reason=f"transient fault ({detail}); retry is safe",
                                      auto=True))
    if run_id:
        actions.append(RecoveryAction(
            action="resume", target=run_id,
            reason="workflow runs persist per-step state; resume continues without re-execution",
            auto=True))
    if has_checkpoint:
        actions.append(RecoveryAction(action="fork", target=sid,
                                      reason="checkpoint exists — fork and retry from safety"))
    if kind in (FailureKind.OUTPUT, FailureKind.TOOL_MISUSE):
        actions.append(RecoveryAction(action="rollback", target=sid,
                                      reason="undo partial mutations, then fix inputs and re-run"))
    actions.append(RecoveryAction(action="escalate", target=sid,
                                  reason=f"needs human judgment ({kind.value}: {detail})"))
    summary = f"{kind.value}: {detail}" if detail else kind.value
    return RecoveryPlan(subject=run_id or sid, failure_kind=kind,
                        actions=actions, summary=summary)


def auto_recover_workflow(run_id: str, engine: Any) -> dict[str, Any]:
    """Execute the safe subset: resume interrupted runs, report the rest.

    Returns ``{run_id, recovered: bool, status, actions_taken}``.
    Never executes mutating actions (fork/rollback/skip) — those stay
    proposed in the plan for approval.
    """
    from agent_vcr.workflows.models import RunStatus

    record = engine.runs.load(run_id)
    if record.status != RunStatus.RUNNING:
        return {"run_id": run_id, "recovered": False, "status": record.status.value,
                "actions_taken": [],
                "note": "run is terminal — human recovery required"}
    resumed = engine.resume(run_id)
    return {"run_id": run_id, "recovered": resumed.status == RunStatus.COMMITTED,
            "status": resumed.status.value, "actions_taken": ["resume"],
            "note": "" if resumed.status == RunStatus.COMMITTED else resumed.error}


def auto_recover_superagent(
    player: VCRPlayer,
    has_checkpoint: bool = False,
) -> RecoveryPlan:
    """Plan recovery for a SuperAgent run (no side-effect execution)."""
    return plan_recovery(player, has_checkpoint=has_checkpoint)


def failure_signature(player: VCRPlayer) -> str:
    """Cluster key of a session's dominant error (links to §27 clusters)."""
    for frame in reversed(player.frames):
        if frame.frame_type == FrameType.ERROR:
            message = frame.metadata.error_message or ""
            return cluster_key(frame.metadata.error_type or "Error",
                               frame.node_name, normalize_message(message))
    return ""
