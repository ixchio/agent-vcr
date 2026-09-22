"""Reliability intelligence models (Phase 6, design doc §27–29, §48–49).

The intelligence layer turns accumulated executions (VCR sessions) into
decisions: what fails repeatedly (clusters), how risky an agent is
(scores), how to recover (plans), which model to use (routing), and
where money burns (costs). Every model is JSON-serializable for API/CLI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FailureCluster(BaseModel):
    """One repeated failure shape across sessions."""

    key: str  # error_type + node + normalized template hash
    error_type: str
    node_name: str
    template: str  # message with volatile spans normalized away
    occurrences: int = 0
    sessions: list[str] = Field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    sample_message: str = ""


class RiskScore(BaseModel):
    """0–100 risk for a session/agent (higher = riskier) + factors."""

    subject: str
    score: float = 0.0
    level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    factors: dict[str, float] = Field(default_factory=dict)
    sessions_analyzed: int = 0


class FailureKind(str, Enum):
    TRANSIENT = "transient"  # timeouts, rate limits, connection blips
    TOOL_MISUSE = "tool_misuse"  # validation errors, unknown tools
    POLICY = "policy"  # denials, budget blocks, forbidden tools
    APPROVAL = "approval"  # human denied
    OUTPUT = "output"  # verification/state mismatch, handler errors
    UNKNOWN = "unknown"


class RecoveryAction(BaseModel):
    """One ranked recovery step; safe ones are auto-executable."""

    action: Literal["retry", "resume", "fork", "rollback", "skip", "escalate"] = "retry"
    target: str = ""
    reason: str = ""
    auto: bool = False  # True only when provably side-effect free


class RecoveryPlan(BaseModel):
    """Ranked recovery for a failed run."""

    subject: str
    failure_kind: FailureKind = FailureKind.UNKNOWN
    actions: list[RecoveryAction] = Field(default_factory=list)
    summary: str = ""


class ModelStats(BaseModel):
    """Observed performance of one model across sessions."""

    model: str
    calls: int = 0
    success_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_calls / self.calls if self.calls else 0.0

    @property
    def avg_cost(self) -> float:
        return self.total_cost_usd / self.calls if self.calls else 0.0


class RouteDecision(BaseModel):
    """Model choice with audit trail."""

    model: str
    reason: str = ""
    candidates: list[str] = Field(default_factory=list)


class CostBreakdown(BaseModel):
    """Where the money went + ghost-cacheable waste."""

    subject: str
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    by_model: dict[str, float] = Field(default_factory=dict)
    by_step: list[dict[str, Any]] = Field(default_factory=list)
    duplicate_calls: list[dict[str, Any]] = Field(default_factory=list)
    estimated_ghost_savings_usd: float = 0.0


class Diagnosis(BaseModel):
    """'Why did my agent fail?' answer (§49)."""

    session_id: str
    verdict: Literal["ok", "failed", "degraded"] = "ok"
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    likely_cause: str = ""
    cluster_key: str = ""
    risk: RiskScore | None = None
    recovery: RecoveryPlan | None = None
