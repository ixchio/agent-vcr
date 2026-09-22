"""SuperAgent executable plan models (Phase 4, design doc §8).

A plan is an explicit DAG of steps — not prose. The planner produces it,
:class:`PlanBuilder` validates it (cycle detection, unknown refs, bound
checks), and :class:`SuperAgent` executes it on the VCR kernel with every
transition recorded.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

FailureMode = Literal["abort", "retry", "skip"]


class StepKind(str, Enum):
    TOOL = "tool"
    AGENT = "agent"


class PlanStep(BaseModel):
    """One executable node: tool call or sub-agent invocation."""

    id: str
    kind: StepKind = StepKind.TOOL
    ref: str  # tool name in ToolGateway, or sub-agent name
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    checkpoint_after: bool = False
    requires_approval: bool = False
    timeout_s: float = 120.0
    max_cost_usd: float | None = None
    max_retries: int = 0
    on_failure: FailureMode = "abort"
    check: dict[str, Any] = Field(default_factory=dict)  # verifier spec


class Plan(BaseModel):
    """Validated executable DAG with policy bounds."""

    goal: str
    version: int = 1
    steps: list[PlanStep] = Field(default_factory=list)
    max_total_cost_usd: float | None = None
    max_steps: int | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Envelope every tool/sub-agent handler returns (or plain dict)."""

    output: dict[str, Any] = Field(default_factory=dict)
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


class StepOutcome(BaseModel):
    """Recorded outcome of one executed step."""

    step_id: str
    status: Literal["ok", "failed", "skipped", "blocked", "denied"] = "ok"
    attempts: int = 1
    output: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class RunReport(BaseModel):
    """Final report of a SuperAgent run (also persisted in session metadata)."""

    session_id: str
    goal: str
    status: Literal["committed", "aborted"] = "committed"
    outcomes: list[StepOutcome] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


def coerce_result(raw: Any) -> StepResult:
    """Accept StepResult, plain dict output, or (output, usage) tuples."""
    if isinstance(raw, StepResult):
        return raw
    if isinstance(raw, dict) and set(raw) <= {"output", "tokens_used", "cost_usd", "latency_ms", "error"} and "output" in raw:
        return StepResult(**raw)
    if isinstance(raw, dict):
        return StepResult(output=raw)
    return StepResult(output={"result": raw})
