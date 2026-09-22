"""Workflow definition models (Phase 5, design doc §9).

A workflow is a versioned, durable DAG of steps. Step kinds:

- ``tool`` / ``agent`` — leaf execution (Phase 4 gateway/handlers).
- ``checkpoint`` — named restore point in the VCR trace.
- ``condition`` — if/then/else branch on prior outputs.
- ``parallel`` — named branches executed concurrently, joined on completion.
- ``for_each`` — loop a body over a list with an iteration cap.
- ``approval`` — explicit human gate (also available per leaf step).

Leaf steps may declare ``compensate`` (saga action, run in reverse
completion order on abort) and ``on_failure`` (abort/retry/skip).
Every definition carries a monotonic ``version``; runs pin the exact
definition snapshot they executed.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

FailureMode = Literal["abort", "retry", "skip"]


class StepKind(str, Enum):
    TOOL = "tool"
    AGENT = "agent"
    CHECKPOINT = "checkpoint"
    CONDITION = "condition"
    PARALLEL = "parallel"
    FOR_EACH = "for_each"
    APPROVAL = "approval"


class CompensateSpec(BaseModel):
    """Saga compensation for one leaf step."""

    ref: str
    kind: Literal["tool", "agent"] = "tool"
    inputs: dict[str, Any] = Field(default_factory=dict)


class ConditionSpec(BaseModel):
    """Predicate over resolved outputs: <ref> <op> <value>."""

    ref: str  # "$step.path" reference
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "truthy", "falsy"] = "=="
    value: Any = None


class WorkflowStep(BaseModel):
    """One node in a workflow definition."""

    id: str
    kind: StepKind = StepKind.TOOL
    ref: str = ""  # tool/agent name; checkpoint label; approval note
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    timeout_s: float = 120.0
    max_retries: int = 0
    retry_delay_s: float = 0.0
    on_failure: FailureMode = "abort"
    check: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    checkpoint_after: bool = False
    # condition kind
    condition: ConditionSpec | None = None
    then_steps: list[WorkflowStep] = Field(default_factory=list)
    else_steps: list[WorkflowStep] = Field(default_factory=list)
    # parallel kind: named branches of linear step lists
    branches: dict[str, list[WorkflowStep]] = Field(default_factory=dict)
    # for_each kind
    for_each: str = ""  # "$step.path" resolving to a list
    item_var: str = "item"
    body: list[WorkflowStep] = Field(default_factory=list)
    max_iterations: int = 100
    # saga compensation for leaf steps
    compensate: CompensateSpec | None = None


class TriggerSpec(BaseModel):
    """How a workflow run starts: manual call, webhook, or schedule name."""

    type: Literal["manual", "webhook", "schedule"] = "manual"
    schedule: str = ""
    secret: str = ""  # webhook shared secret (empty = no check)


class WorkflowDef(BaseModel):
    """Versioned workflow definition (the reproducible unit)."""

    name: str
    version: int = 1
    description: str = ""
    triggers: list[TriggerSpec] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    max_total_cost_usd: float | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMMITTED = "committed"
    ABORTED = "aborted"


class CompletedCompensation(BaseModel):
    """A compensation owed (pushed on leaf success, popped on abort)."""

    step_path: str
    ref: str
    kind: str = "tool"
    inputs: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """Durable run state — the crash-recovery contract (one JSON file)."""

    run_id: str
    workflow: str
    workflow_version: int
    definition: WorkflowDef
    status: RunStatus = RunStatus.RUNNING
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    completed_top_steps: list[str] = Field(default_factory=list)
    compensations: list[CompletedCompensation] = Field(default_factory=list)
    compensated: bool = False
    error: str = ""
    session_id: str = ""
    callback_url: str = ""
    created_at: str = ""
    updated_at: str = ""
