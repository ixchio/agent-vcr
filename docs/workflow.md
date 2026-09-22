# Durable Workflow Engine: Architecture & Specification

> **Package:** `agent_vcr.workflows`
> **Status:** Production / Stable
> **Runtime Kernel:** VCR Execution Kernel (`execution/v1`)
> **Distributed Backends:** Pluggable (`LocalWorkflowBackend`, `InngestWorkflowBackend`)

---

## 1. Executive Summary & Problem Statement

Autonomous AI agent workflows are inherently non-deterministic, long-running, and prone to volatile external failures (model rate limits, context window overflow, hallucinated tool arguments, third-party API outages). Traditional orchestration frameworks treat agent runs as ephemeral scripts or naive DAG pipelines:

1. **Transient State Loss:** If an agent process crashes or is killed during step 7 of an 8-step pipeline, intermediate results are discarded, forcing an expensive, non-deterministic re-run from step 1.
2. **Side-Effect Pollution:** When an agent partially succeeds (e.g., provisions cloud infrastructure, charges a credit card, writes files to disk) and then encounters a fatal failure, orphan side effects remain active with no structured rollback mechanism.
3. **Double-Spend on Distributed Retries:** Re-executing non-deterministic LLM calls on worker failover causes state drift, token waste, and hallucination divergence.
4. **Fragile Scheduling:** Distributed schedulers often require heavy dependencies (Redis, Celery, Temporal server fleets) just to execute periodic agent tasks.

The **Agent VCR Workflow Engine** (`agent_vcr.workflows`) resolves these problems by coupling **step-level durability** directly to the VCR execution kernel. Every workflow run executes against a pinned, content-versioned definition snapshot. Execution state is persisted atomically after every top-level step via `RunStore`. If a process crashes, `WorkflowEngine.resume()` reloads the persisted session and state outputs, continuing from the exact point of interruption without re-executing completed work.

If execution aborts, a built-in **Saga Reverse-Compensation Engine** unwinds side effects in strict Last-In, First-Out (LIFO) completion order.

---

## 2. Core Execution Topology

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Workflow Trigger                                        │
│             (Manual SDK Call / Webhook HMAC Secret / Cron & Interval Scheduler)         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   WorkflowEngine                                        │
│  1. DAG Topological Validation (Cycle / Dangling Ref Checks)                             │
│  2. Pin Workflow Snapshot (Content Hash Versioning via WorkflowRegistry)                 │
│  3. Initialize VCR Session (Append-only JSONL execution/v1 log)                          │
│  4. Instantiate RunRecord in RunStore (<run_id>.json)                                    │
└───────┬───────────────────────────────────┬─────────────────────────────────────────────┘
        │                                   │
        ▼                                   ▼
┌───────────────────────────────┐   ┌─────────────────────────────────────────────────────┐
│    Sequential / Join Steps    │   │          Parallel & Looping Flow Control            │
│  - ToolGateway leaf execution │   │  - Parallel Branches (ThreadPoolExecutor Join)      │
│  - Sub-agent delegation       │   │  - For-Each Loops (Bounded max_iterations)          │
│  - Verifier check assertions  │   │  - Condition Branches (Predicate evaluation)        │
│  - Named VCR Checkpoints      │   │  - Human-in-the-Loop Approvals (Approver Protocol)  │
└───────┬───────────────────────┘   └───────────────────────┬─────────────────────────────┘
        │                                                   │
        └───────────────────────────┬───────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
           [Step Succeeded]                    [Step Aborted]
                  │                                   │
                  ▼                                   ▼
┌───────────────────────────────────┐ ┌───────────────────────────────────────────────────┐
│     Atomic Persistence Gate       │ │          Saga Reverse-Compensation Stack          │
│ 1. Record Frame in VCR Session    │ │ 1. Pop completed compensations in LIFO order      │
│ 2. Append step to completed_steps │ │ 2. Dispatch undo actions (ToolGateway or Agent)   │
│ 3. Push CompensateSpec to stack   │ │ 3. Record compensation & failure frames in VCR    │
│ 4. Atomic RunStore fsync rewrite  │ │ 4. Persist run record with status = ABORTED       │
└───────────────────────────────────┘ └───────────────────────────────────────────────────┘
```

---

## 3. Workflow Definition & DAG Compilation

Workflows are declared as versioned, declarative data structures (`WorkflowDef`). Leaf execution steps wire inputs dynamically using JSON-pointer / dot-notation references (`$step_id.key`).

### 3.1 Data Model Specification

```python
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
    ref: str                           # Tool name or agent handler identifier
    kind: Literal["tool", "agent"] = "tool"
    inputs: dict[str, Any] = Field(default_factory=dict)

class ConditionSpec(BaseModel):
    ref: str                           # "$step.path" reference
    op: Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "truthy", "falsy"] = "=="
    value: Any = None

class WorkflowStep(BaseModel):
    id: str
    kind: StepKind = StepKind.TOOL
    ref: str = ""                      # Tool/agent name, checkpoint label, or note
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    timeout_s: float = 120.0
    max_retries: int = 0
    retry_delay_s: float = 0.0
    on_failure: FailureMode = "abort"
    check: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    checkpoint_after: bool = False

    # Conditional branching
    condition: ConditionSpec | None = None
    then_steps: list[WorkflowStep] = Field(default_factory=list)
    else_steps: list[WorkflowStep] = Field(default_factory=list)

    # Parallel branches
    branches: dict[str, list[WorkflowStep]] = Field(default_factory=dict)

    # Iteration (loops)
    for_each: str = ""                 # Reference resolving to an iterable list
    item_var: str = "item"
    body: list[WorkflowStep] = Field(default_factory=list)
    max_iterations: int = 100

    # Saga compensation
    compensate: CompensateSpec | None = None

class WorkflowDef(BaseModel):
    name: str
    version: int = 1
    description: str = ""
    triggers: list[TriggerSpec] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    max_total_cost_usd: float | None = None
    labels: dict[str, str] = Field(default_factory=dict)
```

### 3.2 Compilation & Topological Validation

Before any execution occurs, `WorkflowEngine._check_dag()` compiles the step graph into a strictly ordered execution plan:

1. **Cycle Detection:** A depth-first traversal tracks current visit stacks. If node $A$ depends on node $B$ and node $B$ transitively depends on node $A$, a `ValueError("Dependency cycle: A -> B -> A")` is raised.
2. **Dangling Dependency Verification:** Every identifier in `step.depends_on` must exist within the top-level definition.
3. **Duplicate ID Prevention:** Duplicate step identifiers at the top level are rejected at compile time.
4. **Deterministic Topological Sorting:** Dependencies are visited in sorted lexicographical order to ensure deterministic dispatch across platforms.

---

## 4. Step-Level Durability & Crash Recovery

### 4.1 Persistence Model: `RunStore`

Durability is backed by atomic filesystem transactions. Each active workflow run is represented by a single, comprehensive `RunRecord`:

```python
class RunRecord(BaseModel):
    run_id: str
    workflow: str
    workflow_version: int
    definition: WorkflowDef             # Pinned snapshot of executed definition
    status: RunStatus = RunStatus.RUNNING
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    completed_top_steps: list[str] = Field(default_factory=list)
    compensations: list[CompletedCompensation] = Field(default_factory=list)
    compensated: bool = False
    error: str = ""
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""
```

#### The Atomic Rewrite Protocol
To prevent corrupt files caused by sudden power loss, kernel panics, or SIGKILL events during writes, `RunStore` uses atomic file replacement:

```python
def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(path))      # POSIX atomic rename
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
```

### 4.2 Crash Recovery via `resume()`

When a crashed process restarts, recovery requires zero re-execution of previously completed steps:

```python
# 1. Initialize engine on existing persistence directories
engine = WorkflowEngine(gateway, run_store=RunStore(".vcr/workflow_runs"), vcr_dir=".vcr")

# 2. Resume execution from run_id
recovered_record = engine.resume(run_id)
assert recovered_record.status == RunStatus.COMMITTED
```

#### Under the Hood:
1. `engine.runs.load(run_id)` reads `<run_id>.json` and validates schema integrity.
2. `recorder.attach_session(record.session_id)` binds the VCR JSONL recorder to the active session file, dynamically recalculating frame offsets and repairing session header totals.
3. `_drive()` iterates through the pre-computed topological step order. Any step whose `id` already exists in `record.completed_top_steps` is bypassed entirely, while its cached outputs in `record.outputs` remain available for downstream `$step_id.key` references.
4. Execution seamlessly continues from the first unexecuted step.

---

## 5. Saga Reverse-Compensation Engine

Autonomous agents interact with external real-world systems. If step 4 fails after steps 1, 2, and 3 succeeded, rolling back the filesystem via git is insufficient — external side effects must be undone.

### 5.1 The Compensation Contract

Any leaf step (`tool` or `agent`) can declare a `CompensateSpec`:

```yaml
id: provision_vm
kind: tool
ref: cloud_compute_create_vm
inputs:
  instance_type: "t4g.xlarge"
  region: "us-east-1"
compensate:
  ref: cloud_compute_terminate_vm
  inputs:
    instance_id: "$provision_vm.instance_id"
```

### 5.2 LIFO Execution Semantics

1. When `provision_vm` completes successfully, `WorkflowEngine` resolves the compensation inputs against the live execution context and pushes a `CompletedCompensation` record onto `record.compensations`.
2. If any subsequent step encounters an unhandled exception or verification failure with `on_failure="abort"`:
   - The engine catches `_AbortRun`.
   - It marks the run error and flushes state.
   - It enters `_compensate(record, recorder)`.
3. The engine pops compensation operations in **strict Last-In, First-Out (LIFO)** order.
4. Each compensation action is dispatched through the `ToolGateway` or agent handler, and its execution is recorded in the VCR session as a `compensate:<ref>` frame.

```
Execution Order:       Step 1 (DB Write) ──► Step 2 (S3 Upload) ──► Step 3 (API Charge - FAILS)
                                                                           │
                                                                    Trigger Abort
                                                                           │
Compensation Order:    Undo Step 2 (S3 Delete) ◄── Undo Step 1 (DB Delete) ◄┘
```

### 5.3 Resilience During Compensation Failures

What happens if a compensation step itself fails?
- The workflow engine catches the exception, logs a warning, appends the error to `record.error`, records a `compensate:<ref>` error frame in VCR, and **continues executing the remaining compensations on the stack**.
- `record.compensated` is set to `False` to flag that manual operator intervention is required.
- If a crash occurs *during* compensation, calling `engine.resume(run_id)` detects `record.error` and immediately resumes the remaining compensations without re-running the failed leaf step.

---

## 6. Flow Control Primitives

### 6.1 Conditional Branching (`condition`)
Supports branching on resolved upstream data:

```python
WorkflowStep(
    id="evaluate_risk",
    kind=StepKind.CONDITION,
    condition=ConditionSpec(ref="$fraud_check.score", op=">=", value=80),
    then_steps=[WorkflowStep(id="block_card", ref="freeze_account")],
    else_steps=[WorkflowStep(id="send_notification", ref="email_user")],
    depends_on=["fraud_check"],
)
```

Operators supported: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `truthy`, `falsy`.

### 6.2 Parallel Fan-Out / Join (`parallel`)
Executes named branch lists concurrently using a bounded `ThreadPoolExecutor`:

```python
WorkflowStep(
    id="gather_intelligence",
    kind=StepKind.PARALLEL,
    branches={
        "osint": [WorkflowStep(id="whois", ref="lookup_domain")],
        "internal": [WorkflowStep(id="db_query", ref="lookup_crm")],
    },
)
```

- Each branch executes in parallel.
- Results are joined and namespaced into the run context:
  `$gather_intelligence.osint.whois.registrar`
  `$gather_intelligence.internal.db_query.customer_id`
- If any parallel branch fails, sibling branches are collected, and an aggregated `_AbortRun` triggers saga compensation.

### 6.3 Iterative Loops (`for_each`)
Executes a sequence of steps over a resolved array, protected against run-away agent loops:

```python
WorkflowStep(
    id="process_documents",
    kind=StepKind.FOR_EACH,
    for_each="$query_docs.document_ids",
    item_var="doc_id",
    max_iterations=50,  # Strict safeguard against unbounded loops
    body=[
        WorkflowStep(id="ocr", ref="ocr_extract", inputs={"id": "$doc_id"}),
        WorkflowStep(id="embed", ref="vectorize", inputs={"text": "$ocr.text"}),
    ],
)
```

Outputs are collected and returned as an array:
`record.outputs["process_documents"] = {"items": [...], "count": N}`

### 6.4 Human-in-the-Loop Gate (`approval`)
Pauses or gates execution based on the `Approver` protocol:

```python
class Approver(abc.ABC):
    @abc.abstractmethod
    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """Evaluate approval request. Fail-closed: approved=False on error."""
```

- `AutoApprover`: Approves automatically (useful in CI / headless testing).
- `CallbackApprover`: Calls custom webhooks, CLI prompts, or Slack notifications.
- When an approval request is rejected:
  - If `step.on_failure == "skip"`: The step is skipped, outputs set to `{}`.
  - If `step.on_failure == "abort"`: `_AbortRun` is raised and compensation initiates.

---

## 7. Zero-Dependency Deterministic Scheduling

The `Scheduler` (`agent_vcr.workflows.schedules`) provides cron, interval, and one-shot execution with zero external daemons.

### 7.1 Parser Implementation & Cron Capabilities

The scheduler features a custom, zero-dependency 5-field cron parser:

$$\text{Minute (0–59)} \quad \text{Hour (0–23)} \quad \text{Day-of-Month (1–31)} \quad \text{Month (1–12)} \quad \text{Day-of-Week (0–7)}$$

Features:
- Sub-intervals: `*/15 * * * *` (every 15 minutes)
- Discrete lists: `10,20,40 * * * *`
- Explicit ranges: `9-17 * * * 1-5` (business hours, Monday–Friday)
- Normalization: `0` and `7` both map to Sunday
- Timezone awareness: Coerces naive timestamps to UTC to prevent arithmetic crashes

### 7.2 Schedule Kinds

| Kind | Parameter | Behavior |
|---|---|---|
| `interval` | `interval_s=3600.0` | Fires every $N$ seconds elapsed from `last_run_at`. |
| `cron` | `cron="0 2 * * *"` | Fires once per matching UTC minute. Deduplicated at minute resolution. |
| `at` | `at="2026-10-01T00:00:00Z"` | Fires exactly once when UTC time passes threshold; then marked spent. |

### 7.3 Planning vs Execution Separation

```python
scheduler = Scheduler(directory=".vcr/schedules")

# 1. tick() is a pure, side-effect-free planner (auditable)
due_schedules = scheduler.tick(now=datetime.now(timezone.utc))

# 2. run_pending() executes due workflows via the WorkflowEngine
records = scheduler.run_pending(engine=workflow_engine)
```

---

## 8. Pluggable Backends & Inngest Distributed Memoization

To support multi-node worker clusters, Agent VCR abstracts workflow dispatch behind the `WorkflowBackend` interface:

```python
class WorkflowBackend(abc.ABC):
    @abc.abstractmethod
    def dispatch(self, workflow: WorkflowDef, inputs: dict[str, Any] | None = None) -> RunRecord | str: ...
    @abc.abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None: ...
    @abc.abstractmethod
    def resume_run(self, run_id: str) -> RunRecord | None: ...
    @abc.abstractmethod
    def cancel_run(self, run_id: str) -> bool: ...
```

### 8.1 Inngest Distributed Execution (`InngestWorkflowBackend`)

When running in distributed environments with the Inngest SDK (`inngest >= 0.5.0`):

1. **Event Dispatch:** Workflows trigger via distributed events:
   `agent_vcr/<workflow_name>.triggered` with payload containing `run_id`, `version`, and `inputs`.
2. **Step Memoization for Non-Deterministic LLMs:**
   In distributed workflows, worker nodes can crash mid-execution. Re-running an LLM step produces a different completion, which can cause downstream drift. `InngestWorkflowBackend.memoize_step(run_id, step_id, output)` records the step result. Upon distributed worker retry, `get_memoized_step(run_id, step_id)` returns the exact previous output, preserving determinism and eliminating redundant LLM API costs.

---

## 9. Concrete Implementation Example

The following example demonstrates a multi-step customer refund workflow with validation, human approval, database mutation, payment processing, and automatic saga compensation:

```python
from agent_vcr.workflows import (
    WorkflowDef,
    WorkflowStep,
    WorkflowEngine,
    StepKind,
    CompensateSpec,
    ConditionSpec,
)
from agent_vcr.superagent import ToolGateway, CallbackApprover

# 1. Initialize Tool Gateway
gateway = ToolGateway()

@gateway.tool("validate_order", parameters={"order_id": {"type": "string"}})
def validate_order(order_id: str) -> dict:
    return {"order_id": order_id, "amount": 250.00, "status": "eligible"}

@gateway.tool("mark_db_refunded", parameters={"order_id": {"type": "string"}})
def mark_db_refunded(order_id: str) -> dict:
    # Mutates internal database
    return {"order_id": order_id, "db_status": "refund_pending"}

@gateway.tool("unmark_db_refunded", parameters={"order_id": {"type": "string"}})
def unmark_db_refunded(order_id: str) -> dict:
    # Saga compensation to revert DB mutation
    return {"order_id": order_id, "db_status": "active"}

@gateway.tool("stripe_refund", parameters={"amount": {"type": "number"}, "order_id": {"type": "string"}})
def stripe_refund(amount: float, order_id: str) -> dict:
    # Simulate network failure to demonstrate compensation
    raise RuntimeError("Stripe API Connection Timeout")

# 2. Define Workflow with Saga Compensation & Approval
refund_workflow = WorkflowDef(
    name="customer_refund_pipeline",
    version=1,
    steps=[
        WorkflowStep(
            id="validate",
            ref="validate_order",
            inputs={"order_id": "ord_98765"},
        ),
        WorkflowStep(
            id="manager_approval",
            kind=StepKind.APPROVAL,
            ref="Refund amount exceeds $100 threshold",
            depends_on=["validate"],
        ),
        WorkflowStep(
            id="update_database",
            ref="mark_db_refunded",
            inputs={"order_id": "$validate.order_id"},
            depends_on=["manager_approval"],
            compensate=CompensateSpec(
                ref="unmark_db_refunded",
                inputs={"order_id": "$validate.order_id"},
            ),
        ),
        WorkflowStep(
            id="execute_stripe",
            ref="stripe_refund",
            inputs={
                "amount": "$validate.amount",
                "order_id": "$validate.order_id",
            },
            depends_on=["update_database"],
            max_retries=2,
            retry_delay_s=0.5,
            on_failure="abort",
        ),
    ],
)

# 3. Execute with Engine
engine = WorkflowEngine(
    gateway=gateway,
    approver=CallbackApprover(lambda req: True, name="finance_lead"),
)

record = engine.start(refund_workflow)

# When stripe_refund fails after updating the database:
assert record.status.value == "aborted"
assert record.compensated is True  # unmark_db_refunded was automatically executed!
```
