# SuperAgent Runtime

SuperAgent turns a goal into a bounded, auditable execution of tools and sub-agents. It is
the control layer over the VCR kernel: it does not replace the recorder, replay system, or
workflow engine.

The runtime executes a validated `Plan` DAG in deterministic dependency order:

```text
validate plan → resolve inputs → policy → budget → approval → execute → verify → checkpoint
```

Every decision and result is written to the VCR session. That makes a run inspectable,
replayable, and usable as evidence in Agent CI.

## When to use it

Use SuperAgent when an agent needs a small, explicit plan with tool boundaries, approval
gates, budget limits, output checks, and a durable trace. Use the
[workflow engine](workflow.md) when the work also needs schedules, persisted run state,
resume after process failure, parallel/conditional control flow, or compensation.

SuperAgent is local and synchronous today. The optional Railway control-plane foundation does
not turn it into a managed multi-tenant service.

## The building blocks

| Component | Responsibility |
| --- | --- |
| `PlanBuilder` | Validates IDs, known refs, bounds, dependency cycles, and topological order before execution. |
| `ToolGateway` | Registers callable tools, validates the supported JSON-schema subset, and enforces each tool timeout. |
| `Policy` | Denies named tools or agents, caps step count, and requires approval for selected or irreversible refs. |
| `BudgetTracker` | Stops a run before its token, cost, latency, or declared step-cost cap is exceeded. |
| `Approver` | Makes an explicit allow/deny decision when a step needs approval. `CallbackApprover` is the normal UI/API bridge. |
| `Verifier` | Checks result envelopes using `equals`, `contains`, `required_keys`, and `forbid_keys`. Error envelopes never pass. |
| `VCRRecorder` | Records the plan, decisions, outputs, verification, checkpoints, and any recovery attempt. |

## First local run

Create a handler module. The CLI reads `TOOLS`, optional `AGENTS`, and optional
`APPROVAL_REQUIRED` from it.

```python
# handlers.py
def lookup_ticket(ticket_id: str) -> dict:
    return {"ticket_id": ticket_id, "status": "open", "priority": "high"}


lookup_ticket.PARAMS = {
    "type": "object",
    "required": ["ticket_id"],
    "properties": {"ticket_id": {"type": "string"}},
}

TOOLS = {"lookup_ticket": lookup_ticket}
AGENTS = {}
```

Then define a plan. A value such as `$lookup.status` resolves from the output of the named
dependency.

```json
{
  "goal": "inspect a support ticket",
  "max_total_cost_usd": 0.25,
  "steps": [
    {
      "id": "lookup",
      "kind": "tool",
      "ref": "lookup_ticket",
      "inputs": {"ticket_id": "t_123"},
      "checkpoint_after": true,
      "check": {"required_keys": ["ticket_id", "status"]}
    }
  ]
}
```

Run it from the repository checkout:

```bash
vcr superagent run --plan plan.json --handlers handlers.py --vcr-dir .vcr
```

The command prints a `RunReport`. Save its `session_id`; it identifies the recording in
`.vcr/`. A committed report has `status: "committed"`. With the default `on_failure: "abort"`,
a blocked, denied, or failed step returns an aborted report and a non-zero CLI exit code.

## Python API

Use the API when the application already owns its tool registry or approval surface.

```python
from agent_vcr import VCRRecorder
from agent_vcr.superagent import (
    BudgetTracker,
    CallbackApprover,
    Policy,
    SuperAgent,
    ToolGateway,
    ToolSpec,
)


def lookup_ticket(ticket_id: str) -> dict:
    return {"ticket_id": ticket_id, "status": "open"}


gateway = ToolGateway()
gateway.register(ToolSpec(
    name="lookup_ticket",
    handler=lookup_ticket,
    parameters={
        "type": "object",
        "required": ["ticket_id"],
        "properties": {"ticket_id": {"type": "string"}},
    },
))

agent = SuperAgent(
    gateway,
    policy=Policy(approval_required=["send_reply"]),
    budget=BudgetTracker(max_total_cost_usd=0.25, max_total_tokens=4_000),
    approver=CallbackApprover(lambda request: False, name="application"),
    recorder=VCRRecorder(output_dir=".vcr"),
)

plan = {
    "goal": "inspect ticket before replying",
    "steps": [
        {
            "id": "lookup",
            "kind": "tool",
            "ref": "lookup_ticket",
            "inputs": {"ticket_id": "t_123"},
            "check": {"contains": {"status": "open"}},
        }
    ],
}

try:
    report = agent.run(plan["goal"], plan, tags=["support"])
finally:
    agent.shutdown()

print(report.model_dump_json(indent=2))
```

The example's approval callback fails closed. Replace it with a function that delegates to
your authenticated UI or service. Mark each consequential tool with `requires_approval=True`
on its `ToolSpec`, require it in `Policy`, or set `requires_approval` on the plan step.

## Plan behavior

Each `PlanStep` has an `id`, a `kind` (`tool` or `agent`), a registered `ref`, and `inputs`.
Steps can reference outputs from a dependency with `$step_id.key`, declare `depends_on`, set a
per-step `timeout_s`, retry with `max_retries`, and add an output `check`.

`on_failure` is `abort` by default. With `skip`, independent work may continue, but steps
depending on the skipped output are blocked. Use checkpoints before important transitions:
`checkpoint_after: true` writes a VCR checkpoint after a verified successful step.

The plan's `max_steps` is validated before execution. Its `max_total_cost_usd` is adopted when
the supplied budget has no cost cap. Set hard token, cost, and latency caps directly on
`BudgetTracker` for runtime enforcement.

## Approval, verification, and recovery

Policies run before the tool or agent handler. A denied step is never invoked. If approval is
required, the runtime records the redacted request and the approver's decision before it
continues or stops.

Verification happens after every handler result. An output can require exact equality, a
subset of fields, required keys, or the absence of forbidden keys. Handler exceptions and
`StepResult(error=...)` fail verification.

For an aborting failure, SuperAgent can ask the intelligence layer for a recovery action. It
records the proposed action and automatically retries only an action marked safe and named
`retry`, up to the configured recovery limit. This is not external-side-effect compensation;
use the [workflow engine](workflow.md) when compensating actions are required.

## Optional planning and integrations

`LLMPlanner` can create a plan JSON proposal through LiteLLM, then passes it through the same
`PlanBuilder` validation and repair loop. It needs the optional LLM dependency and provider
credentials at planning time; hand-authored JSON plans need neither.

`ToolGateway.discover_mcp()` can discover compatible JSON-RPC MCP tools from an endpoint and
register them before a plan runs. Treat every discovered tool as an external capability: apply
schemas, timeouts, approval rules, and budgets before you allow it in a production plan.

## What to inspect after a run

1. Read `RunReport.outcomes` for the outcome, attempts, output, and reasons for every step.
2. Open the VCR session with the normal player or dashboard to inspect plan, policy, approval,
   verification, and checkpoint frames together.
3. Convert a stable session into a replay or Agent CI baseline only after reviewing the side
   effects and expected outputs.

See [Architecture](architecture.md) for how SuperAgent fits into the VCR kernel and the
[workflow guide](workflow.md) for durable multi-run orchestration.
