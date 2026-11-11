# SuperAgent: a plan runner with guardrails

Give it a small plan of tool calls. It runs them in order, with a policy, a budget, approvals and
an output check on each step, and records every decision as it goes.

> 🧪 **Experimental.** Works and is tested, but the API may change between minor releases.

**TL;DR**

- You write a plan (JSON or a dict): steps that call tools or sub-agents, wired together with
  `$step.key` references.
- Before each step runs: is it forbidden? over budget? does it need a human? After it runs: did
  the output pass its `check`?
- Every decision (plan, policy, approval, output, verification, checkpoint) is a frame in a `.vcr`
  recording, so you can inspect it, replay it, or turn it into an [Agent CI](../agent-ci.md)
  baseline.

## Try it in 60 seconds

No API key needed. Create two files.

`handlers.py`: the CLI reads `TOOLS`, plus `AGENTS` and `APPROVAL_REQUIRED` if they exist.

```python
def lookup_ticket(ticket_id: str) -> dict:
    return {"ticket_id": ticket_id, "status": "open", "priority": "high"}


def send_reply(ticket_id: str, text: str) -> dict:
    return {"sent": True, "ticket_id": ticket_id}


lookup_ticket.PARAMS = {
    "type": "object",
    "required": ["ticket_id"],
    "properties": {"ticket_id": {"type": "string"}},
}

TOOLS = {"lookup_ticket": lookup_ticket, "send_reply": send_reply}
APPROVAL_REQUIRED = ["send_reply"]  # the CLI asks you y/N before this runs
```

`plan.json`:

```json
{
  "goal": "answer a support ticket",
  "steps": [
    {
      "id": "lookup",
      "ref": "lookup_ticket",
      "inputs": {"ticket_id": "t_123"},
      "check": {"required_keys": ["ticket_id", "status"]},
      "checkpoint_after": true
    },
    {
      "id": "reply",
      "ref": "send_reply",
      "inputs": {"ticket_id": "$lookup.ticket_id", "text": "On it!"}
    }
  ]
}
```

Run it:

```bash
vcr superagent run --plan plan.json --handlers handlers.py
# Approve tool:send_reply (step reply, tool 'send_reply' requires approval)? [y/N] y
# { "session_id": "20261111_170408_24459be3", "status": "committed", "outcomes": [...] }
```

Answer `n` and the report says `"status": "aborted"`, the `reply` step is `"denied"`, and the
command exits with code 1. The recording is in `.vcr/<session_id>.vcr`.

## How it works

```text
 plan ──► validate (ids, refs, cycles) ──► for each step, in dependency order:

   resolve $refs ─► policy ─► budget ─► approval ─► run (timeout) ─► verify ─► checkpoint
                      │         │          │                           │
                    deny      over cap   denied                      fails
                      └─────────┴──────────┴──────────► step fails ◄───┘
                                                          │
                                  on_failure: "abort" (stop) or "skip" (go on)
```

- **The plan is checked before anything runs.** Unknown tools, duplicate ids and dependency
  cycles raise `ValueError` up front. Using `$lookup.x` in a step's inputs adds the dependency
  on `lookup` for you.
- **A denied step is never called.** `Policy` can forbid tools or agents, cap the number of steps,
  and require approval for named refs, step ids or "irreversible" refs.
- **Budgets stop the next step.** `BudgetTracker` checks before each step whether the run has
  already reached its cost, token or latency cap. One expensive step can still go past the cap.
  A step's `max_cost_usd` is the only pre-step guard against that.
- **Verification runs on every result.** `check` supports `equals`, `contains`, `required_keys`
  and `forbid_keys`. A handler exception or an error result always fails.
- **Secrets are redacted** before inputs and outputs reach the recording: keys like `api_key` and
  `password`, plus token-shaped strings (`sk-…`, `ghp_…`, JWTs, PEM keys).
- **Recovery is conservative.** On an aborting failure it asks the
  [intelligence layer](intelligence.md) for a recovery plan. The only thing it does on its own is
  retry a transient failure of a tool registered `idempotent=True`, at most
  `max_recovery_attempts` times (default 2).

## Everyday recipes

### Use it from Python

```python
from agent_vcr import VCRRecorder
from agent_vcr.superagent import (
    AutoApprover, BudgetTracker, Policy, SuperAgent, ToolGateway,
)

gateway = ToolGateway()


@gateway.tool("lookup_ticket", parameters={
    "type": "object", "required": ["ticket_id"],
    "properties": {"ticket_id": {"type": "string"}},
})
def lookup_ticket(ticket_id: str) -> dict:
    return {"ticket_id": ticket_id, "status": "open"}


@gateway.tool("send_reply", requires_approval=True)
def send_reply(ticket_id: str, text: str) -> dict:
    return {"sent": True}


@gateway.tool("delete_ticket")
def delete_ticket(ticket_id: str) -> dict:
    return {"deleted": True}


agent = SuperAgent(
    gateway,
    policy=Policy(forbid_tools=["delete_ticket"], max_steps=10),
    budget=BudgetTracker(max_total_cost_usd=0.25, max_total_tokens=4_000),
    approver=AutoApprover(allow_refs=["send_reply"]),  # approves only this ref
    recorder=VCRRecorder(output_dir=".vcr"),
)

plan = {
    "goal": "reply to a ticket",
    "steps": [
        {"id": "lookup", "ref": "lookup_ticket", "inputs": {"ticket_id": "$inputs.ticket"},
         "check": {"contains": {"status": "open"}}},
        {"id": "reply", "ref": "send_reply",
         "inputs": {"ticket_id": "$lookup.ticket_id", "text": "Fixed in 1.6.0"}},
    ],
}

try:
    report = agent.run(plan["goal"], plan, inputs={"ticket": "t_123"})
finally:
    agent.shutdown()

print(report.status, [(o.step_id, o.status) for o in report.outcomes])
# committed [('lookup', 'ok'), ('reply', 'ok')]
```

A plan step that calls `delete_ticket` comes back `blocked` with the reason
`tool 'delete_ticket' is forbidden by policy`, and the handler never runs.

### Pick an approver

| Approver | What it does |
|---|---|
| `AutoApprover()` | **Denies everything.** The default, so an approval step never runs unattended by accident. |
| `AutoApprover(allow_refs=[...])` | Approves only the refs you list. `deny_refs` always wins. |
| `AutoApprover(allow_all=True, max_auto_cost_usd=0.05)` | Approves a step only when its `max_cost_usd` is set and within the ceiling. Without a cost estimate it denies. |
| `CallbackApprover(fn, name="ui")` | Calls `fn(request) -> bool`: your Slack button, web UI or prompt. If `fn` raises, the answer is no. |

The CLI uses a `CallbackApprover` that asks on stdin. With no terminal attached (CI, a pipe with
no answer) it denies.

### Retries, timeouts and skipping

Each step takes `timeout_s` (default 120), `max_retries` and `on_failure` (`"abort"` or
`"skip"`). A skipped step's output is `{}`. Steps that depend on it are `blocked`, and a blocked
step aborts the run unless it's marked `"skip"` too. Independent steps carry on.
A tool call that times out keeps running on its thread, because Python can't kill a thread. So a
timed-out step is **only retried when its tool is registered `idempotent=True`**.

### Nest agents

`other_agent.as_handler(sub_plan)` turns a whole SuperAgent run into a handler. Put it in
`agents={...}` and call it with `"kind": "agent"`. Its cost is rolled up so
[`vcr intel costs`](intelligence.md) doesn't count it twice.

### Let a model write the plan (optional)

`LLMPlanner(model="...")` asks a model through LiteLLM (`pip install "ai-agent-vcr[llm]"`, plus
that provider's API key), then runs the proposal through the same validation. If validation
fails, the errors go back to the model for another try (`max_repair_attempts`, default 1).
Plans you write by hand need neither LiteLLM nor a key.

### Pull tools from an MCP server

`gateway.discover_mcp("http://host/mcp")` calls `tools/list` over JSON-RPC and registers every
tool it finds. Treat those as outside code: forbid, require approval for, or time-limit them
before they reach a real plan.

## What it can't do

- **No undo for side effects.** A failed run stops, but an email already sent stays sent. For
  compensating actions, use the [workflow engine](workflow.md).
- **No persistence or resume.** A SuperAgent run lives in one process. If it crashes, the
  recording survives but the run doesn't continue. The [workflow engine](workflow.md) persists
  state after every step.
- **Synchronous and local.** One plan at a time per agent, in your process. It's not a hosted
  service.
- **Budgets are checked between steps**, not during one. A single step can overshoot the cap.
- **Timed-out calls can't be cancelled.** They finish (or hang) on a daemon thread in the
  background.
- **Redaction is pattern-based.** It catches common key names and token formats, not every
  secret.

## Cheat sheet

| API / command | What it does |
|---|---|
| `vcr superagent run --plan P --handlers H` | Run a plan file. Also `--forbid-tools a,b`, `--approval-required a,b`, `--max-steps N`, `--max-cost USD`, `--goal`, `--vcr-dir`. Exit 1 unless committed. |
| `SuperAgent(gateway, agents=, policy=, budget=, approver=, recorder=)` | Build the runner. |
| `agent.run(goal, plan, inputs=, tags=, session_id=)` | Run and return a `RunReport` (`status`, `outcomes`, totals). |
| `agent.shutdown()` | Stop the tool gateway's workers. |
| `ToolGateway().tool(name, parameters=, requires_approval=, timeout_s=, idempotent=)` | Register a tool (decorator). Or `register(ToolSpec(...))`. |
| `Policy(forbid_tools=, forbid_agents=, approval_required=, irreversible_refs=, max_steps=)` | Rules checked before every step. |
| `BudgetTracker(max_total_cost_usd=, max_total_tokens=, max_total_latency_ms=)` | Per-run caps. |
| `AutoApprover(...)` / `CallbackApprover(fn)` | Approval decisions. |
| Step fields | `id`, `kind` (`tool`/`agent`), `ref`, `inputs`, `depends_on`, `check`, `timeout_s`, `max_retries`, `max_cost_usd`, `on_failure`, `requires_approval`, `checkpoint_after` |
| References | `$step.key.path`, `$inputs.key`, `$$x` for a literal `$x` |

**Next:** [Workflow engine](workflow.md) · [Reliability intelligence](intelligence.md) ·
[Recording and replay](../recording-and-replay.md)
