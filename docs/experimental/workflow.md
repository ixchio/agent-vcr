# Workflows: agent pipelines that survive crashes and clean up after themselves

A durable DAG of tool and agent steps. State is saved after every step, so a run that dies picks
up where it stopped. If it fails halfway, it undoes its own side effects in reverse order.

> 🧪 **Experimental.** Works and is tested, but the API may change between minor releases.

**TL;DR**

- Describe a workflow in JSON or Python: steps, dependencies, retries, approvals, loops,
  branches, parallel fan-out.
- Every run is a JSON file you can resume after a crash, plus a `.vcr` recording of each decision.
- A step can declare a `compensate` action ("cancel the booking"). If the run aborts, the
  compensations run newest first, like a video game reloading your last save.

## Try it in 60 seconds

From a clone of the repo, with no API key:

```bash
vcr workflow validate examples/hello_workflow.json
# Valid: hello-agent-vcr (2 steps, order: greet, uppercase)

vcr workflow run --file examples/hello_workflow.json --handlers examples/workflow_handlers.py
# Run c9f8863adb37: committed (2 steps, session 20261111_165916_5f2540c5)

vcr workflow show c9f8863adb37      # use your run id: full run state, outputs included
```

The workflow calls `greet`, then feeds `$greet.message` into `uppercase`:

```json
{
  "id": "uppercase",
  "kind": "tool",
  "ref": "uppercase",
  "inputs": {"value": "$greet.message"},
  "depends_on": ["greet"],
  "checkpoint_after": true,
  "check": {"required_keys": ["message"]}
}
```

The handlers file is plain Python. The CLI registers every function in its `TOOLS` dict (and
agents in `AGENTS`, if there are any):

```python
def greet(name: str = "world") -> dict[str, str]:
    return {"message": f"Hello, {name}!"}


def uppercase(value: str = "") -> dict[str, str]:
    return {"message": value.upper()}


TOOLS = {"greet": greet, "uppercase": uppercase}
```

`vcr workflow show` prints `"outputs": {"greet": {"message": "Hello, Agent VCR!"}, "uppercase":
{"message": "HELLO, AGENT VCR!"}}`.

## How it works

```text
 start ─► validate DAG ─► save run file ─► for each top-level step (dependency order):
                                              │
            policy ─► budget ─► approval ─► call (timeout, retries) ─► verify
                                              │
                    ok ──► save outputs + owed compensation ──► next step
                    fail ─► skip, or ABORT ──► run compensations newest-first
```

- **Saved after every top-level step.** A run is one file, `.vcr/workflow_runs/<run_id>.json`,
  with the exact definition it ran, the outputs so far and the compensations owed. Writes are
  atomic (temp file, then rename).
- **Resume skips finished work.** `resume` reloads that file and continues from the first
  unfinished top-level step. Loops, branches and parallel blocks count as one step, so a crash in
  the middle of a loop re-runs the whole loop.
- **Definitions are versioned.** `vcr workflow run --file` stores the definition in
  `.vcr/workflows/<name>.v<N>.json`. The version only goes up when the content changes.
- **Same guardrails as [SuperAgent](superagent.md).** Policy, budgets, approvals and output
  checks run on every leaf step. Every decision lands in the run's `.vcr` recording, with secrets
  redacted.
- **Each run gets its own budget.** It starts from the engine's caps, and the workflow's
  `max_total_cost_usd` can only make it stricter.

## Step kinds

| `kind` | What it does | Key fields |
|---|---|---|
| `tool` (default) | Calls a registered tool | `ref`, `inputs`, `check`, `compensate` |
| `agent` | Calls a handler from `agents={...}` with one inputs dict | same as `tool` |
| `checkpoint` | Records a named restore point in the recording | `ref` = label |
| `approval` | A human gate. Output `{"approved": true}` | `ref` = the question |
| `condition` | if / else on an earlier output | `condition`, `then_steps`, `else_steps` |
| `parallel` | Named branches run at the same time, then join | `branches` |
| `for_each` | Runs `body` once per item in a list | `for_each`, `item_var`, `body`, `max_iterations` |

Any step can also set `depends_on`, `timeout_s` (default 120), `max_retries`, `retry_delay_s`,
`on_failure`, `idempotent`, `requires_approval` and `checkpoint_after`.

## Everyday recipes

### Pass inputs and wire outputs

A string starting with `$` is a reference:

| You write | You get |
|---|---|
| `"$inputs.who"` | the run's inputs (`--inputs '{"who": "Ada"}'`) |
| `"$greet.message"` | step `greet`'s output, key `message` (dots go deeper, numbers index lists) |
| `"$doc"` | the loop variable inside a `for_each` body |
| `"$$greet.message"` | the literal text `$greet.message` |
| `"$HOME"`, `"$10"` | left alone: not a known step, input or loop variable |

```json
{
  "name": "hello-inputs",
  "steps": [
    {"id": "greet", "ref": "greet", "inputs": {"name": "$inputs.who"}},
    {"id": "shout", "ref": "uppercase", "inputs": {"value": "$greet.message"},
     "depends_on": ["greet"]},
    {"id": "literal", "ref": "uppercase", "inputs": {"value": "$$greet.message"}}
  ]
}
```

```bash
vcr workflow run --file inputs_workflow.json --handlers examples/workflow_handlers.py \
    --inputs '{"who": "Ada"}'
# outputs: greet → "Hello, Ada!", shout → "HELLO, ADA!", literal → "$GREET.MESSAGE"
```

If a reference points at a step that ran but doesn't have that key (for example, a skipped
step), the run aborts with `unresolvable input`.

### Retry flaky steps, cap slow ones

```python
import time

from agent_vcr.superagent import ToolGateway
from agent_vcr.workflows import WorkflowDef, WorkflowEngine, WorkflowStep

gateway = ToolGateway()
calls = {"flaky": 0}


@gateway.tool("flaky_fetch")
def flaky_fetch(url: str) -> dict:
    calls["flaky"] += 1
    if calls["flaky"] < 3:
        raise ConnectionError("connection reset")
    return {"status": 200}


@gateway.tool("slow_lookup", idempotent=True)  # safe to run twice
def slow_lookup() -> dict:
    time.sleep(0.5)
    return {"ok": True}


wf = WorkflowDef(
    name="resilient",
    steps=[
        WorkflowStep(id="fetch", ref="flaky_fetch", inputs={"url": "https://example.com"},
                     max_retries=3, retry_delay_s=0.1),  # waits 0.1 s, 0.2 s, 0.4 s
        WorkflowStep(id="lookup", ref="slow_lookup", timeout_s=0.2, max_retries=1,
                     on_failure="skip"),
    ],
)

engine = WorkflowEngine(gateway)
record = engine.start(wf)
engine.shutdown()
print(record.status.value, "after", calls["flaky"], "fetch attempts")
# committed after 3 fetch attempts
print(record.outputs["lookup"])
# {'skipped': True, 'reason': "ToolTimeoutError: Tool 'slow_lookup' exceeded 0.2s (...)"}
```

- `retry_delay_s` doubles after each retry. A cancel request cuts the wait short.
- `on_failure`: `"abort"` (default) stops the run and compensates. `"skip"` records
  `{"skipped": true, "reason": ...}` and carries on. `"retry"` is `"abort"` with at least one
  retry.
- **Timeouts and `idempotent`.** Python can't stop a thread, so a timed-out call may still be
  running and doing its side effect. A timed-out step is only retried when the step or its tool
  is marked `idempotent=True`. Otherwise it fails straight away, and the recording notes the
  attempt may still be running.

### Undo side effects with compensation

```python
from agent_vcr.superagent import ToolGateway
from agent_vcr.workflows import CompensateSpec, WorkflowDef, WorkflowEngine, WorkflowStep

gateway = ToolGateway()


@gateway.tool("reserve_seat")
def reserve_seat(flight: str) -> dict:
    print(f"reserved a seat on {flight}")
    return {"booking_id": "bk_42"}


@gateway.tool("cancel_seat")
def cancel_seat(booking_id: str) -> dict:
    print(f"cancelled {booking_id}")
    return {"cancelled": booking_id}


@gateway.tool("charge_card")
def charge_card(booking_id: str) -> dict:
    raise RuntimeError("payment provider is down")


trip = WorkflowDef(
    name="book-trip",
    steps=[
        WorkflowStep(
            id="reserve", ref="reserve_seat", inputs={"flight": "VCR-88"},
            compensate=CompensateSpec(ref="cancel_seat",
                                      inputs={"booking_id": "$reserve.booking_id"}),
        ),
        WorkflowStep(id="charge", ref="charge_card", depends_on=["reserve"],
                     inputs={"booking_id": "$reserve.booking_id"}),
    ],
)

engine = WorkflowEngine(gateway)
record = engine.start(trip)
engine.shutdown()
print(record.status.value, record.compensated)
print(record.error)
# reserved a seat on VCR-88
# cancelled bk_42
# aborted True
# Step charge failed: RuntimeError: payment provider is down
```

- Only `tool` and `agent` steps can have `compensate`. The compensation's inputs are worked out
  when the step **succeeds**, then stored in the run file.
- On abort, compensations run newest first. The step that failed has nothing to undo.
- Compensations go through the same policy, approval and timeout checks
  (`CompensateSpec.timeout_s`, defaulting to the step's own). The step-count cap doesn't apply,
  so the cap that caused the abort can't block the undo.
- If a compensation fails or is denied, the rest still run, `record.compensated` is `False`,
  and `record.error` says which one. That's your cue to check by hand.

### Ask a human first (approvals fail closed)

```python
from agent_vcr.superagent import AutoApprover, CallbackApprover, ToolGateway
from agent_vcr.workflows import StepKind, WorkflowDef, WorkflowEngine, WorkflowStep

gateway = ToolGateway()


@gateway.tool("deploy")
def deploy(env: str) -> dict:
    return {"deployed": env}


release = WorkflowDef(
    name="release",
    steps=[
        WorkflowStep(id="sign_off", kind=StepKind.APPROVAL, ref="ship to prod?"),
        WorkflowStep(id="ship", ref="deploy", inputs={"env": "prod"},
                     depends_on=["sign_off"], requires_approval=True),
    ],
)

for approver in [
    AutoApprover(),                                        # denies everything
    AutoApprover(allow_refs=["ship to prod?", "deploy"]),  # approves only these refs
    CallbackApprover(lambda req: req.ref != "deploy", name="ops"),
]:
    engine = WorkflowEngine(gateway, approver=approver)
    record = engine.start(release)
    print(type(approver).__name__, record.status.value, "|", record.error or "-")
gateway.shutdown()
# AutoApprover aborted | Step sign_off denied approval: no auto-approval policy configured ...
# AutoApprover committed | -
# CallbackApprover aborted | Step ship denied approval: human denied
```

- **`AutoApprover()` denies everything.** That's on purpose (since 1.6.0): a gated step never
  runs unattended by accident. List the refs you trust in `allow_refs`. For an `approval` step,
  the ref is its `ref` text.
- `AutoApprover(allow_all=True)` approves nothing in workflows. Workflow approval requests carry
  no cost estimate, and without one only allow-listed refs pass.
- A denial aborts the run (and compensates), or skips the step if it has `on_failure="skip"`.
- **Who decides where.** `vcr workflow run` asks you `[y/N]` on stdin, and no answer means no.
  `vcr workflow resume`, `vcr workflow schedule tick` and `vcr intel recover --execute` use
  `AutoApprover()`, so they deny every approval. Over HTTP, `vcr-server` also denies every
  approval. To get approvals in those places, run the engine from Python with your own approver.

### Loops, parallel branches and conditions

```python
from agent_vcr.superagent import ToolGateway
from agent_vcr.workflows import (
    ConditionSpec, StepKind, WorkflowDef, WorkflowEngine, WorkflowStep,
)

gateway = ToolGateway()
gateway.tool("list_docs")(lambda: {"ids": ["a", "b", "c"]})
gateway.tool("summarize")(lambda doc: {"summary": f"summary of {doc}"})
gateway.tool("word_count")(lambda doc: {"words": 120})
gateway.tool("score")(lambda: {"value": 91})
gateway.tool("publish")(lambda: {"published": True})
gateway.tool("hold")(lambda: {"held": True})

wf = WorkflowDef(
    name="flow-control",
    steps=[
        WorkflowStep(id="docs", ref="list_docs"),
        WorkflowStep(
            id="each_doc", kind=StepKind.FOR_EACH, depends_on=["docs"],
            for_each="$docs.ids", item_var="doc", max_iterations=50,
            body=[WorkflowStep(id="sum", ref="summarize", inputs={"doc": "$doc"})],
        ),
        WorkflowStep(
            id="fan_out", kind=StepKind.PARALLEL,
            branches={
                "left": [WorkflowStep(id="wc", ref="word_count", inputs={"doc": "a"})],
                "right": [WorkflowStep(id="q", ref="score")],
            },
        ),
        WorkflowStep(
            id="gate", kind=StepKind.CONDITION, depends_on=["fan_out"],
            condition=ConditionSpec(ref="$fan_out.right.q.value", op=">=", value=80),
            then_steps=[WorkflowStep(id="go", ref="publish")],
            else_steps=[WorkflowStep(id="wait", ref="hold")],
        ),
    ],
)

engine = WorkflowEngine(gateway)
record = engine.start(wf)
engine.shutdown()
print(record.status.value)            # committed
print(record.outputs["each_doc"])     # {'items': [{'summary': 'summary of a'}, ...], 'count': 3}
print(record.outputs["fan_out"])      # {'left.wc': {'words': 120}, 'right.q': {'value': 91}}
print(record.outputs["gate.go"])      # {'published': True}
```

- **`for_each`** needs a list. More items than `max_iterations` (default 100) **aborts** the run;
  the list is never cut short without telling you. The output is each iteration's last body step
  plus a `count`.
- **`parallel`** runs each branch on its own thread. Read a result as
  `$fan_out.<branch>.<step>.<key>`. If any branch fails, the others finish first, then the run
  aborts.
- **`condition`** operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `truthy`,
  `falsy`. A reference that can't be resolved counts as `None`. Results of the branch that ran
  are stored as `gate.go`, so `$gate.go.published` works.

### Survive a crash, resume

A handlers file that kills its own process on step 2, the first time only:

```python
import os


def step(n: int) -> dict:
    if n == 2 and os.environ.get("CRASH"):
        os._exit(1)  # simulate the machine dying mid-run
    print(f"ran step {n}", flush=True)
    return {"n": n}


TOOLS = {"step": step}
```

```json
{
  "name": "three-steps",
  "steps": [
    {"id": "one", "ref": "step", "inputs": {"n": 1}},
    {"id": "two", "ref": "step", "inputs": {"n": 2}, "depends_on": ["one"]},
    {"id": "three", "ref": "step", "inputs": {"n": 3}, "depends_on": ["two"]}
  ]
}
```

```bash
CRASH=1 vcr workflow run --file three_steps.json --handlers crashy_handlers.py
# ran step 1          ...and the process dies
vcr workflow runs     # status: "running"; copy the run_id
vcr workflow resume 134f06aabc83 --handlers crashy_handlers.py
# ran step 2
# ran step 3
# Run 134f06aabc83: committed
```

Step 1 didn't run again. Resuming a finished run does nothing and just prints its status.
`vcr intel recover <session.vcr> --run-id <id> --execute` does the same resume for you
([intelligence](intelligence.md)).

If the crash happens while compensating, `resume` doesn't re-run the failed step. It goes
straight back to compensating.

### Cancel a run

There's no CLI command for this. From Python, in any process:

```python
from agent_vcr.workflows import RunStore

RunStore(".vcr/workflow_runs").request_cancel("0a18d94052f9", "wrong dataset")
```

Cancelling is cooperative. The step that's running finishes, then the engine sees the
`<run_id>.cancel` marker before the next step, compensates and marks the run `aborted` with
`cancelled: true`. In the same process, `engine.cancel(run_id, reason)` does the same. A run
nobody is driving is compensated and aborted right away.

### Run on a schedule

```bash
vcr workflow schedule put --name nightly --workflow hello-agent-vcr --kind cron --cron "0 2 * * *"
vcr workflow schedule put --name every-5-min --workflow hello-agent-vcr --kind interval --interval-s 300
vcr workflow schedule list
vcr workflow schedule tick --handlers examples/workflow_handlers.py   # runs whatever is due, once
vcr workflow schedule remove --name every-5-min
```

**`tick` runs what's due once and exits. There's no daemon.** Call it every minute from cron,
systemd or your CI scheduler:

```text
* * * * * cd /path/to/project && vcr workflow schedule tick --handlers handlers.py
```

How cron schedules behave:

- **Five fields, in UTC, matched to the minute:** `*`, `*/15`, `5/10`, `1,15`, `9-17`, `1-5/2`.
  `0` and `7` both mean Sunday. A bad expression is rejected by `put`
  (`Cron field '61' out of range 0-59`).
- **Day-of-month *or* day-of-week**, like classic cron: `0 9 1 * 1` fires on the 1st *and* on
  every Monday.
- **Missed slots fire once, not as a burst.** If `tick` didn't run for three hours, an hourly
  schedule fires once on the next tick.
- **At most once per slot.** A schedule is marked as run *before* the workflow starts, and the
  schedules file is locked, so two ticking processes never both run it. If the run crashes,
  that slot is not retried.
- `--kind interval` fires when `--interval-s` seconds have passed since the last run (straight
  away the first time). `--kind at --at 2026-12-01T00:00:00Z` fires once, then never again.

In Python: `Scheduler(".vcr/schedules").run_pending(engine, WorkflowRegistry().get)`, plus
`cron_matches(expr, when)` and `next_due_in(spec, now)` for planning.

### Pick a backend

| Backend | What you get |
|---|---|
| `LocalWorkflowBackend` (default) | Runs in your process through `WorkflowEngine`. Everything on this page. |
| `InngestWorkflowBackend` | A thin adapter (`pip install "ai-agent-vcr[inngest]"`). `dispatch()` creates the run file and sends an `agent_vcr/<name>.triggered` event. `memoize_step()` / `get_memoized_step()` keep a step's output so a retried worker doesn't call the model again. **It doesn't run your steps:** you write the Inngest function. Its `cancel_run()` marks the run aborted **without** compensating. |

`create_workflow_backend("auto")` picks Inngest only when the SDK is installed *and* an
`inngest_client` or `INNGEST_EVENT_KEY` is present. Otherwise you get the local backend.

Webhook-style starts: `engine.trigger(workflow, inputs, secret=...)` checks the secret of any
`webhook` trigger in the definition (constant-time compare) and raises `PermissionError` on a
mismatch. `vcr-server` exposes workflows over HTTP; see the [dashboard guide](../dashboard.md).

## What it can't do

- **It can't take back what the world already saw.** Compensation runs *your* undo action. If
  there isn't one (an email, a payment you can't refund), nothing is undone. Compensation can
  also fail, and then `compensated` is `False`.
- **Durability is per top-level step.** A crash inside a loop, branch or parallel block re-runs
  that whole block on resume. Make those steps idempotent.
- **Timed-out calls keep running.** Threads can't be killed, so a timeout doesn't stop the side
  effect. Mark only truly idempotent steps `idempotent=True`.
- **No built-in scheduler process.** Something has to call `schedule tick`. Cron times are UTC
  only.
- **No CLI cancel, and approvals outside `vcr workflow run` are always denied** unless you
  drive the engine from Python.
- **The Inngest backend is an adapter, not an executor.**
- **Local files, one machine.** Runs live in `.vcr/workflow_runs/`. Two engines resuming the same
  run in different processes aren't protected from each other. Inside one engine, a second
  drive of a live run raises `RuntimeError`.

## Cheat sheet

| Command / API | What it does |
|---|---|
| `vcr workflow validate FILE` | Check the DAG and print the execution order. |
| `vcr workflow run --file FILE \| --workflow NAME --handlers H [--inputs JSON]` | Run it (`--file` also registers it). Also `--forbid-tools`, `--max-cost`. Exit 1 unless committed. |
| `vcr workflow list` / `runs [--workflow N]` / `show RUN_ID` | Registered workflows, runs, one run's full state. |
| `vcr workflow resume RUN_ID --handlers H` | Continue an interrupted run. |
| `vcr workflow schedule put\|list\|remove\|tick` | Manage and fire schedules. |
| Shared flags | `--workflows-dir`, `--runs-dir`, `--schedules-dir`, `--vcr-dir` (all under `.vcr/` by default) |
| `WorkflowEngine(gateway, agents=, policy=, budget=, approver=, run_store=, vcr_dir=)` | The engine. |
| `engine.start(wf, inputs)` / `resume(id)` / `step_once(id)` / `cancel(id, reason)` / `trigger(wf, inputs, secret)` | Run lifecycle. |
| `WorkflowRegistry().save(wf)` / `.get(name, version)` | Versioned definitions. |
| `RunStore().load(id)` / `.list()` / `.request_cancel(id, reason)` | Run files. |
| `RunRecord` | `status` (`running`/`committed`/`aborted`), `outputs`, `completed_top_steps`, `compensated`, `cancelled`, `error`, `usage`, `session_id` |

**Next:** [SuperAgent](superagent.md) · [Reliability intelligence](intelligence.md) ·
[Recording and replay](../recording-and-replay.md)
