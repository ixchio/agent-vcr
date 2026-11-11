# Reliability intelligence: ask your recordings why things broke

`vcr intel` reads the `.vcr` files your agents already produce and tells you what failed, how
often, how risky an agent looks, where the money went, which model to use and how to recover.

> 🧪 **Experimental.** Works and is tested, but the API may change between minor releases.

**TL;DR**

- Six commands: `why`, `failures`, `risk`, `costs`, `routes` and `recover`. All output JSON.
- Offline and deterministic: pattern matching and arithmetic over your recordings. No model
  calls, no API key, no network.
- It **suggests**. The only thing it ever does by itself is resume an interrupted
  [workflow](workflow.md) run, and only when you pass `--execute`.

## Try it in 60 seconds

Make four fake runs: two models, one flaky booking API.

```python
# make_runs.py
from agent_vcr import VCRRecorder

rec = VCRRecorder(output_dir=".vcr")
for run in range(4):
    rec.start_session(session_id=f"run-{run}")
    model = "gpt-4o-mini" if run % 2 else "claude-haiku"
    rec.record_llm_call(model=model, messages=[{"role": "user", "content": "plan"}],
                        response={"content": "ok"}, tokens_input=200, tokens_output=50,
                        latency_ms=400, cost_usd=0.002)
    rec.record_step("tool:search", {"q": "flights to Lisbon"}, {"hits": 3})
    if model == "gpt-4o-mini":  # every gpt-4o-mini run hits the same flaky API
        rec.record_error("tool:book", {"flight": f"TP{run}00"},
                         ConnectionError(f"connection reset after {run * 1000}ms"),
                         latency_ms=30_000, model=model)
    rec.save()
```

```bash
python make_runs.py
vcr intel why .vcr/run-3.vcr
```

```json
{
  "session_id": "run-3",
  "verdict": "failed",
  "timeline": [{"frame": 2, "node": "tool:book", "type": "error",
                "message": "connection reset after 3000ms", "at": "..."}],
  "likely_cause": "connection reset after 3000ms (seen across fleet — cluster dc7dd3af)",
  "risk": {"score": 55.0, "level": "MEDIUM", "...": "..."},
  "recovery": {"failure_kind": "transient", "actions": [
    {"action": "retry", "auto": false,
     "reason": "transient fault (...); step idempotency is unknown — a retry may repeat side effects, ..."},
    {"action": "escalate", "auto": false, "...": "..."}]}
}
```

One command gives you the error timeline, the fact that this failure has happened before, a risk
score, and a recovery plan that refuses to retry blindly. `vcr intel why` exits with code 1 when
the verdict is `failed`, so you can use it in scripts.

## How it works

```text
 .vcr files ──► error frames, denials, failed checks, models, costs, latency
                      │
   why ── one session: timeline + cluster + risk + recovery plan
   failures ── same error shape across sessions (ids/numbers/paths normalized away)
   risk ── 0-100 score vs a baseline session
   costs ── spend by model and step, repeated tool calls
   routes ── cheapest model that meets a success bar
   recover ── ranked actions; --execute resumes interrupted workflow runs only
```

- **Clusters ignore the noise.** `connection reset after 2000ms` and `... 3000ms` become the
  template `connection reset after <n>`. UUIDs, hex, timestamps, paths and numbers are
  normalized, then grouped by error type and node.
- **Failure kinds** come from the recording: `approval` (a human said no), `policy` (denied or
  over budget), `transient` (timeouts, rate limits, 429/502/503/504, connection resets),
  `tool_misuse` (validation errors, unknown tools, bad arguments), `output` (failed checks and
  other errors) and `unknown`.
- **Risk** adds up five weighted factors: errors (up to 70), denials (10), drift, meaning new
  tools or models compared with the baseline (8), cost spikes (6) and latency spikes (6).
  Under 30 is `LOW`, under 65 is `MEDIUM`, anything higher is `HIGH`. The baseline is the oldest
  session in the folder, or the one you name with `--baseline`.
- **Recovery never guesses about side effects.** A retry is marked `auto: true` only when the
  failed step is known to be idempotent. `resume` (for a workflow run id) is the only other auto
  action. `fork` and `rollback` are suggestions for you to act on.

## Everyday recipes

### What keeps breaking?

```bash
vcr intel failures              # clusters across .vcr/*.vcr, most frequent first
vcr intel failures --dir runs/  # another folder (non-recursive)
# [{"error_type": "ConnectionError", "node_name": "tool:book",
#   "template": "connection reset after <n>", "occurrences": 2, "sessions": ["run-1", "run-3"], ...}]
```

### How risky is this agent?

```bash
vcr intel risk                       # average over every session in .vcr/
vcr intel risk --baseline run-0      # compare against a known-good session
# {"score": 27.5, "level": "LOW", "factors": {"errors": 22.5, "drift": 2.0, "latency": 3.0, ...}}
```

### Where did the money go?

```bash
vcr intel costs
# total_cost_usd, by_model, the 10 most expensive steps, and duplicate_calls:
# identical tool calls (same name, same inputs) repeated across runs, with
# estimated_ghost_savings_usd if those repeats were served by Ghost Replay
```

Spend from a nested sub-agent is counted once, not twice, when its own session is in the same
folder. Turning those savings into real ones is [Ghost Replay](../recording-and-replay.md).

### Which model should I use?

```bash
vcr intel routes                            # cheapest model with >= 80% success
vcr intel routes --min-success 0.95 --max-cost 0.01
vcr intel routes --prefer claude-haiku,gpt-4o-mini
# {"model": "claude-haiku", "reason": "cheapest viable: $0.0020/successful call at 100% success",
#  "constraints_met": true, "stats": {...}}
```

"Cheapest" means spend per *successful* call, so a model that always fails never looks free.
If nothing meets your bar you still get an answer, with `"constraints_met": false` and a reason.
It never quietly relaxes the bar. Only frames tagged with a model count (LLM calls, and errors
recorded with `model=`).

### Recover a run

```bash
vcr intel recover .vcr/run-3.vcr                    # print the ranked plan
vcr intel recover .vcr/<session>.vcr --run-id <id>  # a workflow run: adds "resume" (auto)
vcr intel recover .vcr/<session>.vcr --run-id <id> --execute --handlers handlers.py
# {"run_id": "91f4e9375248", "recovered": true, "status": "committed", "actions_taken": ["resume"]}
```

`--execute` only resumes a workflow run that is still `running` (it crashed or was interrupted).
A finished or aborted run is left alone with `"note": "run is terminal — human recovery
required"`. It uses `AutoApprover()`, so any approval step on the way is denied.

### From Python

```python
from agent_vcr.intelligence import breakdown, cluster_sessions, diagnose, score_agent

print(diagnose(".vcr/run-3.vcr").likely_cause)
print([c.template for c in cluster_sessions(".vcr")])
print(score_agent(".vcr").level, breakdown(".vcr").total_cost_usd)
```

Also available: `plan_recovery(player, run_id=, has_checkpoint=, idempotent=)`,
`auto_recover_workflow(run_id, engine)`, and `ModelRegistry().ingest_dir(".vcr")` with
`route(registry, prefer=, min_success=, max_avg_cost=)`.

## What it can't do

- **It only knows what was recorded.** No cost metadata means $0. No `model=` means routing
  can't see it. Errors swallowed by your agent are invisible.
- **Heuristics, not understanding.** Classification is regex over error text and frame names.
  A strange error message lands in `output` or `unknown`.
- **Scores are relative.** Risk compares a folder with its own baseline session, not with the
  rest of the world.
- **It reads one flat folder** of `*.vcr` files (subfolders are skipped). By default it reads at
  most 500 sessions for `failures` and `routes`, and 200 for `risk` and `costs`.
- **It doesn't fix anything except resuming workflow runs.** Retry, fork and rollback are yours
  to approve.

## Cheat sheet

| Command | What it does |
|---|---|
| `vcr intel why SESSION.vcr [--siblings-dir D] [--run-id ID]` | Diagnose one session against its neighbours. Exit 1 on `failed`. |
| `vcr intel failures [--dir .vcr] [--limit 500]` | Cluster repeated errors. |
| `vcr intel risk [--dir .vcr] [--baseline SESSION_ID]` | 0–100 risk score with factors. |
| `vcr intel costs [--dir .vcr]` | Spend by model and step, duplicate tool calls, cacheable savings. |
| `vcr intel routes [--dir] [--prefer a,b] [--min-success 0.8] [--max-cost USD]` | Pick a model from evidence. |
| `vcr intel recover SESSION.vcr [--run-id ID] [--execute --handlers H]` | Recovery plan, or resume an interrupted workflow run. Also `--runs-dir`, `--vcr-dir`. |

**Next:** [Workflow engine](workflow.md) · [SuperAgent](superagent.md) ·
[Agent CI](../agent-ci.md)
