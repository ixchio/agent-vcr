# ⏪ Recording, time travel & Ghost Replay

A VCR for your agent's brain: record every step, rewind to any of them, branch off with
"what if?", and replay a run you liked without paying the model again.

**TL;DR**

- `VCRRecorder` writes each step, LLM call, tool call, error and checkpoint to a `.vcr` file (one
  JSON object per line). `VCRPlayer` lets you jump, diff, fork and resume.
- `vcr run.vcr` opens a terminal debugger. `vcr bundle export` turns a run into a checksummed
  `.vcrb` file you can attach to a bug report.
- **Ghost Replay** promotes a reviewed run to a "golden run" and hands back its recorded outputs:
  no model calls, no tokens. Re-run only the steps you changed.

## Try it in 60 seconds

```bash
pip install "ai-agent-vcr[tui]"
```

Save this as `record_demo.py` and run `python record_demo.py`. No API key needed: the "LLM call"
is made up, the recording is real.

```python
from agent_vcr import VCRPlayer, VCRRecorder

rec = VCRRecorder(output_dir=".vcr")
rec.start_session("support-triage", tags=["demo"])

ticket = {"ticket_id": "t_123", "text": "I was charged twice", "status": "new"}
rec.record_llm_call(
    model="demo-model",
    messages=[{"role": "user", "content": ticket["text"]}],
    response={"content": "billing"},
    tokens_input=40, tokens_output=2, latency_ms=310.0, cost_usd=0.0004,
)
rec.record_tool_call("lookup_payments", {"ticket_id": "t_123"}, {"charges": 2}, latency_ms=12.0)
done = {**ticket, "status": "classified", "queue": "billing"}
rec.record_step("classify", input_state=ticket, output_state=done)
rec.record_checkpoint(done, label="classified")
path = rec.save()

player = VCRPlayer.load(path)
print(player.list_nodes())
print(player.goto_checkpoint("classified")["queue"])
print(player.get_total_tokens(), player.get_total_cost())
```

```text
['llm:demo-model', 'tool:lookup_payments', 'classify', 'checkpoint:classified']
billing
42 0.0004
```

Now scrub through it in the terminal:

```bash
vcr .vcr/support-triage.vcr
```

`j`/`k` (or the arrow keys) move between frames, `1` `2` `3` show input / output / diff, `s`
searches, `e` edits a state, `r` writes the selected (or edited) state to `resume_state.json`
next to the recording so a script can pick it up, `q` quits. `vcr tui <file>` does the same thing.
For the web dashboard, see [dashboard.md](dashboard.md).

## How it works

```text
 your agent ──record_*()──▶ VCRRecorder ──append──▶ .vcr/support-triage.vcr
                                                      line 1: session header
                                                      line 2+: one frame each
                                                                │
         ┌───────────────────────────┬────────────────────────┤
         ▼                           ▼                        ▼
   VCRPlayer                  GoldenRunCache              vcr bundle
   goto / diff / fork /       Ghost Replay: hand back     .vcrb = recording +
   resume, terminal UI        recorded outputs            sha256 manifest
```

- A **frame** is one step: `node_name`, `frame_type` (`node`, `llm`, `tool`, `error`,
  `checkpoint`), `input_state`, `output_state`, a timestamp, and metadata (model, latency, tokens,
  cost, your own `custom` dict). Typed helpers name frames `llm:<model>`, `tool:<name>` and
  `checkpoint:<label>`.
- **The `.vcr` file is JSONL.** Line 1 is the session header (id, parent, tags, totals, schema
  `execution/v1`). Every other line is a frame, in execution order. Frames are buffered (100 by
  default), appended with `fsync`, and `save()` rewrites the header atomically so the totals
  match. `manifest.json` in the same folder indexes your sessions.
- **The reader is forgiving.** A truncated or malformed line is skipped and counted in
  `player.skipped_lines`. Load with `VCRPlayer.load(path, strict=True)` to fail instead. Unknown
  fields are ignored, so older and newer files keep loading.
- **Recording never crashes your agent.** Sets, bytes, dataclasses, Pydantic models, datetimes
  and arbitrary objects are stored safely: sets and bytes come back as themselves, dataclasses and
  models as plain dicts, anything else as its `repr`.
- **Reusing a session id is safe.** `start_session("x")` when `x.vcr` already has data moves the
  old run to `x.vcr.prev` (one backup, so a third run replaces it). Pass
  `if_exists="overwrite"` or `if_exists="error"` to change that.

## Everyday recipes

These use the `.vcr/support-triage.vcr` file from the 60-second demo.

### Jump to any frame, checkpoint or moment

```python
from agent_vcr import VCRPlayer

player = VCRPlayer.load(".vcr/support-triage.vcr")

state = player.goto_frame(2)                 # output state after frame 2
print(state["status"])
print(player.get_input_state(2)["status"])   # what frame 2 saw going in
print(player.list_checkpoints())
print(player.goto_time(player.frames[1].timestamp))
print(player.get_errors(), player.get_total_latency() > 0)

diff = player.compare_frames(1, 2)           # output of frame 1 vs output of frame 2
print(sorted(diff["added"]), diff["removed"])
print(player.verify_integrity())
```

`goto_time` takes a `datetime` or an ISO 8601 string and picks the closest frame.
`compare_frames` returns `added`, `removed`, `modified` (with `before` / `after`) and `unchanged`.
`verify_integrity()` checks that every frame belongs to the session, that timestamps don't go
backwards, and that the header's frame count matches.

### Record from async code

```python
import asyncio

from agent_vcr import AsyncVCRRecorder


async def main() -> None:
    async with AsyncVCRRecorder(output_dir=".vcr") as rec:   # saves on exit
        await rec.start_session("async-run")
        await rec.record_llm_call("demo-model", [{"role": "user", "content": "hi"}],
                                  {"content": "hello"}, tokens_input=3, tokens_output=1)
        await rec.record_tool_call("search", {"q": "refund policy"}, ["doc-1"])
        await rec.record_checkpoint({"answer": "hello"}, label="answered")


asyncio.run(main())
```

Same API as the sync recorder, with `await` and non-blocking file I/O. `AsyncVCRPlayer`
(`from agent_vcr import AsyncVCRPlayer`, then `await AsyncVCRPlayer.load(path)`) covers the
navigation half: `goto_frame`, `goto_time`, `goto_checkpoint`, `list_checkpoints`, totals and
`verify_integrity`. The sync `VCRRecorder` also works as a plain `with` block that saves on exit.

### Fork: "what if the plan had been different?"

```python
from agent_vcr import VCRRecorder

rec = VCRRecorder(output_dir=".vcr")
rec.start_session("plan-run")
rec.record_step("plan", {"goal": "refund"}, {"goal": "refund", "plan": "refund in full"})
rec.record_step("act", {"plan": "refund in full"}, {"refunded": 100})
rec.save()

# "What if the plan had been different?" Branch after frame 0 with an override.
branch = rec.fork(from_frame=0, new_session_id="plan-run-partial",
                  state_overrides={"plan": "refund half"})
branch.record_step("act", {"plan": "refund half"}, {"refunded": 50})
branch.save()
print(branch.get_session().parent_session_id, [f.node_name for f in branch.frames])
```

```text
plan-run ['fork:plan', 'act']
```

`fork()` starts a child session from frame N's **output** state and remembers its parent. The
original recording is left alone. Pass `in_place=True` to save the current session and keep
recording the branch on the *same* recorder object, which matters when a framework wrapper
(LangGraph, CrewAI) holds a reference to it.

### Resume: re-run your code from a recorded frame

```python
from agent_vcr import ResumeConfig, ResumeMode, VCRPlayer


def act(state: dict) -> dict:
    return {"refunded": 50 if state["plan"] == "refund half" else 100}


player = VCRPlayer.load(".vcr/plan-run.vcr")
child = player.resume(
    act,
    ResumeConfig(from_frame=1, mode=ResumeMode.FORK, state_overrides={"plan": "refund half"},
                 new_session_id="plan-run-retry"),
)
print(child, VCRPlayer.load(".vcr/plan-run-retry.vcr").goto_frame(0))
```

```text
plan-run-retry {'refunded': 50}
```

`resume()` calls your function **once** with frame N's **input** state (plus overrides) and
records the result as a `resumed_execution` frame in a child session. The child is written next
to the recording it came from, whatever your working directory is.

| Mode | What your function gets |
|---|---|
| `FORK` (default) | Frame N's input state with `state_overrides` applied |
| `REPLAY` | Frames before N are copied into the child; your function gets the output of frame N-1. Overrides are not applied |
| `MOCK` | Like `FORK`, plus your `inject_mocks` dict under the `__mocks__` key |

### Keep appending to an existing run

```python
from agent_vcr import VCRRecorder

rec = VCRRecorder(output_dir=".vcr")
rec.attach_session("support-triage")      # load it and append, don't start over
rec.record_step("reply", {"queue": "billing"}, {"sent": True})
rec.save()
```

Handy after a crash: totals keep counting from where the file left off.

### Ship a bug report as a bundle

```bash
vcr bundle export .vcr/support-triage.vcr -o support-triage.vcrb
vcr bundle verify support-triage.vcrb
vcr reproduce support-triage.vcrb
vcr bundle import support-triage.vcrb -d imported/
```

A `.vcrb` is a gzipped tar holding `execution.vcr` (the recording, byte for byte) and
`bundle.json`: session id, schema version, frame count, the SHA-256 of `execution.vcr`, the git
commit of the folder it came from (when there is one), models, tools, checkpoints, error count,
tokens and cost. Without `-o`, the bundle lands next to the recording as `<session_id>.vcrb`.

- `export` refuses a recording that fails `verify_integrity()`.
- `verify` checks the hash without unpacking, prints a JSON report with `"ok"`, and exits `1`
  when the bundle is tampered or unreadable — safe to use as a CI step.
- `reproduce` verifies, loads and integrity-checks the recording, then prints a summary. It
  **never runs agent code**. It exits `1` if anything is off.
- `import` verifies first, then writes `<session_id>.vcr` into the folder. It rejects a session
  id that would write outside that folder.

### 👻 Ghost Replay: re-use a run you trust

```python
from agent_vcr import FrameMetadata, GhostReplayMode, GoldenRunCache, VCRRecorder

# 1. A run you reviewed and liked (pretend each step cost real tokens).
rec = VCRRecorder(output_dir=".vcr")
rec.start_session("triage-golden")
rec.record_step("classify", {"text": "charged twice"}, {"queue": "billing"},
                metadata=FrameMetadata(tokens_used=300, cost_usd=0.003))
rec.record_step("draft", {"queue": "billing"}, {"reply": "Refund on its way"},
                metadata=FrameMetadata(tokens_used=900, cost_usd=0.009))
rec.record_step("send", {"reply": "Refund on its way"}, {"sent": True},
                metadata=FrameMetadata(tokens_used=0, cost_usd=0.0))
rec.save()

# 2. Promote it, keyed by the task AND everything that should invalidate it.
cache = GoldenRunCache(".vcr/golden")
identity = cache.build_identity(model="demo-model@2026-01", prompt_hash="sha256:abc123",
                                code_commit="0123abc")
cache.save_golden_run("triage ticket", rec, identity=identity)

# 3. Replay it: no model calls, no tokens.
outputs, ledger = cache.replay_with_mode("triage ticket", GhostReplayMode.FULL, identity=identity)
print(outputs[-1], ledger)
```

```text
{'sent': True} CostLedger(saved=100% | $0.0120 | 1200 tokens | -0ms)
```

Each replay is itself recorded as a new session in `.vcr/` (pass `recorder=` to choose where).
The cache is `index.json` plus one `<fingerprint>.vcr` per golden run, stored by file name, so you
can commit `.vcr/golden/` and it works from any directory or CI checkout.

### 👻 Re-run only what you changed

```python
from agent_vcr import GhostReplayMode, GoldenRunCache

cache = GoldenRunCache(".vcr/golden")
identity = cache.build_identity(model="demo-model@2026-01", prompt_hash="sha256:abc123",
                                code_commit="0123abc")


def run_step(node_name: str, state: dict) -> dict:
    """Your real step code. Ghost Replay calls it only for the steps it re-runs."""
    print("  re-running", node_name, "with", state)
    if node_name == "draft":
        return {"reply": "New, friendlier refund reply"}
    if node_name == "send":
        return {"sent": True}
    return {"queue": "billing"}


# You changed the draft prompt (frame 1). HYBRID re-runs it and everything after it.
outputs, ledger = cache.replay_with_mode("triage ticket", GhostReplayMode.HYBRID,
                                         step_executor=run_step, changed_steps={1},
                                         identity=identity)
print(ledger.summary()["step_sources"], f"{ledger.savings_percent:.0f}% saved")

# LIVE re-runs only frame 1 and trusts the cache for the rest.
_, ledger = cache.replay_with_mode("triage ticket", "live", step_executor=run_step,
                                   changed_steps={1}, identity=identity)
print(ledger.summary()["step_sources"])

# BRANCH: replay history before frame 2, re-run frame 2 onward.
_, ledger = cache.replay_with_mode("triage ticket", "branch", step_executor=run_step,
                                   from_frame=2, identity=identity)
print(ledger.summary()["step_sources"])
```

```text
  re-running draft with {'queue': 'billing'}
  re-running send with {'reply': 'New, friendlier refund reply'}
['golden_cache', 'changed_step', 'downstream_invalidated'] 25% saved
  re-running draft with {'queue': 'billing'}
['golden_cache', 'changed_step', 'golden_cache']
  re-running send with {'reply': 'Refund on its way'}
['golden_cache', 'golden_cache', 'changed_step']
```

| Mode | Re-runs | Use it when |
|---|---|---|
| `full` | Nothing. Every output comes from the recording | Demos, fixtures, "show me exactly what happened" |
| `hybrid` | `changed_steps` **and every step after the first one** | The safe default: later steps may depend on what changed |
| `live` | Only `changed_steps` | You know the other steps don't depend on them |
| `branch` | `from_frame` and everything after it | Time-travel debugging from one point |

A re-run step gets the previous step's output as its input (or the recorded input for the first
step). `full` takes no `step_executor`; every other mode requires one.

**Identity is what makes a cache hit safe.** `build_identity()` takes `model`, `prompt_hash`,
`code_commit`, `tool_schema_hash`, `dependency_lock_hash`, `environment` and an `extra` dict, and
drops the ones you leave out. It is normalized with sorted keys (nested too), so key order never
causes a miss, but any real change does. The task string is matched after trimming and
lowercasing. With no identity at all, only the task string has to match: fine for a demo, too
weak for real reuse. A miss raises `KeyError`.

**The cost ledger** (`ledger.summary()`) compares the replay with the golden run: tokens, cost
and latency for each, what was saved, `steps_replayed`, `steps_rerun`, `step_sources` and
`cache_hit_reason`. Re-run steps are charged the golden frame's recorded tokens and cost, because
your step function doesn't report usage (`rerun_cost_estimated` is `true` then).
`savings_percent` is measured by cost, falls back to tokens, then to the share of replayed steps
when the golden run has no cost data. So a replay that re-ran everything reports 0%, not 100%.

## What it can't do

- **Ghost Replay is not an HTTP or LLM interceptor.** It hands back recorded *step outputs*,
  keyed by task + identity. It doesn't patch your model client, and it doesn't detect what
  changed: you choose `changed_steps` (or `from_frame`) yourself.
- **"Zero cost" means zero provider calls on a `full` replay.** Local CPU and disk still apply,
  and partial modes run your real code, with its real cost.
- **Resume and fork rewind the recording, not the world.** Files on disk, databases and sent
  emails stay as they are. For file rollback inside your own agent use
  [ACIDWorkspace](acid-workspace.md). For coding-agent CLIs, use the [airbag](airbag.md).
- **`resume()` runs your function once,** not step by step through the rest of the graph. Your
  function decides how far it goes.
- **A bundle proves bytes, not correctness.** The hash covers `execution.vcr` only. `bundle.json`
  is descriptive metadata and is not signed.
- **Recordings hold what your agent saw:** prompts, state, model replies, tool arguments and
  results, errors. Redact secrets before recording, keep `.vcr` folders and CI artifacts private,
  use synthetic fixtures, and review a run before promoting it to golden.
- `.vcr.prev` keeps one previous run per session id, not a history.

## Cheat sheet

| Do this | With |
|---|---|
| Start / attach a recording | `VCRRecorder(output_dir=".vcr", diff_mode=False)`, `.start_session(id, tags=, metadata=, if_exists="rotate")`, `.attach_session(id)` |
| Record | `.record_step(node, input, output, metadata=)`, `.record_llm_call(...)`, `.record_tool_call(...)`, `.record_error(node, input, exc, latency_ms)`, `.record_checkpoint(state, label=)` |
| Save | `.save()` (or use `with VCRRecorder(...) as rec:`) |
| Async | `AsyncVCRRecorder` (same methods, `await`), `await AsyncVCRPlayer.load(path)` |
| Load | `VCRPlayer.load(path, strict=False)`, `VCRPlayer.load_by_id(id, vcr_dir=".vcr")` |
| Navigate | `goto_frame(i)`, `goto_checkpoint(label)`, `goto_time(ts)`, `get_input_state(i)`, `get_output_state(i)`, `get_state_at_node(name)` |
| Inspect | `list_nodes()`, `list_checkpoints()`, `get_errors()`, `get_node_executions(name)`, `get_total_tokens()`, `get_total_cost()`, `get_total_latency()`, `verify_integrity()`, `to_dict()` |
| Diff | `compare_frames(a, b)` |
| Branch | `recorder.fork(from_frame, new_session_id=, state_overrides=, in_place=False)` |
| Re-run | `player.resume(fn, ResumeConfig(from_frame=, mode=, state_overrides=, inject_mocks=, new_session_id=))` |
| Terminal debugger | `vcr run.vcr` or `vcr tui run.vcr` (needs `ai-agent-vcr[tui]`) |
| Bundles | `vcr bundle export <file.vcr> [-o out.vcrb]`, `vcr bundle verify`, `vcr bundle import [-d dir]`, `vcr reproduce` |
| Golden runs | `GoldenRunCache(".vcr/golden")`, `.save_golden_run(task, rec, identity=)`, `.has_golden_run`, `.list_golden_runs()`, `.invalidate(task, identity)` |
| Ghost Replay | `.replay_with_mode(task, mode, step_executor=, changed_steps=, from_frame=, identity=, recorder=)` |
| Identity | `GoldenRunCache.build_identity(model=, prompt_hash=, code_commit=, tool_schema_hash=, dependency_lock_hash=, environment=, extra=)` |

`diff_mode=True` saves space on long runs: it stores a JSON-patch delta per frame and leaves out
an input that is identical to the previous output. The player rebuilds it, and inputs that differ
(LLM prompts, tool arguments) are always kept.

**Next:** [Agent CI](agent-ci.md) (golden runs as regression tests) ·
[ACID workspace](acid-workspace.md) (roll back files too) ·
[Integrations](integrations/README.md) (record LangGraph and CrewAI automatically)
