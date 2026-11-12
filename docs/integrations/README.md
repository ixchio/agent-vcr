# Integrations — plug Agent VCR into whatever you use 🔌

Raw Python, LangGraph, CrewAI, OpenHands or Claude Code: a few lines and every step of your agent
lands in a `.vcr` file you can rewind, replay and test against.

**TL;DR**

- Every integration writes the same thing: one JSONL frame per step (node, LLM call, tool call,
  error, checkpoint). Same file format, same tools afterwards.
- LangGraph: pass `LangGraphCallback` in `config` — no graph changes. CrewAI: wrap `kickoff()`.
- Recording never breaks your agent: integration errors are logged, your exceptions and return
  values pass through untouched.

## Try it in 60 seconds

No framework, no API key. Save as `raw.py` and run `python raw.py`:

```python
from agent_vcr import VCRRecorder


def classify(state):
    return {**state, "label": "network"}


recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session("my-agent")

before = {"task": "classify incident"}
after = classify(before)
recorder.record_step("classify", before, after)
recorder.record_llm_call(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Classify: DNS timeout"}],
    response={"content": "network"},
    tokens_input=12,
    tokens_output=1,
    latency_ms=340.0,
)
recorder.record_tool_call("lookup_runbook", {"label": "network"}, {"url": "runbooks/network"},
                          latency_ms=8.0)
try:
    raise TimeoutError("pager API timed out")
except TimeoutError as exc:
    recorder.record_error("notify", {"channel": "#ops"}, exc, latency_ms=5000.0)
recorder.record_checkpoint(after, label="classified")

print(recorder.save())  # .vcr/my-agent.vcr
```

Open it with `vcr .vcr/my-agent.vcr` (needs the `[tui]` extra — see [dashboard.md](../dashboard.md)).

## How it works

```text
 your framework       adapter call           recorder         disk
 ──────────────       ──────────────────     ─────────────    ──────────────
 node finished  ──▶   record_step()      ─┐
 LLM answered   ──▶   record_llm_call()  ─┤
 tool returned  ──▶   record_tool_call() ─┼─▶ VCRRecorder ──▶ .vcr/<id>.vcr
 exception      ──▶   record_error()     ─┘   (thread-   │    (JSONL, one
                                               safe)     │     frame/line)
                                                         └──▶ on_frame_recorded()
```

- Adapters only translate framework events into the five `record_*` calls. Nothing else.
- Frames are buffered and flushed every 100 frames and on `save()`; `with VCRRecorder() as r:`
  saves on exit.
- `start_session("id")` never silently destroys an old run: an existing `id.vcr` is moved to
  `id.vcr.prev` (pass `if_exists="overwrite"` or `"error"` to change that).
- What you do with the file — time travel, Ghost Replay, CI — lives in
  [recording-and-replay.md](../recording-and-replay.md) and [agent-ci.md](../agent-ci.md).

Install the extra that matches your framework:

```bash
pip install ai-agent-vcr                 # raw Python, Claude Code recording, Sentinel
pip install "ai-agent-vcr[langgraph]"    # LangGraph
pip install "ai-agent-vcr[crewai]"       # CrewAI
pip install "ai-agent-vcr[openhands]"    # OpenHands SDK 1.x (Python 3.12+)
pip install "ai-agent-vcr[openhands-legacy]"  # OpenHands legacy V0 (openhands-ai event stream)
```

## Raw Python

The five calls from the 60-second demo are the whole API:

| Call | Frame type | Use it for |
| --- | --- | --- |
| `record_step(node_name, input_state, output_state, metadata=None)` | node | any step: state in, state out |
| `record_llm_call(model, messages, response, tokens_input, tokens_output, latency_ms, cost_usd=None)` | LLM call | model calls (tokens and cost add up per session) |
| `record_tool_call(tool_name, tool_input, tool_output, latency_ms, error=None)` | tool call | tools and APIs |
| `record_error(node_name, input_state, error, latency_ms, model=None)` | error | exceptions |
| `record_checkpoint(state, label="checkpoint")` | checkpoint | named restore points |

Prefer decorators? `vcr_record(recorder, node_name=...)` (in `agent_vcr.integrations.langgraph`)
and `vcr_task(recorder, task_name=...)` / `vcr_task_async(...)` (in
`agent_vcr.integrations.crewai`) record any plain function. Neither needs the framework installed.

### Live forwarding with `on_frame_recorded`

Give the recorder a callback and it is called with every new `Frame` right after it is recorded.
Use it to stream frames to the dashboard's `/api/push` endpoint (run `vcr serve` first — see
[dashboard.md](../dashboard.md)):

```python
import os
import urllib.request

from agent_vcr import VCRRecorder

SERVER = "http://127.0.0.1:8000"
TOKEN = os.environ.get("AGENT_VCR_AUTH_TOKEN", "")


def push(frame):
    req = urllib.request.Request(
        f"{SERVER}/api/push",
        data=frame.model_dump_json().encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=2).close()


recorder = VCRRecorder(on_frame_recorded=push)
recorder.start_session("live-demo")
recorder.record_step("plan", {"task": "ship it"}, {"plan": ["test", "deploy"]})
recorder.save()
```

The callback runs inline, inside the recorder's lock, so keep it fast (short timeout, or hand the
frame to a background queue). If it raises, the error is swallowed and logged at debug level — the
frame is still recorded.

## LangGraph

> Needs `pip install "ai-agent-vcr[langgraph]"`. Snippets below assume you already have a
> `StateGraph` builder named `builder`.

### Recommended: a callback, zero graph changes

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import LangGraphCallback

recorder = VCRRecorder()
recorder.start_session("langgraph-run")

app = builder.compile()
result = app.invoke(
    {"task": "repair the API"},
    config={"callbacks": [LangGraphCallback(recorder)]},
)
recorder.save()
```

You get one frame per graph node (from LangGraph's `langgraph_node` metadata), plus LLM calls with
token usage when the provider reports it, plus tool calls. Turn the extras off with
`LangGraphCallback(recorder, record_llm_calls=False, record_tool_calls=False)`.

### Wrap the whole graph

```python
from agent_vcr.integrations.langgraph import VCRLangGraph

VCRLangGraph(recorder).wrap_graph(builder)   # before compile() — recommended
app = builder.compile()

# or, on an already compiled graph:
# VCRLangGraph(recorder).wrap_graph(app)
```

`wrap_graph` wraps every node in place. LangGraph keeps injecting `config` / `runtime` / `store`,
retry and cache policies still apply, compiled subgraphs and `__start__` are left alone, and
wrapping twice is a no-op.

### Wrap single nodes

```python
vcr = VCRLangGraph(recorder)
builder.add_node("planner", vcr.wrap_node("planner", planner))
builder.add_node("fetch", vcr.wrap_node("fetch", fetch_docs))   # async def works too
```

Sync and `async def` nodes are both supported, and the original signature is preserved.

### `Command` returns and interrupts

A node that returns `Command(update=..., goto=...)` is recorded as
`{"__command__": {"goto": ...}, "update": {...}}`. LangGraph control flow (`interrupt()`,
`Command(graph=Command.PARENT)`, cancellation) is **not** recorded as an error.

Agent VCR does not replace LangGraph's checkpointer. Keep the checkpointer for graph-native
persistence; add Agent VCR for portable recordings, Ghost Replay, Agent CI and filesystem rollback.

## CrewAI

> Needs `pip install "ai-agent-vcr[crewai]"`. Snippets assume a `crew = Crew(agents=..., tasks=...)`.

### Wrap the kickoff (simplest)

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI

recorder = VCRRecorder()
recorder.start_session("research-crew")

vcr_crew = VCRCrewAI(recorder)
result = vcr_crew.kickoff(crew, inputs={"topic": "content-aware rollback"})
# result = await vcr_crew.kickoff_async(crew, inputs={...}, timeout=300)
# results = vcr_crew.kickoff_for_each(crew, [{"topic": "a"}, {"topic": "b"}])
recorder.save()
```

For the duration of the call it installs recording callbacks on the crew, every agent
(`step_callback`) and every task (`callback`), chains any callbacks you set yourself, and restores
all of them afterwards — even if the kickoff raises. Each agent step and each finished task becomes a
frame; a failed kickoff becomes an error frame.

With `kickoff_async(timeout=...)`, CrewAI keeps running its worker thread after the timeout; the
VCR callbacks are removed regardless.

### Listen on CrewAI's event bus

```python
from agent_vcr.integrations.crewai import VCRCrewCallback

with VCRCrewCallback(recorder):
    crew.kickoff()

# or, manually:
listener = VCRCrewCallback(recorder).attach()
try:
    crew.kickoff()
finally:
    listener.detach()
```

This records task start/finish/failure, agent runs, tool usage and LLM calls
(`record_llm_calls=False` to skip those). The event bus is process-global: while attached, events
from **every** crew in the process go into this recorder.

⚠️ `Crew(callbacks=[VCRCrewCallback(recorder)])` does nothing — CrewAI's `Crew` has no `callbacks`
field. A listener that was never attached emits a `RuntimeWarning` saying exactly that when it is
garbage-collected. Use `with` or `attach()`.

## OpenHands

> Targets the current **OpenHands SDK 1.x** (`pip install openhands-sdk openhands-tools`,
> verified against 1.7.3). The legacy V0 `openhands-ai` event stream still works through
> `attach()` — see the end of this section.

- **Record every tool call:** `VCRRuntime` is a conversation callback that writes one frame per
  tool call — arguments, output, exit code, error, latency.
- **Stop destructive commands before they run:** the airbag snapshots the workspace before every
  mutating action and blocks the irreversible ones. Full guide and hook install:
  [airbag.md](../airbag.md).
- **Code-quality guard:** Sentinel checks every file `file_editor` writes and, on a blocker,
  rejects the agent's next action until the file is fixed. Guide: [sentinel.md](../sentinel.md).
- **Rollback for any repo-writing agent:** `ACIDWorkspace` gives your agent a git-branch
  transaction with savepoints and rollback. Full guide: [acid-workspace.md](../acid-workspace.md).

All three callbacks together (`agent`, `ws` are yours):

```python
from openhands.sdk import Conversation

from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect
from agent_vcr.integrations.openhands_hook import VCRRuntime
from openhands_sentinel import Sentinel

airbag = AirbagCallback()                 # snapshot first, block irreversible commands
vcr = VCRRuntime(session_id="fix-login")  # one frame per tool call
sentinel = Sentinel()                     # AST check after every file_editor write
conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag, vcr, sentinel])
protect(conversation, airbag)
sentinel.bind(conversation)

conversation.send_message("Fix the failing login test")
conversation.run()

print(vcr.save())          # .vcr/fix-login.vcr
print(sentinel.save())     # .vcr/sentinel-openhands.vcr
```

OpenHands calls every callback synchronously, *before* an action runs and again when its result
arrives. `VCRRuntime` pairs each `ActionEvent` with its result by `tool_call_id`:

| Frame field | What's in it |
| --- | --- |
| `node_name` | `openhands:<tool>` — `openhands:terminal`, `openhands:file_editor`, ... |
| `input_state` | `tool`, `args` (the action, e.g. `command` or `command`/`path`/`file_text`), `tool_call_id`, `action_id`, `thought` |
| `output_state.status` | `ok`, `error` (tool error / `AgentErrorEvent`), `rejected` (`UserRejectObservation`), `no_observation` |
| `output_state` | `output` (observation text), `exit_code` (terminal), `observation` (`path`, `new_content`, ...) |
| `metadata` | `latency_ms`; `error_type` / `error_message` for errors and rejections |

- Frames over 50 KB are logged and replaced by a 1000-character preview that keeps `tool`,
  `status` and `exit_code` (`max_frame_bytes=`, `truncate_oversized=False` to only warn).
- `save()` also records actions that never got a result (the tool crashed) as `no_observation`.
- If the recorder you pass already has an active session, frames are appended to it — give
  `VCRRuntime` and `Sentinel` the same `VCRRecorder` to get one file.
- Recording problems are logged, never raised into the agent run. Message, state and system
  events are not recorded.

**Legacy V0 (`openhands-ai`, `pip install "ai-agent-vcr[openhands-legacy]"`):** call
`VCRRuntime().attach(runtime.event_stream)` instead. Every event on the stream becomes its own
frame (`openhands:CmdRunAction`, `openhands:CmdOutputObservation`, ...), with the same size
guardrail. `attach()` refuses an SDK conversation: SDK callbacks can only be passed when the
`Conversation` is created.

## Claude Code — record every session 🎥

```bash
vcr init --claude-code
```

This creates `.vcr/` and merges recording hooks for `PreToolUse`, `PostToolUse`, `FileChanged`,
`Stop` and `SessionEnd` into `.claude/settings.json`. From then on, each Claude Code session is
recorded to `.vcr/claude-<session_id>.vcr`, one frame per hook event.

- Your existing hooks are kept. Re-running the command replaces its own entries instead of
  adding duplicates.
- Each hook is a fast, stdlib-only process with a 10 s timeout. It always exits 0, so a recording
  problem can never block Claude Code.
- Long strings (file bodies, tool output) are cut to 8192 characters
  (`AGENT_VCR_HOOK_MAX_FIELD_CHARS`). `AGENT_VCR_DIR` changes the output folder.
- The hook command uses the absolute path of the Python that ran `vcr init`. Review
  `.claude/settings.json` before committing it — hooks run local commands, and that path may not
  exist on a teammate's machine.

This **records** what Claude did. It does not stop or undo anything. For snapshots before every
Bash/Write/Edit and `vcr undo`, install the airbag: `vcr airbag install claude` —
see [airbag.md](../airbag.md). You can use both.

## Build your own adapter 🛠️

Got a framework we don't cover? An adapter is a thin mapping. Checklist:

1. **Map events to calls:** node done → `record_step`, model done → `record_llm_call`, tool done →
   `record_tool_call`, exception → `record_error`, durable boundary → `record_checkpoint`.
2. **Never change behavior:** return the framework's value as-is and re-raise its exception. Wrap
   your recording code in `try/except` and log — recording must never break the run.
3. **Don't record control flow as errors** (interrupts, cancellation, "go to parent" signals).
4. **Redact before recording.** Whatever you pass in is written to disk.
5. **Measure, don't guess:** time with `time.perf_counter()`; only report tokens the provider
   actually returned.
6. **No duplicate frames:** if two hooks deliver the same event, record it once.
7. **Restore what you patch** (callbacks, methods) in a `finally`.
8. **Test** the sync, async, error and cancellation paths.

The JSON schemas in [`../../schemas/`](../../schemas/) are the language-neutral file contract.

## What it can't do

- **Frameworks change their callbacks.** Payloads come from LangGraph and CrewAI; re-check your
  recordings when you upgrade them.
- **OpenHands SDK: local conversations only.** `VCRRuntime` and Sentinel were verified with a
  local `Conversation` (callbacks in the same process). A `RemoteConversation` against an agent
  server is untested. `latency_ms` is measured between callbacks, so it is the tool's wall time
  plus a little SDK overhead, not a profiler reading.
- **Recording is not rollback.** These adapters capture state; undoing file changes is
  [ACIDWorkspace](../acid-workspace.md) or the [airbag](../airbag.md).
- **The CrewAI event-bus listener is process-wide**, so two crews running at once in one process
  land in the same recording. Use `VCRCrewAI.kickoff` per crew instead.
- **Only what you pass gets recorded.** Non-JSON values are serialized best-effort; huge states
  make huge files.
- **No auto-instrumentation.** Nothing is patched globally; you opt in per run.

## Cheat sheet

| You use | Do this | Needs |
| --- | --- | --- |
| Plain Python | `VCRRecorder().start_session(...)` + `record_*` | — |
| Live push | `VCRRecorder(on_frame_recorded=fn)` | `vcr serve` running |
| LangGraph | `app.invoke(x, config={"callbacks": [LangGraphCallback(rec)]})` | `[langgraph]` |
| LangGraph | `VCRLangGraph(rec).wrap_graph(builder)` / `.wrap_node(name, fn)` | `[langgraph]` |
| CrewAI | `VCRCrewAI(rec).kickoff(crew)` / `kickoff_async` / `kickoff_for_each` | `[crewai]` |
| CrewAI | `with VCRCrewCallback(rec): crew.kickoff()` | `[crewai]` |
| OpenHands SDK | `Conversation(..., callbacks=[VCRRuntime()])` | `openhands-sdk` |
| OpenHands SDK | `protect(conversation, AirbagCallback())` | `openhands-sdk` |
| OpenHands V0 | `VCRRuntime().attach(runtime.event_stream)` | `[openhands-legacy]` |
| Claude Code | `vcr init --claude-code` | — |
| n8n / HTTP | `POST /api/workflows/{name}/trigger` | `[dashboard]` |

## Next

- [n8n.md](n8n.md) — trigger Agent VCR workflows from n8n over HTTP.
- [recording-and-replay.md](../recording-and-replay.md) — what to do with the `.vcr` file.
- [dashboard.md](../dashboard.md) — watch runs live in the terminal or browser.
