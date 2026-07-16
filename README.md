<div align="center">

# 📼 Agent VCR

### Your AI agent corrupted the codebase. Now what?
**The only tool that physically deletes hallucinated files — `git reset --hard`, not just state rollback.**

<br>
<a href="https://pypi.org/project/ai-agent-vcr/"><img src="https://img.shields.io/pypi/v/ai-agent-vcr?style=flat-square&color=00d4aa&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/ixchio/agent-vcr/actions"><img src="https://img.shields.io/github/actions/workflow/status/ixchio/agent-vcr/ci.yml?style=flat-square&label=CI" alt="CI"></a>
<a href="https://codecov.io/gh/ixchio/agent-vcr"><img src="https://img.shields.io/codecov/c/github/ixchio/agent-vcr?style=flat-square&color=58a6ff" alt="Coverage"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue?style=flat-square" alt="Python"></a>
<br><br>

```bash
pip install ai-agent-vcr
```

No API keys. No cloud. No vendor lock-in.
Works with [TERX](https://github.com/ixchio/terx) — memory layer for browser agents.

<br>

</div>

---

## The Problem Every AI Agent Developer Has Hit

You ran Claude Code, OpenHands, or a LangGraph agent autonomously.

It wrote 40 files. It failed at step 8. Now you have:
- 23 files that shouldn't exist
- A broken import chain
- State that says "success" on steps that half-ran
- No way to know what the repo looked like before step 6

```
Your repo after a bad autonomous run:

/src/handlers.py          ← hallucinated, breaks import
/src/auth_v2.py           ← duplicate of auth.py, never needed
/src/models_refactor.py   ← partial rewrite, syntax error
/tests/test_fake.py       ← tests for code that doesn't exist
/config/settings_new.py   ← overwrote working config
```

**Every other tool shows you logs. Agent VCR runs `git reset --hard` and deletes every one of those files.**

---

## ACID Rollback — The Feature Nobody Else Has

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

recorder = VCRRecorder()
acid = ACIDWorkspace("/my/workspace", recorder=recorder)

acid.begin(session_id="task-001")        # isolated git branch
acid.savepoint(state, node_name="coder") # checkpoint state + filesystem
acid.savepoint(state, node_name="tester")

# Agent writes 47 files. 23 are hallucinated garbage. Step 6 failed.
acid.rollback(to_frame_index=1)
# git reset --hard
# All 23 files: physically deleted from disk. Not hidden. Gone.
# Pre-existing ignored files like .env are preserved.

acid.commit()                            # merge only the clean branch
```

```
Before rollback:          After rollback:
/src/handlers.py    ✗     DELETED
/src/auth_v2.py     ✗     DELETED
/tests/fake_test.py ✗     DELETED
/src/utils.py       ✓     kept
/src/models.py      ✓     kept
```

**LangSmith** shows you what happened. **LangFuse** shows you what happened. **Arize** shows you what happened.

Agent VCR **changes** what happened.

---

## Ghost Replay — Never Pay for the Same Task Twice

Agent succeeds? Save it. Run it again for free forever.

```python
from agent_vcr.golden_cache import GoldenRunCache

cache = GoldenRunCache()
identity = GoldenRunCache.build_identity(
    model="gpt-4o",
    prompt_hash="prompt:v3",
    code_commit="abc123",
    tool_schema_hash="tools:v1",
)
cache.save_golden_run("Build a REST API with JWT auth", recorder, identity=identity)

# Every future run of the same task:
outputs, ledger = cache.replay("Build a REST API with JWT auth", identity=identity)
print(ledger)
```

```
RUN 1 (original)      RUN 2 (Ghost Replay)
─────────────────     ─────────────────────
Tokens:   4,100       Tokens:      0
Cost:   $0.0123       Cost:    $0.00
Time:  2,350ms        Time:      1ms

💰 100% savings · $0.0123 saved · 4,100 tokens · 2,349ms faster
```

---

## Time-Travel Debugging

Agent fails at step 8 of 10? Don't re-run from zero.

```python
from agent_vcr import VCRPlayer
from agent_vcr.models import ResumeConfig

player = VCRPlayer.load(".vcr/my_run.vcr")

# See exact state at every step
print(player.goto_frame(6))   # {'files_written': [...], 'plan': '...'}
print(player.get_errors())    # what broke and where

# Fix the prompt. Resume from step 6. Skip steps 0-5.
player.resume(
    agent_callable=coder,
    config=ResumeConfig(
        from_frame=6,
        state_overrides={"plan": "use SQLAlchemy instead of raw SQL"}
    )
)
```

```
Without Agent VCR          With Agent VCR
──────────────────         ──────────────────────────
Agent fails step 8         Agent fails step 8
Patch the code             player.goto_frame(7)
Re-run ALL 10 steps        Fix the state
$0.04 + 2 min wasted       Resume from step 7
Repeat for every bug       Done. $0.00 extra.
```

---

## Who This Is For

**You need this if you're running:**
- Claude Code / Cursor autonomous mode
- OpenHands on real codebases
- LangGraph agents that write files
- CrewAI pipelines with filesystem access
- Any autonomous coding agent on a repo you care about

**You don't need this if you're only:**
- Doing RAG / chatbots (no filesystem risk)
- Already happy with LangSmith for tracing

---

## Quick Start

### Record

```python
from agent_vcr import VCRRecorder

recorder = VCRRecorder()
recorder.start_session("my_run")

state = {"query": "build a REST API"}
state = planner(state)
recorder.record_step("planner", input_state, state)

state = coder(state)
recorder.record_step("coder", input_state, state)

recorder.save()  # → .vcr/my_run.vcr
```

Or use the context manager — frames are saved even if the agent crashes:

```python
with VCRRecorder() as recorder:
    recorder.start_session("my_run")
    # ... your agent code ...
```

### Rewind & Fix

```python
player = VCRPlayer.load(".vcr/my_run.vcr")

diff = player.compare_frames(5, 6)
# {'added': {'bad_file': '...'}, 'modified': {'plan': '...'}}

player.resume(
    agent_callable=coder,
    config=ResumeConfig(from_frame=5, state_overrides={"plan": "fixed"})
)
```

---

## Integrations

### LangGraph — one line

```python
from langgraph.graph import StateGraph
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph

recorder = VCRRecorder()
graph = VCRLangGraph(recorder).wrap_graph(graph)  # ← one line, that's it

result = graph.invoke({"query": "Build a todo app"})
recorder.save()
```

### CrewAI

```python
from agent_vcr.integrations.crewai import VCRCrewAI

recorder = VCRRecorder()
recorder.start_session("crew_run")
result = VCRCrewAI(recorder).kickoff(crew)
recorder.save()
```

```bash
pip install "ai-agent-vcr[crewai]"
pip install "ai-agent-vcr[langgraph]"
```

### Raw Python (decorator)

```python
from agent_vcr.integrations.langgraph import vcr_record

@vcr_record(recorder, node_name="research_step")
def research(state: dict) -> dict:
    return {"findings": search(state["query"])}
```

---

## Sentinel — Real-Time Code Guardian

Catches what the agent wrote before it moves to the next step.

```python
from openhands_sentinel import Sentinel

sentinel = Sentinel(recorder=recorder)
sentinel.attach(runtime.event_stream)  # 3 lines. auto-intercepts every write.
```

```
STEP 2: Agent writes handlers.py
🛡️ SENTINEL: VIOLATIONS DETECTED
  CRITICAL  hash_password() already exists in auth/utils.py:8 — reuse it
  CRITICAL  handle_auth_request() is 109 lines (limit: 40) — break it up
  CRITICAL  Cyclomatic complexity: 32 (limit: 8)

STEP 3: Agent self-corrects
🛡️ SENTINEL: handlers.py — CLEAN ✓
```

| | Without Sentinel | With Sentinel |
|---|---|---|
| Agent writes bad code | ✓ | ✓ |
| Sentinel catches it | — | **< 10ms** |
| Agent self-corrects | — | **done** |
| Human reviews PR | manual | **zero** |
| Cost | 2× LLM + human time | 1 extra LLM call |

Standalone scan:

```bash
sentinel scan ./my-ai-project
sentinel watch ./my-ai-project
```

---

## TUI Debugger

```bash
vcr .vcr/my_run.vcr
```

```
┌──────────────────────────────────────────────────────────┐
│ 📼 Agent VCR                  Session: my_run · 8 frames │
├──────────────────────────────────────────────────────────┤
│ ▶ Frame 0  │ planner     │ 100ms  │ ●                    │
│   Frame 1  │ researcher  │  250ms │ ●                    │
│   Frame 2  │ coder       │  480ms │ ✗ ERROR              │
│   Frame 3  │ tester      │   80ms │ ●                    │
├──────────────────────────────────────────────────────────┤
│  { "query": "build a todo app", "plan": null }           │
├──────────────────────────────────────────────────────────┤
│ ←/→ navigate  │ e edit  │ d diff  │ r resume  │ q quit   │
└──────────────────────────────────────────────────────────┘
```

**Keybindings:** `↑/↓` or `j/k` navigate · `e` edit state · `1/2/3` input/output/diff · `r` resume · `s` search · `q` quit

Claude Code hooks:

```bash
vcr init --claude-code
```

---

## DAG Visualization

```bash
pip install "ai-agent-vcr[dashboard]"
vcr-server --vcr-dir .vcr --auth-token local-secret
# http://127.0.0.1:8000
```

```
original_run ──────────────────────────────────────────► [done]
               │ frame 3
               ╰──► fork_v1 ──► [coder] ──► [tester] ──► [done]
               ╰──► fork_v2 ──► [coder] ──► [done]
```

Live WebSocket streaming. Every fork is a branch. Errors in red.

---

## vs Everything Else

> **Honest take:** LangSmith, Langfuse, Arize Phoenix, and AgentOps are serious platforms with large teams. They are observability tools — they show you what happened. Agent VCR is an intervention tool — it lets you change what happened. Different category. The overlap is tracing. Everything else diverges.

| Capability | 📼 Agent VCR | LangSmith | LangFuse | AgentOps | Arize Phoenix |
|---|---|---|---|---|---|
| Record execution traces | ✅ | ✅ | ✅ | ✅ | ✅ |
| Production dashboards | Local | ✅ best-in-class | ✅ | ✅ | ✅ |
| Eval / scoring pipelines | ❌ | ✅ | ✅ | ✅ | ✅ |
| Time-travel / session replay | ✅ | ❌ | ❌ | ✅ (view only) | ❌ |
| **Edit state & resume mid-chain** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **ACID filesystem rollback** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Ghost Replay (zero tokens)** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Sentinel (real-time code guard)** | **✅** | ❌ | ❌ | ❌ | ❌ |
| **Fork from any frame** | **✅** | ❌ | ❌ | ❌ | ❌ |
| TUI debugger | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fully local / self-hosted | ✅ | ❌ Cloud | ✅ | ❌ Cloud | ✅ |
| Framework-agnostic | ✅ | ⚠️ LangChain | ✅ | ✅ | ✅ |

**AgentOps** — closest competitor on time-travel. It lets you view past sessions. It does not let you edit state and resume, fork a session, rollback the filesystem, or replay for zero tokens. If you need view-only replay, AgentOps is mature. If you need to actually intervene, you need Agent VCR.

**Use LangSmith/Langfuse/Phoenix/AgentOps** for production tracing and evals. **Use Agent VCR** when you need to actually fix a broken run without re-running it, rollback filesystem damage, or replay a successful run for free.

---

## vs LangGraph's Built-In Checkpointer

LangGraph's checkpointer is solid if you're 100% LangGraph and only need state inspection.

The gap: when your agent writes files to disk and fails, the checkpointer rolls back the state object. **The files stay.** Agent VCR runs `git reset --hard`. The files are gone.

| | LangGraph Checkpointer | Agent VCR |
|---|---|---|
| Checkpoint in-memory state | ✅ | ✅ |
| **Rollback files from disk** | ❌ | **✅** |
| **Ghost Replay (zero tokens)** | ❌ | **✅** |
| **Sentinel (code guardian)** | ❌ | **✅** |
| Works with CrewAI, raw Python | ❌ | ✅ |
| JSONL format (git-diffable) | ❌ | ✅ |
| Session forking | ❌ | ✅ |

---

## Performance

Every benchmark is enforced in CI. If it regresses, CI fails.

```bash
pip install -e ".[dev]"
pytest tests/benchmarks/ -v --benchmark-only
```

| Benchmark | Limit | What it measures |
|---|---|---|
| `test_benchmark_recorder_overhead` | **< 5ms mean** | Serialize + buffer one state snapshot |
| `test_benchmark_file_write_speed` | **> 1,000 frames/sec** | Sustained write throughput (10K frames) |
| `test_benchmark_load_speed` | **< 500ms** | Load a 10,000-frame session from disk |
| `test_benchmark_goto_frame` | **< 1ms** | Random-access time-travel to any frame |

Historical results: [ixchio.github.io/agent-vcr/dev/bench/](https://ixchio.github.io/agent-vcr/dev/bench/)

---

## Storage Format

Plain JSONL. One object per line.

```jsonl
{"type": "session", "data": {"session_id": "my_run", "created_at": "..."}}
{"type": "frame", "data": {"node_name": "planner", "input_state": {...}, "output_state": {...}}}
{"type": "frame", "data": {"node_name": "coder", ...}}
```

- **Human-readable** — open in any text editor
- **Git-diffable** — review agent state in PRs
- **Append-only** — safe for concurrent agents, no full-file rewrites
- **Streamable** — parse line-by-line without loading the full file

---

## API Reference

### `VCRRecorder`

```python
recorder = VCRRecorder(
    output_dir=".vcr",
    auto_save=True,
    diff_mode=False,
)

recorder.start_session(session_id="my_run", tags=["prod"])
recorder.record_step(node_name, input_state, output_state, metadata)
recorder.record_llm_call(model, messages, response, tokens_input, tokens_output, latency_ms)
recorder.record_tool_call(tool_name, tool_input, tool_output, latency_ms)
recorder.record_error(node_name, input_state, error)
recorder.save() -> Path
recorder.fork(from_frame=3) -> VCRRecorder
```

### `VCRPlayer`

```python
player = VCRPlayer.load(".vcr/my_run.vcr")

player.goto_frame(index)           # → output state at frame N
player.get_input_state(index)      # → input state at frame N
player.get_errors()                # → [Frame, ...]
player.compare_frames(a, b)        # → {'added': {}, 'removed': {}, 'modified': {}}
player.get_total_cost()            # → float (USD)

player.resume(
    agent_callable,
    config=ResumeConfig(
        from_frame=7,
        state_overrides={"k": "v"},
        mode=ResumeMode.FORK,      # FORK | REPLAY | MOCK
    )
)
```

### `ACIDWorkspace`

```python
acid = ACIDWorkspace("/workspace", recorder=recorder)
acid.begin(session_id="task-001")
acid.savepoint(state, node_name="coder")
acid.rollback(to_frame_index=2)    # git reset --hard
acid.commit()
```

### `GoldenRunCache`

```python
cache = GoldenRunCache(cache_dir=".vcr/golden")
identity = GoldenRunCache.build_identity(model="gpt-4o", code_commit="abc123")
cache.save_golden_run(task_description, recorder, identity=identity)
outputs, ledger = cache.replay(task_description, identity=identity)
cache.invalidate(task_description)
cache.list_golden_runs()
```

---

## Examples

```bash
# ACID rollback + Ghost Replay — start here
python examples/acid_golden_run.py

# Time-travel: rewind, edit state, resume
python examples/time_travel_demo.py

# Sentinel: watch agent self-correct in real time
python examples/sentinel_demo.py

# LangGraph auto-instrumentation
python examples/langgraph_integration.py

# Basic recording and playback
python examples/basic_usage.py
```

---

## Roadmap

- [x] Core recording and playback
- [x] Time-travel resume with state injection
- [x] LangGraph + CrewAI integrations
- [x] Async recorder and player
- [x] Terminal TUI debugger (`vcr`)
- [x] Claude Code hook scaffolding (`vcr init --claude-code`)
- [x] Live dashboard with DAG visualization
- [x] ACID Transactions (git-backed filesystem rollback)
- [x] Ghost Replay (zero-cost replay of successful runs)
- [x] Sentinel — real-time code quality guardian
- [x] Context manager (`with VCRRecorder() as r:`)
- [ ] Claude Code / Cursor integration
- [ ] AutoGen integration
- [ ] Replay regression tests (golden paths as CI assertions)
- [ ] Collaborative debugging (share sessions)
- [ ] Cloud storage backend (S3, GCS)

---

## Community

If Agent VCR saved your repo from a bad autonomous run, share it:

- **OpenHands** — [Discord](https://discord.gg/openhands) `#tools` channel
- **LangGraph** — [Discord](https://discord.gg/langchain) `#community` channel
- **r/LocalLLaMA** — post your ACID rollback story
- **Hacker News** — Show HN posts with real before/after diffs get traction

The best growth comes from developers sharing the moment it saved them. If that's you, a post with your actual corrupted-repo story (even anonymized) is worth more than any ad.

---

## Contributing

```bash
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr
pip install -e ".[dev,tui]"
make verify
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

### 📼

**Every AI agent developer has had a bad run trash their codebase.**
**Agent VCR is the undo button.**

<br>

```bash
pip install ai-agent-vcr
```

<br>

<a href="https://github.com/ixchio/agent-vcr">⭐ Star on GitHub</a> · <a href="https://pypi.org/project/ai-agent-vcr/">📦 PyPI</a> · <a href="https://ixchio.github.io/agent-vcr/">📖 Docs</a>

<br>

<sub>Built by <a href="https://github.com/ixchio">ixchio</a> · MIT License</sub>

</div>
