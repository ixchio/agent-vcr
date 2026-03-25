# Agent VCR

[![CI](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml/badge.svg)](https://github.com/ixchio/agent-vcr/actions)
[![Benchmarks](https://img.shields.io/badge/benchmarks-view%20chart-blue)](https://ixchio.github.io/agent-vcr/dev/bench/)
[![codecov](https://codecov.io/gh/ixchio/agent-vcr/branch/main/graph/badge.svg)](https://codecov.io/gh/ixchio/agent-vcr)
[![PyPI version](https://badge.fury.io/py/ai-agent-vcr.svg)](https://badge.fury.io/py/ai-agent-vcr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time-travel debugging for AI agents.**

[Documentation](https://ixchio.github.io/agent-vcr/) · [Examples](#examples)

---

## The Problem

Building multi-step AI agents (LangGraph, CrewAI, OpenHands) is painfully slow to debug.

When your agent fails on step 8 out of 10, observability tools like LangSmith or LangFuse only show you what went wrong. To fix it, you patch the code and re-run all 10 steps from scratch. Every typo costs minutes of wall time and dollars in wasted tokens.

## The Solution

Agent VCR records your agent's complete state at every step. When something breaks, you rewind to the failing step, edit the state, and resume execution from that exact point. No re-running the whole chain.

LangSmith shows you what happened. **Agent VCR lets you change it.**

---

## Quick Start

```bash
pip install ai-agent-vcr
```

```python
from agent_vcr import VCRRecorder, VCRPlayer

# Record your agent
recorder = VCRRecorder()
recorder.start_session("bug_hunt")
# ... your agent code ...
recorder.save()

# Time-travel and fix
player = VCRPlayer.load(".vcr/bug_hunt.vcr")
state = player.goto_frame(2)      # jump to step 2
state["prompt"] = "Fixed prompt"   # fix the state
player.resume(from_frame=2)        # continue from there
```

---

## What It Does

- **Time Travel** — Jump back to any step. Full state snapshot at every node.
- **State Injection & Resume** — Edit the state at any frame — fix a prompt, patch tool output, inject context — then resume mid-chain.
- **ACID Transactions** — Wrap agent execution in real database-style transactions backed by git. Rollback physically reverts files on disk, not just in-memory state.
- **Golden Run Cache** — Save successful runs as replayable paths. Next time you hit the same task, skip all LLM calls. Same task, zero tokens, instant.
- **React Dashboard** — Run `vcr-server`, open `localhost:8000`. Glassmorphism UI for inspecting state, viewing JSON diffs, live WebSocket streaming.
- **TUI Debugger** — Run `vcr-tui` in your terminal. Navigate frames, press `e` to edit state, press `r` to resume.
- **Visual Diffs** — Color-coded state mutation tracking in Dashboard and TUI.
- **DAG Visualization** — See parallel execution branches, search/filter sessions by tags.
- **Framework Agnostic** — 1-line integration with LangGraph, CrewAI, or raw Python.
- **Git-Friendly Storage** — JSONL files, version controllable, append-only.
- **Production Safe** — `<5ms` overhead per frame. Async-native.

---

## ACID Transactions

Databases solved the partial failure problem 40 years ago with transactions. Agents have the exact same problem — when your agent fails mid-run, you don't just have bad in-memory state. You have files written to disk, commits made, half a codebase that shouldn't exist. Current tools only roll back the state object. The filesystem stays polluted.

Agent VCR wraps agent execution in real transactional semantics:

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

recorder = VCRRecorder()
acid = ACIDWorkspace("/my/workspace", recorder=recorder)

acid.begin(session_id="task-001")        # snapshot workspace, isolated branch
acid.savepoint(state, node_name="coder") # checkpoint filesystem + state together
acid.rollback(to_frame_index=3)          # git reset --hard to savepoint
acid.commit()                            # merge clean branch into main
```

- **BEGIN** creates an isolated git branch. Parallel agents can't interfere with each other.
- **SAVEPOINT** checkpoints both the VCR state and the filesystem. Every frame has a matching git commit.
- **ROLLBACK** runs `git reset --hard` to a previous savepoint. Files your agent hallucinated are gone from disk — not hidden, deleted.
- **COMMIT** merges the successful branch back into main.

---

## Golden Run Cache

When your agent succeeds, save the entire execution as a golden path. Next time you run the same task, replay the cached outputs directly — skipping every LLM call. Only re-run steps whose inputs actually changed.

```python
from agent_vcr.golden_cache import GoldenRunCache

cache = GoldenRunCache()

# After a successful run
cache.save_golden_run("Build a REST API with JWT auth", recorder)

# Next time — instant, zero cost
outputs, ledger = cache.replay("Build a REST API with JWT auth")
print(ledger)  # CostLedger(saved=100% | $0.0123 | 4100 tokens | 2349ms)
```

The `CostLedger` tracks original vs replay tokens, dollars saved, latency saved, and percentage reduction.

---

## Who Is This For?

| If you are... | Agent VCR helps you... |
| --- | --- |
| An AI engineer debugging LangGraph agents | Rewind to the exact failing step, fix state, resume |
| A team lead reviewing agent behavior | Compare execution paths side-by-side with full state diffs |
| A researcher iterating on prompts | Fork from any step, change the prompt, see how downstream behavior changes |
| Building production agents | Record every run in JSONL for audit trails and regression testing |
| Running agents at scale | Cache successful runs and replay them at zero cost |

---

## Comparison

| Feature | Agent VCR | LangSmith | LangFuse | Arize Phoenix |
| --- | --- | --- | --- | --- |
| Record execution traces | Yes | Yes | Yes | Yes |
| Time-travel to any step | Yes | No | No | No |
| Edit state and resume | **Yes** | No | No | No |
| Fork from any frame | Yes | No | No | No |
| ACID transactions (filesystem rollback) | **Yes** | No | No | No |
| Golden Run Cache (zero-cost replay) | **Yes** | No | No | No |
| Compare execution runs | Yes | Yes | Partial | Partial |
| Self-hosted, local-first | Yes | No (cloud) | Yes | Yes |
| Git-friendly format (JSONL) | Yes | No | No | No |
| Framework agnostic | Yes | LangChain only | Yes | Yes |
| Zero external deps | Yes | Cloud required | Cloud required | Yes |
| Setup lines | **3** | ~15 | ~10 | ~10 |

---

## Integrations

### LangGraph

```python
from langgraph.graph import StateGraph
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph

graph = StateGraph()
graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_edge("planner", "coder")

# One line to add recording
recorder = VCRRecorder()
graph = VCRLangGraph(recorder).wrap_graph(graph)

result = graph.invoke({"query": "Build a todo app"})
```

### Raw Python

```python
from agent_vcr.integrations.langgraph import vcr_record

recorder = VCRRecorder()

@vcr_record(recorder, node_name="my_function")
def my_function(data):
    return process(data)

result = my_function({"key": "value"})
```

### CrewAI

Agent VCR hooks into CrewAI's `step_callback` and `task_callback` for automatic frame capture.

```python
from crewai import Crew, Agent, Task
from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI, vcr_task

recorder = VCRRecorder()
recorder.start_session("crew_debug_run")

# Wrap the whole crew — records every thought, tool call, and task
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
vcr_crew = VCRCrewAI(recorder)
result = vcr_crew.kickoff(crew)
recorder.save()

# Or decorate individual functions
@vcr_task(recorder, task_name="research_step")
def research(context: dict) -> str:
    return "findings..."
```

Install with:

```bash
pip install "ai-agent-vcr[crewai]"
```

See [`examples/crewai_integration.py`](examples/crewai_integration.py) for a full runnable demo.

---

## Storage Format

Agent VCR uses JSONL (JSON Lines):

```jsonl
{"type": "session", "data": {"session_id": "abc123", "created_at": "2024-01-01T00:00:00Z", ...}}
{"type": "frame", "data": {"frame_id": "...", "node_name": "planner", "input_state": {...}, "output_state": {...}, ...}}
{"type": "frame", "data": {...}}
```

- Human-readable
- Git-diffable
- Append-only (efficient for streaming)
- Line-by-line parsing (no need to load the entire file)

---

## Performance

Recording overhead is continuously benchmarked in CI to stay under 5ms per frame.

```bash
pytest tests/benchmarks/ -v
```

---

## API Reference

### VCRRecorder

```python
class VCRRecorder:
def start_session(
    self,
    session_id: str = None,
    parent_session_id: str = None,
    forked_from_frame: int = None,
    metadata: dict = None,
    tags: list[str] = None,
) -> Session

def record_step(
    self,
    node_name: str,
    input_state: dict,
    output_state: dict,
    metadata: FrameMetadata = None,
    frame_type: FrameType = FrameType.NODE_EXECUTION,
) -> Frame

def record_llm_call(...)
def record_tool_call(...)
def record_error(...)
def save(self) -> Path
def fork(self, from_frame: int, ...) -> VCRRecorder
```

### VCRPlayer

```python
class VCRPlayer:
@classmethod
def load(cls, filepath: str) -> VCRPlayer

def goto_frame(self, index: int) -> dict
def get_frame(self, index: int) -> Frame
def list_nodes(self) -> list[str]
def get_errors(self) -> list[Frame]
def compare_frames(self, a: int, b: int) -> dict
def resume(self, agent_callable: Callable, config: ResumeConfig) -> str
def export_state(self, frame_index: int) -> dict
```

### ACIDWorkspace

```python
class ACIDWorkspace:
def __init__(self, workspace_path: str, recorder: VCRRecorder = None)
def begin(self, session_id: str) -> None
def savepoint(self, state: dict, node_name: str) -> None
def rollback(self, to_frame_index: int) -> None
def commit(self) -> None
```

### GoldenRunCache

```python
class GoldenRunCache:
def __init__(self, cache_dir: str = ".vcr/golden")
def save_golden_run(self, task: str, recorder: VCRRecorder) -> str
def replay(self, task: str) -> tuple[list[dict], CostLedger]
def invalidate(self, task: str) -> bool
```

### ResumeConfig

```python
class ResumeConfig:
from_frame: int              # Frame to resume from
new_session_id: str = None   # Optional ID for forked session
state_overrides: dict = {}   # State changes to apply
mode: ResumeMode = FORK      # FORK, REPLAY, or MOCK
skip_nodes: list[str] = []   # Nodes to skip during replay
inject_mocks: dict = {}      # Mock values for dependencies
```

---

## Examples

See the [`examples/`](examples/) directory:

- [`basic_usage.py`](examples/basic_usage.py) — Recording and playback
- [`time_travel_demo.py`](examples/time_travel_demo.py) — Full time-travel workflow
- [`langgraph_integration.py`](examples/langgraph_integration.py) — LangGraph auto-instrumentation
- [`acid_golden_run.py`](examples/acid_golden_run.py) — ACID transactions and Golden Run Cache

```bash
python examples/acid_golden_run.py
```

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/agent-vcr/agent-vcr.git
cd agent-vcr
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
pytest tests/benchmarks/ -v
pytest --cov=agent_vcr --cov-report=html
```

---

## Roadmap

- [x] Core recording and playback
- [x] Time-travel resume
- [x] FastAPI server with WebSocket
- [x] LangGraph integration
- [x] Async recorder and player
- [x] Terminal TUI debugger (`vcr-tui`)
- [x] CI/CD integrations
- [x] React dashboard
- [x] CrewAI integration
- [x] ACID Transactions (git-backed filesystem rollback)
- [x] Golden Run Cache (zero-cost replay of successful runs)
- [ ] AutoGen integration
- [ ] Cloud storage backend
- [ ] Collaborative debugging

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by:

- [LangSmith](https://smith.langchain.com/) — the observability paradigm
- [GDB](https://www.gnu.org/software/gdb/) — the time-travel debugging concept
- [Chrome DevTools](https://developer.chrome.com/docs/devtools/) — the UX patterns

---

<p align="center">
  Built by the Agent VCR community
</p>
