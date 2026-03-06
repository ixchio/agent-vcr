# 📼 Agent VCR

[![CI](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml/badge.svg)](https://github.com/ixchio/agent-vcr/actions)
[![codecov](https://codecov.io/gh/ixchio/agent-vcr/branch/main/graph/badge.svg)](https://codecov.io/gh/ixchio/agent-vcr)
[![PyPI version](https://badge.fury.io/py/agent-vcr.svg)](https://badge.fury.io/py/agent-vcr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Time-travel debugging for AI agents.**

[📖 Documentation](docs/index.html) • [🚀 Examples](#examples)

---

## 🛑 The Problem

Building multi-step AI agents (like LangGraph or CrewAI) is painfully slow.

When your agent fails on step 8 out of 10, traditional observability tools only tell you what went wrong. To fix it, you have to patch the prompt or code and **re-run all 10 steps from the beginning**.

Every typo or logic error costs you minutes of waiting and dollars in wasted LLM tokens.

## 💡 The Solution

**Agent VCR makes debugging instant.**

We record your agent's state at every step. When a failure happens, you simply **rewind** to the failing step, **edit** the state to fix the bug, and **resume** execution from that exact point.

LangSmith and LangFuse show you what happened. **Agent VCR lets you change it.**

- 🔌 **Plug & Play**: 1-line integration with LangGraph and others.
- 🚀 **Zero Overhead**: `<5ms` latency penalty per step.
- 📁 **No Vendor Lock-in**: Stores runs locally as git-friendly JSONL.
- 🔄 **Async Native**: Built from the ground up for modern `asyncio` agents.

---

## 🔥 Quick Start

```bash
pip install agent-vcr
```

```python
from agent_vcr import VCRRecorder, VCRPlayer

# 1. Record your agent (One-time setup)
recorder = VCRRecorder()
recorder.start_session("bug_hunt")
# ... your agent code runs here ...
recorder.save()

# 2. Time-Travel & Fix (The magic part)
player = VCRPlayer.load(".vcr/bug_hunt.vcr")

state = player.goto_frame(2)    # Jump back to step 2
state["prompt"] = "Fixed!"      # Fix the bad state
player.resume(from_frame=2)     # Resume execution from step 2
```

## Features

- 🔴 **Live Recording** — Watch your agent execute in real-time via WebSocket
- ⏮️ **Time Travel** — Jump to any step, inspect full state
- ✏️ **State Injection** — Edit state and resume execution
- 🌳 **DAG Visualization** — See parallel execution branches
- 🔌 **Framework Agnostic** — Works with LangGraph, CrewAI, or raw Python
- 📁 **Git-Friendly Format** — JSONL files, version controllable
- 🚀 **Production Performance** — <5ms overhead per frame
- 🔄 **Async-First** — Full async recorder and player support

---

## Who Is This For?

| If you are... | Agent VCR helps you... |
|---|---|
| **An AI engineer** debugging LangGraph agents | Rewind to the exact failing step, fix state, and resume — no re-running the whole chain |
| **A team lead** reviewing agent behavior | Compare two execution paths side-by-side with full state diffs |
| **A researcher** iterating on prompts | Fork from any step, change the prompt, and see how downstream behavior changes |
| **Building production agents** | Record every execution in JSONL for audit trails and regression testing |

---

## How Does It Compare?

| Feature | Agent VCR | LangSmith | LangFuse | Arize Phoenix |
|---|---|---|---|---|
| Record execution traces | ✅ | ✅ | ✅ | ✅ |
| Time-travel to any step | ✅ | ❌ | ❌ | ❌ |
| **Edit state & resume** | ✅ | ❌ | ❌ | ❌ |
| Fork from any frame | ✅ | ❌ | ❌ | ❌ |
| Compare execution runs | ✅ | ✅ | ⚠️ | ⚠️ |
| Self-hosted / local-first | ✅ | ❌ | ✅ | ✅ |
| Git-friendly format (JSONL) | ✅ | ❌ | ❌ | ❌ |
| Framework agnostic | ✅ | ⚠️ LangChain | ✅ | ✅ |
| Zero external dependencies | ✅ | ❌ Cloud | ❌ Cloud | ✅ |
| **Setup lines** | **3** | ~15 | ~10 | ~10 |

---

## Framework Integrations

### LangGraph

```python
from langgraph.graph import StateGraph
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph

# Your existing LangGraph code
graph = StateGraph()
graph.add_node("planner", planner_node)
graph.add_node("coder", coder_node)
graph.add_edge("planner", "coder")

# Add VCR recording with one line
recorder = VCRRecorder()
graph = VCRLangGraph(recorder).wrap_graph(graph)

# Run normally — recording happens automatically
result = graph.invoke({"query": "Build a todo app"})
```

### Raw Python

```python
from agent_vcr.integrations.langgraph import vcr_record

recorder = VCRRecorder()

@vcr_record(recorder, node_name="my_function")
def my_function(data):
return process(data)

# Each call is automatically recorded
result = my_function({"key": "value"})
```

---

## Storage Format

Agent VCR uses **JSONL** (JSON Lines) for storage:

```jsonl
{"type": "session", "data": {"session_id": "abc123", "created_at": "2024-01-01T00:00:00Z", ...}}
{"type": "frame", "data": {"frame_id": "...", "node_name": "planner", "input_state": {...}, "output_state": {...}, ...}}
{"type": "frame", "data": {...}}
```

Benefits:
- ✅ Human-readable
- ✅ Git-diffable
- ✅ Append-only (efficient for streaming)
- ✅ Line-by-line parsing (no need to load entire file)

---

## Performance

Performance is continuously benchmarked in CI to ensure `<5ms` recording overhead.

To run the reproducible benchmarks on your own hardware:
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

See the [`examples/`](examples/) directory for:

- [`basic_usage.py`](examples/basic_usage.py) — Simple recording and playback
- [`time_travel_demo.py`](examples/time_travel_demo.py) — Full time-travel workflow
- [`langgraph_integration.py`](examples/langgraph_integration.py) — LangGraph auto-instrumentation

Run an example:

```bash
python examples/time_travel_demo.py
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/agent-vcr/agent-vcr.git
cd agent-vcr
pip install -e ".[dev]"
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# Benchmarks
pytest tests/benchmarks/ -v

# With coverage
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
- [ ] React dashboard
- [ ] CrewAI integration
- [ ] AutoGen integration
- [ ] Cloud storage backend
- [ ] Collaborative debugging

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Inspired by:
- [LangSmith](https://smith.langchain.com/) — For the observability paradigm
- [GDB](https://www.gnu.org/software/gdb/) — For the time-travel debugging concept
- [Chrome DevTools](https://developer.chrome.com/docs/devtools/) — For the UX patterns

---

<p align="center">
  Built with ❤️ by the Agent VCR community
</p>
