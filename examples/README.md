# Examples

Every example runs without a model key or account. From the repository root:

```bash
pip install -e ".[dev,tui]"
python examples/basic_usage.py
```

| Example | What it shows | Needs |
|---|---|---|
| [`airbag_demo.sh`](airbag_demo.sh) | An agent `rm -rf`s the project; `vcr undo` brings back code, `.env` and git history | `bash`, `git` |
| [`basic_usage.py`](basic_usage.py) | Record a run, save it, load it with the player | — |
| [`time_travel_demo.py`](time_travel_demo.py) | Jump to frames, diff them, fork and resume from a step | — |
| [`async_example.py`](async_example.py) | The async recorder and player | — |
| [`acid_golden_run.py`](acid_golden_run.py) | `ACIDWorkspace` savepoints + rollback, then Ghost Replay of the verified run | `git` |
| [`showcase_demo.py`](showcase_demo.py) | A guided tour of recording, rollback and replay | `git` |
| [`langgraph_integration.py`](langgraph_integration.py) | `VCRLangGraph.wrap_graph()` on a real `StateGraph` | `pip install langgraph` |
| [`crewai_integration.py`](crewai_integration.py) | `VCRCrewAI` and the manual `VCRCrewCallback` API | — (CrewAI optional) |
| [`sentinel_demo.py`](sentinel_demo.py) | Sentinel catching syntax errors, duplicates and complexity | — |
| [`agent_ci_candidates.py`](agent_ci_candidates.py) + [`agent-ci.yml`](agent-ci.yml) | Produce candidate recordings and gate them in GitHub Actions | `--dataset NAME` |
| [`hello_workflow.json`](hello_workflow.json) + [`workflow_handlers.py`](workflow_handlers.py) | A workflow run with `vcr workflow run` (experimental) | — |

```bash
vcr workflow run --file examples/hello_workflow.json --handlers examples/workflow_handlers.py
```
