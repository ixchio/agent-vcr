# Framework and platform integrations

Agent VCR's stable integration boundary is a serializable state transition. If a framework exposes
node, tool, model, error, or lifecycle callbacks, it can write the same JSONL format without a
custom storage backend.

```bash
pip install ai-agent-vcr
pip install "ai-agent-vcr[langgraph]"
pip install "ai-agent-vcr[crewai]"
pip install "ai-agent-vcr[openhands]"
```

## Raw Python

```python
from agent_vcr import VCRRecorder

recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session("custom-agent")
before = {"task": "classify incident"}
after = classify(before)
recorder.record_step("classify", before, after)
recorder.save()
```

## LangGraph

Wrap nodes before compiling the graph:

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph

recorder = VCRRecorder()
recorder.start_session("langgraph-run")
instrumentation = VCRLangGraph(recorder)

graph.add_node("planner", instrumentation.wrap_node("planner", planner))
graph.add_node("executor", instrumentation.wrap_node("executor", executor))
compiled = graph.compile()
result = compiled.invoke({"task": "repair the API"})
recorder.save()
```

Agent VCR does not replace a LangGraph checkpointer. Use the checkpointer for graph-native
persistence and Agent VCR for portable recordings, Ghost Replay, Agent CI, or filesystem rollback.

## CrewAI

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI

recorder = VCRRecorder()
recorder.start_session("research-crew")
result = VCRCrewAI(recorder).kickoff(
    crew,
    inputs={"topic": "content-aware rollback"},
)
recorder.save()
```

The wrapper installs step/task callbacks for the `kickoff()` call and restores prior callbacks
afterward. Review recordings when upgrading CrewAI because callback payloads belong to that
framework and can change.

## OpenHands and repository-writing agents

`ACIDWorkspace` gives any repository-writing agent tracked-file rollback plus private escrow for
eligible pre-existing untracked/ignored files. `openhands_sentinel.Sentinel` can attach to an
OpenHands event stream and record AST findings after file writes. See the
[ACID guide](../acid-workspace.md) for the transaction boundary.

```bash
sentinel scan ./src
sentinel watch ./src
```

## Claude Code hooks

```bash
vcr init --claude-code
```

The command merges Agent VCR lifecycle commands into `.claude/settings.json`, preserves existing
hooks, and skips duplicate Agent VCR entries. Review project hook configuration before committing
it because hooks execute local commands.

## Build another adapter

Map node completion to `record_step`, model completion to `record_llm_call`, tools to
`record_tool_call`, exceptions to `record_error`, and durable boundaries to `record_checkpoint`.

Keep the framework's original return/exception behavior, redact before recording, measure rather
than guess metadata, avoid duplicate frames, and test sync, async, error, and cancellation paths.
The JSON schemas in [`../../schemas/`](../../schemas/) are the language-neutral contract.

For HTTP automation, see the [n8n guide](n8n.md).
