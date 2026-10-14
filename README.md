# Agent VCR

**Recovery, replay, and regression testing for AI agents that change real systems.**

Agent VCR records an agent run as durable JSONL frames, joins state checkpoints to Git-backed
filesystem savepoints, and gives you a controlled path from failure to recovery. Inspect the exact
state, rewind, resume from a checkpoint, roll back files, or replay a verified run without another
model call.

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ai-agent-vcr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ai-agent-vcr)
[![CI](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

```bash
pip install ai-agent-vcr
```

No model key, hosted account, or framework is required for the core runtime.

![Agent VCR terminal demo](docs/demo.gif)

## Why this exists

An agent can update its state and still leave your repository broken. It can write half a feature,
create files that never belonged there, spend tokens repeating successful work, or fail on step 18
after the first 17 steps already mutated the workspace.

Logs and traces explain what happened. Agent VCR is the intervention layer:

```text
run agent
   │
   ├── record state, tool calls, cost, latency, and errors
   ├── checkpoint state + filesystem together
   │
failure
   │
   ├── inspect or diff any frame
   ├── resume or fork from the last good state
   ├── roll the Git workspace back to a savepoint
   └── keep the verified run as a replay or CI baseline
```

The recording format is append-friendly JSONL. You can open it in an editor, diff it in Git,
stream it, bundle it, verify its integrity, and use it without sending execution data to a vendor.

## See it work in one minute

```python
from agent_vcr import VCRPlayer, VCRRecorder


def planner(state: dict) -> dict:
    return {**state, "plan": "add the migration before changing the API"}


recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session("billing-fix")

before = {"task": "add invoice status"}
after = planner(before)
recorder.record_step("planner", before, after)
recorder.record_checkpoint(after, label="plan-approved")
recording = recorder.save()

player = VCRPlayer.load(recording)
print(player.goto_frame(0))
print(player.goto_checkpoint("plan-approved"))
print(player.verify_integrity())
```

Or run the included credential-free examples:

```bash
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,tui,dashboard]"

python examples/basic_usage.py
python examples/time_travel_demo.py
python examples/acid_golden_run.py
```

## What Agent VCR can do

### Record and time-travel through a run

Capture state transitions, model and tool metadata, errors, costs, and latency. Load a recording,
inspect any frame, compare frames, edit state, and resume or fork from a chosen point. Sync and
async recorders and players use the same execution schema.

### Roll back state and files together

`ACIDWorkspace` uses isolated Git branches and commits as savepoints. A rollback restores the
recorded agent state and tracked filesystem state while preserving files that existed before the
transaction.

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

recorder = VCRRecorder()
workspace = ACIDWorkspace("/path/to/clean/repository", recorder=recorder)

workspace.begin(session_id="issue-42")
workspace.savepoint({"phase": "planned"}, node_name="planner")

# Let the agent work inside the isolated branch.

workspace.rollback(to_frame_index=0)  # restore the selected savepoint
workspace.commit()                    # merge only after verification
```

By default, Agent VCR refuses to begin from a dirty tracked worktree. Filesystem recovery is
powerful, so the integration makes the unsafe case explicit instead of guessing.

### Replay known-good work with zero model calls

Ghost Replay saves a verified execution under an identity that can include the model, prompt hash,
code commit, tool schema, dependency lock, and environment. A full identity match returns recorded
outputs without invoking the model. If relevant inputs drift, the cache does not silently call the
old run valid.

Replay modes cover full replay, hybrid replay, live execution, and branching. The cost ledger shows
which steps were replayed, which were rerun, and what was saved.

### Gate agent changes in CI

Turn known-good recordings into versioned replay datasets, evaluate candidate runs, and fail a pull
request on behavioral, cost, latency, or token regressions.

```bash
vcr dataset create refunds
vcr dataset add refunds \
  --task "refund a duplicate payment" \
  --vcr .vcr/golden/refund-agent.vcr
vcr ci run \
  --dataset refunds \
  --candidates .vcr/candidates \
  --fail-on degraded \
  --report agent-ci-report.json \
  --markdown agent-ci-report.md
```

The repository includes a reusable composite GitHub Action in [action.yml](action.yml).

### Run durable workflows

The workflow engine executes dependency-aware DAGs with conditions, parallel fan-out and joins,
bounded `for_each` loops, retries, timeouts, approvals, checkpoints, and saga compensation. Run
state is persisted after each top-level step, so interrupted work can resume without re-executing
completed steps.

```bash
vcr workflow validate examples/hello_workflow.json
vcr workflow run \
  --file examples/hello_workflow.json \
  --handlers examples/workflow_handlers.py
vcr workflow runs --workflow hello-agent-vcr
```

### Put policy around agent execution

SuperAgent executes validated plans through a tool gateway with parameter checks, timeouts, budgets,
approval gates, verification, memory redaction, retries, and recorded checkpoints. Irreversible work
can be made approval-only; missing approval fails closed.

### Diagnose failures from recorded evidence

Reliability intelligence clusters related failures, scores risk, separates cost by model and node,
finds duplicate model work that could be replayed, suggests evidence-based model routes, and ranks
recovery actions. It is decision support: recovery does not mutate a run unless you explicitly ask
it to execute a safe action.

```bash
vcr intel why .vcr/<session-id>.vcr
vcr intel failures --dir .vcr
vcr intel risk --dir .vcr
vcr intel costs --dir .vcr
vcr intel recover .vcr/<session-id>.vcr
```

### Inspect runs locally or self-host the project store

The optional dashboard serves local recordings with live WebSocket updates and state diffs. The
self-hosted project store adds scoped project tokens, integrity-checked uploads, pull, search, and
server-side full replay.

```bash
pip install "ai-agent-vcr[dashboard]"
vcr-server --vcr-dir .vcr --auth-token local-dev-token
# open http://127.0.0.1:8000/dashboard
```

The repository also contains an optional deployment foundation for Supabase Auth, Stripe billing,
and Postgres. It is not presented as a finished multitenant SaaS: SSO, RBAC, tenant-isolated object
storage, managed secrets, and full production operations remain deployment responsibilities.

## Framework integrations

Agent VCR works with raw Python and includes adapters for LangGraph, CrewAI, OpenHands, and Claude
Code hooks.

```bash
pip install "ai-agent-vcr[langgraph]"
pip install "ai-agent-vcr[crewai]"
pip install "ai-agent-vcr[openhands]"
```

```python
# LangGraph
from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph

recorder = VCRRecorder()
instrumented_graph = VCRLangGraph(recorder).wrap_graph(graph)
```

```python
# CrewAI
from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI

recorder = VCRRecorder()
recorder.start_session("research-crew")
result = VCRCrewAI(recorder).kickoff(crew)
recorder.save()
```

For Claude Code hook scaffolding:

```bash
vcr init --claude-code
```

## Build integrations on Agent VCR

You do not need an official adapter to connect another framework. The recorder is the integration
boundary: call `record_step`, `record_llm_call`, `record_tool_call`, `record_error`, or
`record_checkpoint` from your framework's lifecycle hooks.

```python
from agent_vcr import VCRRecorder


def forward_frame(frame) -> None:
    # Send this payload to your event bus, debugger, or internal dashboard.
    print(frame.model_dump_json())


recorder = VCRRecorder(on_frame_recorded=forward_frame)
recorder.start_session("custom-framework-run")

before = {"request": "ship the integration"}
after = your_agent_step(before)
recorder.record_step("custom-agent", before, after)
recorder.save()
```

The extension surfaces are deliberately plain:

- **Python API:** instrument raw functions, framework callbacks, model calls, and tools.
- **Frame callback:** forward each recorded frame to an event bus, UI, or telemetry pipeline.
- **JSONL and JSON Schema:** consume recordings in another language without importing Python.
- **Replay bundles:** exchange portable `.vcrb` files with an identity manifest and SHA-256 checks.
- **HTTP and WebSocket server:** build a custom dashboard or synchronize sessions remotely.
- **CLI and GitHub Action:** call Agent VCR from shell scripts, CI systems, or release gates.
- **Pluggable execution:** register SuperAgent tool handlers, workflow handler modules, or a custom
  sandbox behind the published interfaces.

Integration guides and templates live in [docs/integrations](docs/integrations). If your framework
can expose a step boundary and serializable state, it can connect to Agent VCR without changing the
recording format.

## Install only what you need

```bash
pip install ai-agent-vcr                  # recorder, player, replay, workflows, CI
pip install "ai-agent-vcr[tui]"          # terminal debugger
pip install "ai-agent-vcr[dashboard]"    # local dashboard and server
pip install "ai-agent-vcr[sandbox]"      # E2B sandbox adapter
pip install "ai-agent-vcr[llm]"          # LiteLLM planner
pip install "ai-agent-vcr[vector]"       # Qdrant-backed vector memory
pip install "ai-agent-vcr[all]"          # all optional integrations
```

Python 3.9–3.13 is tested in CI.

## Where it fits

Agent VCR is not a replacement for LangSmith, Langfuse, Phoenix, or AgentOps. Those products focus
on observing, evaluating, and operating model calls. Agent VCR focuses on intervention after an
agent changes state or files: checkpoint, rewind, resume, roll back, replay, and regression-gate.
They can be used together.

It also does not make arbitrary external side effects transactional. Git can restore a workspace;
it cannot undo an email, payment, or third-party API call. Put irreversible tools behind approval
and compensation policies.

## Verification

The release gate runs linting, strict typing, unit tests, end-to-end tests, performance limits,
package builds, and distribution validation:

```bash
python -m pip install -e ".[dev,tui,dashboard]"
make verify
```

Benchmark history is published at [ixchio.github.io/agent-vcr/dev/bench](https://ixchio.github.io/agent-vcr/dev/bench/).
The execution, bundle, dataset, workflow, cloud, intelligence, and SuperAgent contracts are exported
as versioned JSON Schemas under [schemas](schemas).

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and data contracts](docs/architecture.md)
- [Workflow engine](docs/workflow.md)
- [SuperAgent runtime](docs/superagent.md)
- [Development and release verification](docs/development.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Contributing

Issues and pull requests are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), run
`make verify`, and include a focused test for behavior changes.

## License

Agent VCR is open-source software released under the [MIT License](LICENSE).

<sub>Copyright © 2026 GOD and <a href="https://github.com/ixchio">ixchio</a>.</sub>
