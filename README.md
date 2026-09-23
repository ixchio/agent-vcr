# Agent VCR

**The recovery layer for agents that touch real code and systems.**

Agent VCR records agent state, tool calls, and checkpoints so a failed run becomes an artifact you can inspect, rewind, replay, compare, or roll back—instead of a reason to start again from zero.

[![CI](https://img.shields.io/github/actions/workflow/status/ixchio/agent-vcr/ci.yml?style=flat-square&label=CI)](https://github.com/ixchio/agent-vcr/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-proprietary-red?style=flat-square)](LICENSE)

```bash
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tui,dashboard]"

vcr init
python examples/basic_usage.py
```

No model key is required for the local examples. This repository is proprietary software;
access, use, and distribution are governed by [LICENSE](LICENSE).

## Why Agent VCR

Agents make changes before a human can review them. When a run fails after writing files, calling tools, or modifying state, logs tell you what happened—but they do not restore a usable point in time.

Agent VCR gives the operator a recovery loop:

```text
record a run  ->  inspect a frame  ->  recover or roll back  ->  keep the good run as a golden baseline
     |                                                                           |
     +------------------- gate future candidate runs in CI --------------------+
```

The runtime is local-first. Recordings are append-only JSONL (`.vcr`), readable in a text editor, diffable in Git, and usable without a hosted account.

## What it does

| Need | Agent VCR capability |
| --- | --- |
| An agent failed after changing files | Git-backed savepoints and filesystem rollback with `ACIDWorkspace` |
| A run failed late | Time-travel to any recorded frame, edit state, then fork or resume |
| A known task repeats | Ghost Replay returns a verified golden run with zero model calls |
| A regression reaches a pull request | Replay datasets, regression reports, and a GitHub Action |
| A multi-step process dies midway | Durable workflows with checkpoints, retries, compensation, schedules, and resume |
| You need evidence across runs | Failure clustering, risk scoring, recovery planning, model routing, and cost analysis |
| An agent needs isolation | Local and E2B sandbox adapters with a common interface |
| You need framework coverage | Raw Python, LangGraph, CrewAI, OpenHands, and Claude Code hooks |

## Start locally

### Record and inspect a run

```bash
vcr init
python examples/basic_usage.py

# Replace this with the path printed by the example.
vcr tui .vcr/<session-id>.vcr
```

Minimal Python integration:

```python
from agent_vcr import ResumeConfig, VCRPlayer, VCRRecorder


def planner(state: dict) -> dict:
    return {**state, "plan": "add an idempotent migration"}


recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session()
before = {"request": "add invoices"}
recorder.record_step("planner", before, planner(before))
recording = recorder.save()

player = VCRPlayer.load(recording)
player.resume(
    planner,
    ResumeConfig(
        from_frame=0,
        state_overrides={"request": "add invoices with a migration"},
    ),
)
```

### Roll back a coding agent safely

`ACIDWorkspace` creates an isolated Git branch, saves Git-backed checkpoints, and can restore both agent state and tracked files to a selected savepoint.

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

recorder = VCRRecorder(output_dir=".vcr")
workspace = ACIDWorkspace("/path/to/repository", recorder=recorder)

workspace.begin(session_id="issue-42")
workspace.savepoint({"task": "implement billing"}, node_name="planner")

# Run the coding agent in the isolated workspace.
workspace.rollback(to_frame_index=0)
workspace.commit()  # Merge only after verification.
```

### Run a durable workflow

The included workflow needs no model provider. It validates the workflow, records tool activity, creates checkpoints, and writes a resumable run record.

```bash
vcr workflow run \
  --file examples/hello_workflow.json \
  --handlers examples/workflow_handlers.py

vcr workflow runs --workflow hello-agent-vcr
vcr workflow resume <run-id> --handlers examples/workflow_handlers.py
```

Read the workflow contract, approval points, retries, compensation actions, schedules, and recovery behavior in [docs/workflow.md](docs/workflow.md).

## Ship reliable agent changes

### Replay bundles and Ghost Replay

Export a recording as a portable, verified bundle before using it elsewhere:

```bash
vcr bundle export .vcr/<session-id>.vcr --output support-fix.vcrb
vcr bundle verify support-fix.vcrb
vcr bundle import support-fix.vcrb --dest-dir .vcr/imported
vcr reproduce support-fix.vcrb
```

For repeated tasks, save a verified run to `GoldenRunCache` with a model, prompt, code, and tool-schema identity. A matching replay returns the recorded outputs without invoking a model. When that identity changes, the cache does not treat the old run as valid.

### Regression gates for agent behavior

Create a replay dataset from known-good runs, then compare candidate recordings in CI:

```bash
vcr ci run \
  --dataset refunds \
  --candidates .vcr/candidates \
  --datasets-dir .vcr/datasets \
  --golden-dir .vcr/golden \
  --report agent-ci-report.json \
  --markdown agent-ci-report.md
```

The repository includes a composite GitHub Action in [action.yml](action.yml). It produces a machine-readable and Markdown report, and can fail a pull request at the configured regression threshold.

### Diagnose and recover from evidence

```bash
vcr intel why .vcr/<session-id>.vcr
vcr intel failures --dir .vcr
vcr intel risk --dir .vcr
vcr intel costs --dir .vcr
vcr intel recover .vcr/<session-id>.vcr
```

These commands turn recordings into failure clusters, risk signals, cost breakdowns, model-routing suggestions, and recovery plans. They are decision support for an operator; they do not silently mutate a run or repository.

## Integrations and execution boundaries

```bash
# Optional framework extras
python -m pip install "ai-agent-vcr[langgraph]"
python -m pip install "ai-agent-vcr[crewai]"
python -m pip install "ai-agent-vcr[openhands]"
python -m pip install "ai-agent-vcr[sandbox]"  # E2B adapter
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
result = VCRCrewAI(recorder).kickoff(crew)
```

For Claude Code, run `vcr init --claude-code` inside the target repository.

## Local dashboard and project store

Run the dashboard and self-hosted project store on your machine:

```bash
vcr-server --vcr-dir .vcr --auth-token local-dev-token
# Open http://127.0.0.1:8000/dashboard
```

The `vcr cloud` commands support login, project/session listing, push, pull, and server-side Ghost Replay against a configured Agent VCR server. It is a self-hosted project store, not a public multitenant SaaS. Keep it behind a project or administrator token until tenant-scoped storage and per-resource authorization are implemented.

## Hosted control-plane status

There is an optional Railway deployment foundation for a future managed offering:

- Supabase Auth token verification through the provider JWKS; no service-role key is required.
- Stripe-hosted subscription checkout and billing portal.
- Signed Stripe webhook handling and an idempotent Postgres billing ledger.
- Railway Docker configuration, health check, migration command, and persistent-volume guidance.

It intentionally does **not** publish local recordings as a shared cloud product. SSO, RBAC, tenant-isolated object storage, autoscaling, managed secrets, and full production operations are not shipped yet. See [the Railway deployment guide](docs/deployment/railway.md) before enabling it.

## Documentation

| Area | Documentation |
| --- | --- |
| Documentation index | [docs/README.md](docs/README.md) |
| Architecture and data contracts | [docs/architecture.md](docs/architecture.md) |
| Workflows | [docs/workflow.md](docs/workflow.md) |
| Development and verification | [docs/development.md](docs/development.md) |
| Railway hosted control plane | [docs/deployment/railway.md](docs/deployment/railway.md) |
| Integration templates | [docs/integrations](docs/integrations) |

Schemas live in [schemas](schemas). The source package is organized around the recorder/player, replay bundles, workflow engine, SuperAgent, intelligence, cloud store, and sandbox adapters.

## Verify a checkout

```bash
python -m pip install -e ".[dev,tui,dashboard]"
make verify
make bench
```

`make verify` runs linting, typing, unit tests, package build checks, and distribution validation. Benchmark limits run separately through `make bench`.

## Contributing

```bash
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,tui,dashboard]"
make verify
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md) before opening an issue or pull request.

## License

Proprietary Software License — see [LICENSE](LICENSE).

<sub>Copyright © 2026 QueryXLabs and <a href="https://github.com/ixchio">ixchio</a>. All rights reserved.</sub>
