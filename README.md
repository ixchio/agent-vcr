# Agent VCR

[![PyPI Version](https://img.shields.io/pypi/v/ai-agent-vcr?style=flat-square&label=PyPI)](https://pypi.org/project/ai-agent-vcr/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ai-agent-vcr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ai-agent-vcr)
[![Python Versions](https://img.shields.io/pypi/pyversions/ai-agent-vcr?style=flat-square)](https://pypi.org/project/ai-agent-vcr/)
[![CI](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/pypi/l/ai-agent-vcr?style=flat-square)](https://github.com/ixchio/agent-vcr/blob/main/LICENSE)

**The undo button for AI agents.** Undo anything your coding agent did — including `rm -rf`
from the shell — record and rewind agent runs, replay verified work without model calls, and
gate agent regressions in CI.

```bash
pip install ai-agent-vcr
vcr airbag install        # hooks for every coding agent on your PATH
vcr undo                  # the agent deleted something? bring it back
```

No model key, account, or framework required. Everything stays on your machine.

![Agent VCR recording, rollback, and replay demo](https://raw.githubusercontent.com/ixchio/agent-vcr/main/docs/demo.gif)

## Why

Coding agents run shell commands. The built-in "rewind" in agent tools only tracks the agent's
own file-edit tool — Claude Code's docs say file changes made by Bash are not tracked, and
Gemini CLI's checkpoints cover only its edit tools. So `rm -rf`, `git reset --hard`,
`sed -i`, a migration against your local SQLite file, or an over-eager "cleanup" can destroy
work that no undo button sees.

Agent VCR's **airbag** snapshots your project *before every tool call* — shell commands
included — and `vcr undo` restores it. Commands a snapshot cannot undo (remote databases,
`terraform destroy`, force-pushes, `npm publish`) are stopped so a human can confirm.

## The airbag in 30 seconds

```text
$ vcr airbag install claude
  ✓ Claude Code: .claude/settings.json

# ...the agent decides to "clean up" and runs `rm -rf ~/code/myapp`...
🛟 Agent VCR airbag: snapshot #1 saved before `rm -rf ~/code/myapp`. Undo with `vcr undo`.

$ vcr undo --project-dir ~/code/myapp
↩ Undid: rm -rf ~/code/myapp  (claude, 12s ago)
  restored 5 files — back to snapshot #1.
    + .claude/settings.json
    + .env
    + .gitignore
    + src/billing.py
    + .git/
  Changed your mind? `vcr redo`
```

Your uncommitted edits, your git-ignored `.env`, and your git history are back. Try it
yourself without an agent: [`examples/airbag_demo.sh`](https://github.com/ixchio/agent-vcr/blob/main/examples/airbag_demo.sh),
or classify any command:

```text
$ vcr airbag check "psql -h prod-db.internal -c 'DROP TABLE invoices'"
⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
   • SQL DROP on a remote database (prod-db.internal) — server data is not in your project files
```

### Supported agents

| Agent | Install | Commands a snapshot cannot undo |
|---|---|---|
| Claude Code | `vcr airbag install claude` | asks you |
| OpenAI Codex CLI | `vcr airbag install codex` | blocked until `vcr airbag allow` |
| Gemini CLI | `vcr airbag install gemini` | asks you (≥ 0.36) |
| Cursor (IDE + CLI) | `vcr airbag install cursor` | blocked until `vcr airbag allow` |
| GitHub Copilot CLI | `vcr airbag install copilot` | asks you |
| OpenHands | `vcr airbag install openhands` or the [Python integration](https://github.com/ixchio/agent-vcr/blob/main/docs/airbag.md#supported-agents) | blocked, or asks with `confirm=True` |
| OpenCode 1.x | `vcr airbag install opencode` | blocked until `vcr airbag allow` |
| Cline | `vcr airbag install cline` | task stopped |
| Devin Local (ex-Windsurf) | `vcr airbag install devin` | blocked until `vcr airbag allow` |
| Anything else | `vcr airbag run -- aider` | snapshots every 2 s (no interception) |

**What it protects:** tracked and untracked files, small ignored "precious" files (`.env*`,
SQLite databases — copied consistently even while open), and your `.git` when a command would
delete or damage it. Snapshots live in a private git store **outside** the project, so deleting
the project cannot delete them, and your own `.git` is never touched.
**What it cannot undo:** anything outside the project, remote databases, cloud resources,
pushed commits, sent requests. Those are stopped for confirmation. Your **local Postgres** can
be covered too: list it in `AGENT_VCR_AIRBAG_DATABASES` and a `DROP TABLE` gets a `pg_dump`
first, so `vcr undo` brings the rows back with the files. Details and limits:
[docs/airbag.md](https://github.com/ixchio/agent-vcr/blob/main/docs/airbag.md).

| Command | |
|---|---|
| `vcr undo [N]` · `vcr redo` | undo / redo the last agent action that changed files |
| `vcr rewind` · `vcr rewind 12` | timeline of snapshots · restore snapshot #12 |
| `vcr airbag check "<cmd>"` | classify a command without running it |
| `vcr airbag allow` | approve a stopped command for one retry |
| `vcr airbag status` · `vcr airbag gc` | installed hooks and store · delete old snapshots |
| `vcr bisect -- pytest -q` | find the agent action that broke your tests — zero model tokens |
| `vcr mcp` | an MCP server, so the agent can check commands and see its own undo history |

### Which of the 40 tool calls broke it? `vcr bisect`

Tests were green before the agent started; now they're red and nothing is committed. Every tool
call has a snapshot, so `vcr bisect` binary-searches them in temporary checkouts — your project
is never touched — and names the culprit:

```text
$ vcr bisect -- pytest -q
  #4 (current)   ✗ fail   (0.4s)
  #1             ✓ pass   (0.3s)
  #2             ✓ pass   (0.3s)
  #3             ✗ fail   (0.3s)

🎯 First bad action: sed -i 's/round(sum(prices), 2)/int(sum(prices))/' billing.py
   claude · Bash · snapshot #2 passes → #3 fails   (4 test runs)
     ~ billing.py
   Go back to just before it:  vcr rewind 2
```

### Give the agent its own safety net: `vcr mcp`

```bash
claude mcp add agent-vcr -- vcr mcp      # Codex: codex mcp add agent-vcr -- vcr mcp
```

The agent can now ask `airbag_check` before a risky command, read its timeline, take a snapshot,
and preview an undo. Restoring files through MCP needs `dry_run: false` **and** `confirm: true`.
See [docs/mcp.md](https://github.com/ixchio/agent-vcr/blob/main/docs/mcp.md).

## More than an undo button

### Record and time-travel agent runs

Every step — state in, state out, tool calls, model calls, cost, latency, errors — is a frame in
an append-only JSONL file you can open, diff, bundle, and replay.

```python
from agent_vcr import VCRPlayer, VCRRecorder

recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session("billing-fix")
recorder.record_step("planner", {"task": "add invoice status"}, {"plan": "migration first"})
recorder.record_checkpoint({"plan": "migration first"}, label="plan-approved")
path = recorder.save()

player = VCRPlayer.load(path)
player.goto_checkpoint("plan-approved")   # jump to any frame or checkpoint
player.verify_integrity()                 # durability checks
```

Fork a run from any frame, edit state, and resume; compare frames; open a recording in the
terminal debugger with `vcr path/to/run.vcr`.

### Roll back files and state together in your own agent

`ACIDWorkspace` gives Python agents transactions over a git repository: savepoints pair a git
commit with the agent's state, rollback restores both, and pre-existing untracked files such as
`.env` are escrowed byte-for-byte outside git history.

```python
from agent_vcr.integrations.openhands import ACIDWorkspace

with ACIDWorkspace("/path/to/repo") as ws:          # begin() on enter
    ws.savepoint({"phase": "planned"}, node_name="planner")
    ...                                             # the agent works on an isolated branch
    ws.rollback(to_frame_index=0)                   # rewind files + state to a savepoint
# clean exit → commit (merge); exception → abort (everything restored)
```

### Ghost Replay and Agent CI

Save a verified run as a *golden* run keyed by model, prompt hash, code commit and tool schema;
replay it with zero model calls, re-run only changed steps, and see what you saved. Turn golden
runs into datasets and fail a pull request on behavioral, cost, token or latency regressions:

```bash
vcr dataset create refunds
vcr dataset add refunds --task "refund a duplicate payment" --vcr .vcr/golden/refund.vcr
vcr ci run --dataset refunds --candidates .vcr/candidates --fail-on degraded
```

A reusable GitHub Action is included ([action.yml](https://github.com/ixchio/agent-vcr/blob/main/action.yml)).

Or test agents the way you test code — the pytest plugin records on the first run and replays
with zero model calls after that, failing the test on a behavior, cost or token regression:

```python
@pytest.mark.agent_vcr(task="triage ticket t_123", identity={"model": "gpt-5"})
def test_triage(agent_vcr):
    run = agent_vcr.run(triage)            # 1st run: calls the agent; after: replays it
    assert run.output["queue"] == "billing"
    agent_vcr.assert_matches_golden(max_cost_usd=0.05)
```

`AGENT_VCR_RECORD=none pytest` in CI fails instead of calling a model. See
[docs/pytest-plugin.md](https://github.com/ixchio/agent-vcr/blob/main/docs/pytest-plugin.md).

### Sentinel

An AST quality guard for agent-written Python: syntax errors, complexity and length spikes,
duplicate functions across files. `sentinel scan .` in CI, or attach it to an OpenHands run.

## Framework integrations

```python
from agent_vcr import VCRRecorder

recorder = VCRRecorder()
recorder.start_session("my-run")      # ... run your agent ..., then recorder.save()
```

```python
# LangGraph — no graph changes: pass the callback per run …
from agent_vcr.integrations.langgraph import LangGraphCallback
app.invoke(inputs, config={"callbacks": [LangGraphCallback(recorder)]})

# … or wrap every node of a StateGraph (before or after compile)
from agent_vcr.integrations.langgraph import VCRLangGraph
graph = VCRLangGraph(recorder).wrap_graph(builder)
```

```python
# CrewAI
from agent_vcr.integrations.crewai import VCRCrewAI, VCRCrewCallback
result = VCRCrewAI(recorder).kickoff(crew)        # or: with VCRCrewCallback(recorder): crew.kickoff()
```

```python
# OpenHands SDK — snapshot before every action, block irreversible ones
from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect
airbag = AirbagCallback()
conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag])
protect(conversation, airbag)
```

Any other framework: call `record_step`, `record_llm_call`, `record_tool_call`, `record_error`
and `record_checkpoint` from its lifecycle hooks, or forward frames live with
`VCRRecorder(on_frame_recorded=...)`. Guides:
[docs/integrations](https://github.com/ixchio/agent-vcr/blob/main/docs/integrations/README.md).

## Architecture

```text
 coding agents ──hooks/plugins──►  airbag ──► shadow git store (outside the project)
 (Claude Code, Codex, …)            │           └─ vcr undo / redo / rewind
                                    ▼
 frameworks ──callbacks──►  recorder ──► .vcr JSONL (execution/v1) ──► player
 (LangGraph, CrewAI, …)        ▲                                        │
                               │                          Ghost Replay ◄┘──► Agent CI
                      ACIDWorkspace (git transactions + escrow)
```

| Layer | Status |
|---|---|
| Recording kernel (`recorder`, `player`, `execution/v1` schema) | stable — old recordings always load |
| Airbag, ACID workspace, Sentinel | stable API |
| Ghost Replay, datasets, Agent CI, bundles | stable |
| LangGraph / CrewAI / OpenHands / Claude Code integrations | stable — tested against the real libraries |
| Dashboard and self-hosted store (`vcr-server`) | supported |
| SuperAgent, workflows, sandbox, intelligence | experimental — APIs may change |

The full package map and design notes are in
[docs/architecture.md](https://github.com/ixchio/agent-vcr/blob/main/docs/architecture.md).

## Install what you need

```bash
pip install ai-agent-vcr                  # core: recorder, player, airbag, replay, CI
pip install "ai-agent-vcr[tui]"          # terminal debugger
pip install "ai-agent-vcr[dashboard]"    # local dashboard and server
pip install "ai-agent-vcr[langgraph]"    # LangGraph integration
pip install "ai-agent-vcr[crewai]"       # CrewAI integration
pip install "ai-agent-vcr[openhands]"    # OpenHands SDK 1.x (Python 3.12+)
pip install "ai-agent-vcr[all]"          # everything optional
```

Python 3.9–3.13 on Linux, macOS and Windows (all tested in CI). The airbag and ACID workspace
need `git` on `PATH`.

## Where it fits

Agent VCR is an **intervention** tool, not an observability platform. LangSmith, Langfuse,
Phoenix and AgentOps observe and evaluate model calls; Agent VCR lets you go back — files, state
and history — after an agent changed them. They work well together. It cannot make third-party
side effects (emails, payments, cloud APIs) transactional; it stops those for confirmation.

## Documentation

- [Airbag](https://github.com/ixchio/agent-vcr/blob/main/docs/airbag.md) — per-agent support, commands, limits
- [Recording, time travel and Ghost Replay](https://github.com/ixchio/agent-vcr/blob/main/docs/recording-and-replay.md)
- [ACID workspace rollback](https://github.com/ixchio/agent-vcr/blob/main/docs/acid-workspace.md)
- [Agent CI regression gates](https://github.com/ixchio/agent-vcr/blob/main/docs/agent-ci.md) ·
  [pytest plugin](https://github.com/ixchio/agent-vcr/blob/main/docs/pytest-plugin.md)
- [MCP server](https://github.com/ixchio/agent-vcr/blob/main/docs/mcp.md) — the airbag as agent tools
- [Integrations](https://github.com/ixchio/agent-vcr/blob/main/docs/integrations/README.md)
- [Architecture](https://github.com/ixchio/agent-vcr/blob/main/docs/architecture.md)
- [All docs](https://github.com/ixchio/agent-vcr/blob/main/docs/README.md) ·
  [Live site](https://ixchio.github.io/agent-vcr/) ·
  [Examples](https://github.com/ixchio/agent-vcr/blob/main/examples/README.md) ·
  [Changelog](https://github.com/ixchio/agent-vcr/blob/main/CHANGELOG.md)

## Contributing

Issues and pull requests are welcome — new agent integrations especially. See
[CONTRIBUTING.md](https://github.com/ixchio/agent-vcr/blob/main/CONTRIBUTING.md) (including
*Adding an agent to the airbag*), run `make verify`, and include a test with every change.
Please follow the [Code of Conduct](https://github.com/ixchio/agent-vcr/blob/main/CODE_OF_CONDUCT.md)
and report vulnerabilities privately per [SECURITY.md](https://github.com/ixchio/agent-vcr/blob/main/SECURITY.md).

## License

[MIT](https://github.com/ixchio/agent-vcr/blob/main/LICENSE) · Copyright © 2026 QueryXLabs and
[ixchio](https://github.com/ixchio).
