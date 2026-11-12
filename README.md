# Agent VCR

[![PyPI Version](https://img.shields.io/pypi/v/ai-agent-vcr?style=flat-square&label=PyPI)](https://pypi.org/project/ai-agent-vcr/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ai-agent-vcr?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ai-agent-vcr)
[![Python Versions](https://img.shields.io/pypi/pyversions/ai-agent-vcr?style=flat-square)](https://pypi.org/project/ai-agent-vcr/)
[![CI](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ixchio/agent-vcr/actions/workflows/ci.yml)
[![MIT License](https://img.shields.io/pypi/l/ai-agent-vcr?style=flat-square)](https://github.com/ixchio/agent-vcr/blob/main/LICENSE)

### Ctrl+Z for AI agents.

Your coding agent just ran `rm -rf`. Or `git reset --hard`. Or `DROP TABLE users` on your dev
database. **`vcr undo` brings it all back**: files, `.env`, git history, even the rows.

```bash
pip install ai-agent-vcr
vcr airbag install        # arms every coding agent on your PATH
vcr undo                  # oops? un-oops.
```

No API key. No account. No cloud. Everything stays on your machine.
Works with **Claude Code, Codex, Gemini CLI, Cursor, Copilot CLI, OpenHands, OpenCode, Cline
and Devin**, on Linux, macOS and Windows.

![Agent VCR recording, rollback, and replay demo](https://raw.githubusercontent.com/ixchio/agent-vcr/main/docs/demo.gif)

---

## The problem

Coding agents don't just edit files: they run shell commands. And the "rewind" buttons built
into agent tools only track the agent's own edit tool. Claude Code's docs say changes made
through Bash aren't tracked; Gemini CLI's checkpoints cover only its edit tools.

So the scary stuff (`rm -rf`, `sed -i`, `git clean -fdx`, a migration against your local DB, an
over-eager "let me clean that up") is exactly what nothing can undo.

**Agent VCR fixes that with an airbag:** a snapshot of your project before *every* tool call,
shell commands included. `vcr undo` restores it. Commands no snapshot can undo
(force-pushes, `terraform destroy`, `npm publish`, a DROP on a remote database) are stopped
until a human says yes.

## 60-second tour

```text
$ vcr airbag install claude
  ✓ Claude Code: .claude/settings.json

# ... the agent decides to "clean up" and runs `rm -rf ~/code/myapp` ...
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

Uncommitted edits, the git-ignored `.env`, and the whole git history: back. Want to see it
without an agent? Run
[`examples/airbag_demo.sh`](https://github.com/ixchio/agent-vcr/blob/main/examples/airbag_demo.sh).
Want to know what the airbag thinks of a command before anyone runs it?

```text
$ vcr airbag check "psql -h prod-db.internal -c 'DROP TABLE invoices'"
⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
   • SQL DROP on a remote database (prod-db.internal) — server data is not in your project files
```

## What's in the box

| | You type | What happens |
|---|---|---|
| 🛟 **Undo anything** | `vcr undo` · `vcr redo` · `vcr rewind 12` | Files, `.env`, SQLite, git history, back to before any agent action |
| 🎯 **Find what broke it** | `vcr bisect -- pytest -q` | Binary-searches the agent's actions and names the one that broke your tests. Zero tokens |
| 🐘 **Rewind your local DB** | `AGENT_VCR_AIRBAG_DATABASES=...` | `pg_dump` before a destructive SQL command; `vcr undo` restores the rows too |
| 🔌 **Give the agent a seatbelt** | `claude mcp add agent-vcr -- vcr mcp` | The agent checks risky commands and reads its own undo history over MCP |
| 📼 **Record & time-travel** | `VCRRecorder` / `vcr run.vcr` | Every step of your own agent as a frame: inspect, fork, resume, diff |
| 👻 **Replay for free** | Ghost Replay | Re-run a verified run with zero model calls |
| 🧪 **Test agents like code** | `@pytest.mark.agent_vcr` | Record once, replay forever, fail the test on behavior/cost/token regressions |
| 🚦 **Gate PRs** | `vcr ci run` + GitHub Action | Block a pull request that makes your agent worse or pricier |

## Supported agents

| Agent | Arm it | When a command can't be undone |
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

The airbag **never grants a permission** your agent wouldn't have asked for. It only adds
snapshots, and stops the things a snapshot can't fix.

## How it works

```text
 agent wants to run a tool
          │
          ▼
 PreToolUse hook ─► classify: read · write · destructive · irreversible
          │                    │                        │
          │          snapshot the project         ask a human / block
          ▼          (shadow git store,           (force-push, prod DB,
   tool runs         outside your project)         cloud deletes, publish)
                               │
                 vcr undo / redo / rewind / bisect
```

- **Snapshots live outside your project** (`~/.local/state/agent-vcr/airbag/…`), in a private
  git store. Deleting the project can't delete them. Your own `.git` is never written, except to
  put back one that was deleted or broken.
- **What gets captured:** tracked and untracked files, small ignored "precious" files (`.env*`,
  SQLite, copied consistently even while open), and your `.git` before a command that would
  wreck it. Dependency trees (`node_modules`, `.venv`) are skipped; reinstall those.
- **Snapshots are cheap:** content-addressed, and an unchanged tree reuses the last one.
- **Every restore is undoable:** the current state is snapshotted first, so `vcr redo` works.

## Find the action that broke it: `vcr bisect`

Tests were green before the agent started. Forty tool calls later they're red and nothing is
committed. Which call did it? `git bisect` can't help: there are no commits. The airbag has
a snapshot before every call, so:

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

Each test run happens in a temporary checkout: your project is never touched. 100 actions take
about 9 runs. Zero model tokens.

## Rewind your local Postgres too

A file snapshot can't undo `DROP TABLE`: the data lives in the database server. For local dev
databases you name explicitly, the airbag runs `pg_dump` right before the destructive SQL:

```bash
export AGENT_VCR_AIRBAG_DATABASES=postgresql://localhost:5433/app_dev
vcr undo     # files AND rows back
```

Only `localhost`, `127.0.0.1`, `::1` and unix sockets are accepted; remote databases are never
dumped and still need a human. If the airbag can't be *certain* where a command connects
(ORM resets, `-f script.sql`, `PGSERVICE`, a `\connect` inside the SQL), it takes the dump
**and** still asks.

## Give the agent its own seatbelt: `vcr mcp`

```bash
claude mcp add agent-vcr -- vcr mcp      # Codex: codex mcp add agent-vcr -- vcr mcp
```

Now the agent can call `airbag_check` before a risky command, read its timeline, take a
snapshot, and preview an undo. Actually restoring files over MCP takes `dry_run: false`
**and** `confirm: true`, and only works on the project the server was started in.
→ [docs/mcp.md](https://github.com/ixchio/agent-vcr/blob/main/docs/mcp.md)

## Building your own agent? Record it.

Every step (state in, state out, tool calls, model calls, cost, latency, errors) becomes a
frame in an append-only JSONL file you can open, diff, bundle and replay.

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

Fork from any frame, edit state, resume. Open a recording in the terminal debugger with
`vcr path/to/run.vcr`. Long runs? `VCRRecorder(max_cached_frames=1000)` keeps memory flat.

### Test agents like code (pytest plugin)

Records the first time, then replays with **zero model calls**, failing the test when
behavior, cost or tokens regress:

```python
@pytest.mark.agent_vcr(task="triage ticket t_123", identity={"model": "gpt-5"})
def test_triage(agent_vcr):
    run = agent_vcr.run(triage)            # 1st run: calls the agent; after: replays it
    assert run.output["queue"] == "billing"
    agent_vcr.assert_matches_golden(max_cost_usd=0.05)
```

`AGENT_VCR_RECORD=none pytest` in CI fails instead of spending tokens.
→ [docs/pytest-plugin.md](https://github.com/ixchio/agent-vcr/blob/main/docs/pytest-plugin.md)

### Gate pull requests (Agent CI)

Turn verified runs into a dataset; fail the PR on behavioral, cost, token or latency
regressions. Comes with a GitHub Action
([action.yml](https://github.com/ixchio/agent-vcr/blob/main/action.yml)).

```bash
vcr dataset create refunds
vcr dataset add refunds --task "refund a duplicate payment" --vcr .vcr/golden/refund.vcr
vcr ci run --dataset refunds --candidates .vcr/candidates --fail-on degraded
```

### Transactions for your agent's workspace

`ACIDWorkspace` pairs each git savepoint with the agent's state. Rollback restores both, and
untracked files like `.env` are escrowed byte-for-byte outside git history.

```python
from agent_vcr.integrations.openhands import ACIDWorkspace

with ACIDWorkspace("/path/to/repo") as ws:          # begin() on enter
    ws.savepoint({"phase": "planned"}, node_name="planner")
    ...                                             # the agent works on an isolated branch
    ws.rollback(to_frame_index=0)                   # rewind files + state to a savepoint
# clean exit → commit (merge); exception → abort (everything restored)
```

### Sentinel

An AST guard for agent-written Python: syntax errors, complexity and length spikes, duplicate
functions. `sentinel scan .` in CI, or plug it into an OpenHands conversation: it stops the agent's next
move when an edit *introduces* a blocker (pre-existing mess is only a warning).

## Framework integrations

```python
# LangGraph: no graph changes, pass the callback per run …
from agent_vcr.integrations.langgraph import LangGraphCallback
app.invoke(inputs, config={"callbacks": [LangGraphCallback(recorder)]})

# … or wrap every node of a StateGraph
from agent_vcr.integrations.langgraph import VCRLangGraph
graph = VCRLangGraph(recorder).wrap_graph(builder)

# CrewAI
from agent_vcr.integrations.crewai import VCRCrewAI
result = VCRCrewAI(recorder).kickoff(crew)

# OpenHands SDK: snapshot before every action, block irreversible ones
from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect
airbag = AirbagCallback()
conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag])
protect(conversation, airbag)
```

Anything else: call `record_step`, `record_llm_call`, `record_tool_call`, `record_error` and
`record_checkpoint` from your framework's hooks.
→ [docs/integrations](https://github.com/ixchio/agent-vcr/blob/main/docs/integrations/README.md)

## What it can't do (honest section)

- **Undo the outside world.** Sent emails, payments, cloud resources, pushed commits, remote
  databases. Those get stopped for a human instead.
- **Snapshot your dependency folders.** `node_modules`, `.venv` and build output are skipped
  on purpose; reinstall them.
- **Read every mind.** The classifier reads POSIX shell. A command it doesn't recognise still
  gets a snapshot first, but an obfuscated or PowerShell-only danger may not be flagged as
  irreversible.
- **Replace observability.** LangSmith, Langfuse, Phoenix and AgentOps *watch* model calls;
  Agent VCR lets you *go back*. Use both.

## Install

```bash
pip install ai-agent-vcr                  # core: airbag, recorder, replay, CI, pytest plugin, MCP
pip install "ai-agent-vcr[tui]"          # terminal debugger
pip install "ai-agent-vcr[dashboard]"    # local web dashboard and server
pip install "ai-agent-vcr[langgraph]"    # LangGraph
pip install "ai-agent-vcr[crewai]"       # CrewAI
pip install "ai-agent-vcr[openhands]"    # OpenHands SDK 1.x (Python 3.12+)
pip install "ai-agent-vcr[all]"          # everything
```

Python 3.9–3.13 on Linux, macOS and Windows, every one tested in CI on every change. The
airbag needs `git` on your `PATH`.

## Stability

| Layer | Status |
|---|---|
| Airbag, bisect, MCP server, ACID workspace, Sentinel | stable API |
| Recording kernel (`recorder`, `player`, `execution/v1` schema) | stable: old recordings always load |
| Ghost Replay, datasets, Agent CI, pytest plugin, bundles | stable |
| LangGraph / CrewAI / OpenHands / Claude Code integrations | stable, tested against the real libraries |
| Dashboard and self-hosted store (`vcr-server`) | supported |
| SuperAgent, workflows, sandbox, intelligence | experimental: APIs may change |

Package map and design notes:
[docs/architecture.md](https://github.com/ixchio/agent-vcr/blob/main/docs/architecture.md).

## Docs

- **[Website](https://ixchio.github.io/agent-vcr/)** · [Documentation](https://ixchio.github.io/agent-vcr/guides.html) · [5-minute quickstart](https://github.com/ixchio/agent-vcr/blob/main/docs/quickstart.md)
- [Airbag](https://github.com/ixchio/agent-vcr/blob/main/docs/airbag.md): every agent, every command, every limit
- [MCP server](https://github.com/ixchio/agent-vcr/blob/main/docs/mcp.md) · [pytest plugin](https://github.com/ixchio/agent-vcr/blob/main/docs/pytest-plugin.md) · [Agent CI](https://github.com/ixchio/agent-vcr/blob/main/docs/agent-ci.md)
- [Recording, time travel and Ghost Replay](https://github.com/ixchio/agent-vcr/blob/main/docs/recording-and-replay.md) · [ACID workspace](https://github.com/ixchio/agent-vcr/blob/main/docs/acid-workspace.md) · [Integrations](https://github.com/ixchio/agent-vcr/blob/main/docs/integrations/README.md)
- [FAQ](https://github.com/ixchio/agent-vcr/blob/main/docs/faq.md) · [All docs](https://github.com/ixchio/agent-vcr/blob/main/docs/README.md) · [Examples](https://github.com/ixchio/agent-vcr/blob/main/examples/README.md) · [Changelog](https://github.com/ixchio/agent-vcr/blob/main/CHANGELOG.md)

## Contributing

PRs welcome, new agent integrations especially. Read
[CONTRIBUTING.md](https://github.com/ixchio/agent-vcr/blob/main/CONTRIBUTING.md) (it covers
*Adding an agent to the airbag*), run `make verify`, and ship a test with every change.
Be kind: [Code of Conduct](https://github.com/ixchio/agent-vcr/blob/main/CODE_OF_CONDUCT.md).
Found a security issue? Report it privately per
[SECURITY.md](https://github.com/ixchio/agent-vcr/blob/main/SECURITY.md).

## License

[MIT](https://github.com/ixchio/agent-vcr/blob/main/LICENSE) · Copyright © 2026 QueryXLabs and
[ixchio](https://github.com/ixchio).
