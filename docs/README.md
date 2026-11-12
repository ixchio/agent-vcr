# Agent VCR docs

**Agent VCR is the undo button for AI agents.** Your agent deleted a folder, rewrote the wrong
file, or nuked a SQLite table? Rewind it. Want to replay a run that worked without paying for the
model again, or stop a pull request that makes your agent worse? That's here too.

```bash
pip install ai-agent-vcr
```

No API key. No account. Nothing leaves your machine. Python 3.9–3.13.

New here? → **[Quickstart: every superpower in 5 minutes](quickstart.md)**

## Pick your path

| You are… | Start with | Then |
|---|---|---|
| 🧑‍💻 **Using a coding agent** (Claude Code, Codex, Cursor, Gemini CLI, Copilot CLI, OpenCode, Cline, Devin, OpenHands) | [Airbag](airbag.md) — undo anything the agent did, shell commands included | [FAQ](faq.md) |
| 🛠️ **Building an agent** (LangGraph, CrewAI, OpenHands SDK, plain Python) | [Recording & time travel](recording-and-replay.md) | [Integrations](integrations/README.md) · [ACID workspace](acid-workspace.md) |
| ✅ **Owning CI / quality** for an agent | [Agent CI](agent-ci.md) — regression tests for agent behavior | [Sentinel](sentinel.md) |
| 🏗️ **Running it for a team** | [Dashboard & self-hosted store](dashboard.md) | [Railway deployment](deployment/railway.md) |
| 🔧 **Contributing** | [CONTRIBUTING.md](../CONTRIBUTING.md) | [Architecture](architecture.md) |

## Every feature, one line each

| Feature | What it does for you | Guide |
|---|---|---|
| 🛟 **Airbag** | Snapshots your project before every coding-agent tool call; `vcr undo` brings it back | [airbag.md](airbag.md) |
| 🧰 **MCP server** | `vcr mcp` gives any MCP agent (Claude Code, Codex, Cursor, Gemini CLI) airbag tools: check a command first, snapshot, preview undo | [mcp.md](mcp.md) |
| ⏪ **Recording & time travel** | Records every step of your agent; jump to any step, diff, fork, resume | [recording-and-replay.md](recording-and-replay.md) |
| 👻 **Ghost Replay** | Re-runs a verified run from its recording — zero model calls, zero tokens | [recording-and-replay.md](recording-and-replay.md) |
| 🔁 **ACID workspace** | Transactions for your own agent: savepoint, rollback, commit — files and state together | [acid-workspace.md](acid-workspace.md) |
| ✅ **Agent CI** | Fails a pull request when the agent's behavior, cost or speed regresses | [agent-ci.md](agent-ci.md) |
| 🧪 **pytest plugin** | Records your agent once in a test, replays it on every later run — zero tokens in CI | [pytest-plugin.md](pytest-plugin.md) |
| 🔍 **Sentinel** | Catches broken, bloated or duplicated code your agent just wrote | [sentinel.md](sentinel.md) |
| 📺 **Dashboard & TUI** | Browse recordings in the terminal or a local web UI; share them with your team | [dashboard.md](dashboard.md) |
| 🔌 **Integrations** | LangGraph, CrewAI, OpenHands, Claude Code, n8n, or your own framework | [integrations/](integrations/README.md) |

🧪 **Experimental** (works and is tested, API may change):
[SuperAgent runtime](experimental/superagent.md) ·
[Workflow engine](experimental/workflow.md) ·
[Reliability intelligence](experimental/intelligence.md)

## The honest fine print

* Agent VCR restores **your project directory** — files, `.env`, local SQLite, your `.git`. It
  cannot un-send an email, un-charge a card, or un-drop a table on a server. The airbag spots
  those commands and makes a human confirm first.
* Recordings contain what your agent saw and did (prompts, tool inputs, outputs). Keep `.vcr`
  folders private or redact before recording.

## Words you'll see

| Word | Meaning |
|---|---|
| **Snapshot** | The airbag's copy of your project, taken right before an agent tool call |
| **Session / recording** | One agent run, saved as a `.vcr` file (one JSON object per line) |
| **Frame** | One step inside a recording: state in, state out, timing, cost |
| **Checkpoint** | A named frame you can jump back to (`"plan-approved"`) |
| **Savepoint** | An ACID workspace checkpoint that also pins the files (a git commit) |
| **Golden run** | A recording you've verified and want to replay or test against |
| **Ghost Replay** | Re-running a golden run from its recording instead of calling the model |
| **Bundle** | A portable, checksummed `.vcrb` file you can share or attach to a bug |

## More

[Architecture](architecture.md) · [FAQ](faq.md) · [Examples](../examples/README.md) ·
[Changelog](../CHANGELOG.md) · [Security](../SECURITY.md) · [Live site](https://ixchio.github.io/agent-vcr/)
