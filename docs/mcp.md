# 🧰 MCP server — let your agent see and drive its own safety net

`vcr mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server. Register it once,
and any MCP-capable agent (Claude Code, Codex CLI, Cursor, Gemini CLI, …) gets the airbag as tools:
it can ask "would this command be stopped?" before running it, take a save point before a risky
refactor, read its own timeline, and preview an undo for you.

You ask your agent to "drop the old migrations and force-push the cleanup". Before it runs
anything, it calls `airbag_check`:

```text
git push --force origin main  →  irreversible
  • force-push / remote delete rewrites shared history
  advice: A snapshot cannot undo this. Explain what it does and get the user's explicit
          go-ahead before running it; the airbag will stop it otherwise.
```

So it stops and asks you first, instead of running into the airbag's block.

**TL;DR**

- `claude mcp add agent-vcr -- vcr mcp` (or the one-liner for your agent below), and the agent
  gets 8 tools: status, timeline, check, snapshot, undo, rewind, and two for `.vcr` recordings.
- Everything is read-only except `airbag_snapshot`, which only writes to the snapshot store
  outside your project.
- `airbag_undo` and `airbag_rewind` **only preview** unless called with `dry_run: false` **and**
  `confirm: true`. Their descriptions tell the model to do that only when you asked.
- No extra dependencies: the server uses only the Python standard library. Nothing leaves your
  machine.

## Try it in 30 seconds

No agent needed. Pipe two JSON-RPC requests into the server and read the answers:

```bash
pip install ai-agent-vcr
cd your-project
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"me","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"airbag_check","arguments":{"command":"git push --force origin main"}}}' \
  | vcr mcp
# {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"agent-vcr",…}}}
# {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{…}"}],"structuredContent":{"command":"git push --force origin main","level":"irreversible",…},"isError":false}}
```

The server exits when its input closes. Then register it with your agent (next section) and ask
it: *"what does the Agent VCR airbag say about `rm -rf build`?"*

## Register it with your agent

| Agent | Command or config | Where it lands |
|---|---|---|
| Claude Code | `claude mcp add agent-vcr -- vcr mcp` | `~/.claude.json`, this project only (`-s user` for every project, `-s project` for a shared `.mcp.json`) |
| OpenAI Codex CLI | `codex mcp add agent-vcr -- vcr mcp` | `~/.codex/config.toml` (all projects) |
| Gemini CLI | `gemini mcp add agent-vcr vcr mcp` | `.gemini/settings.json` (`-s user` for `~/.gemini/settings.json`) |
| Cursor | the JSON below | `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` |

Cursor (and any client that uses the common `mcpServers` format, Gemini CLI included):

```json
{
  "mcpServers": {
    "agent-vcr": {
      "command": "vcr",
      "args": ["mcp"]
    }
  }
}
```

Codex writes this to `~/.codex/config.toml`:

```toml
[mcp_servers.agent-vcr]
command = "vcr"
args = ["mcp"]
```

`vcr` must be on the `PATH` the agent sees. If it isn't (a virtualenv, pipx), use the full path,
for example `claude mcp add agent-vcr -- /home/you/.venv/bin/vcr mcp`.

What was verified: Claude Code 2.1.270 added the server and `claude mcp list` reported
`agent-vcr: vcr mcp - ✔ Connected`; `codex mcp add` (codex-cli 0.154.0) and `gemini mcp add`
(Gemini CLI 0.37.1) wrote exactly the config shown above. The Cursor snippet follows
[Cursor's MCP docs](https://cursor.com/docs/context/mcp); it was not tried in Cursor itself.
Agent VCR does not edit these configs for you: the commands above are each agent's own.

### Which project do the tools act on?

The directory the agent starts the server in, which is normally your project. Every airbag tool also takes an optional `project_dir`, and you can pin the default:

```bash
claude mcp add agent-vcr -- vcr mcp --project-dir /path/to/project
```

The project is found the same way `vcr undo` finds it (the git top-level, or the nearest
protected directory), so a subdirectory works too.

## The tools

| Tool | What it does | Changes anything? |
|---|---|---|
| `airbag_status` | Protected or not, policy, snapshot store and count, latest snapshot, which agents have hooks | No |
| `airbag_timeline` | Recent protected actions, newest first: snapshot number, when, agent, tool, command, level (`limit`, default 20) | No |
| `airbag_check` | Classifies a shell command **without running it**: `read`, `write`, `destructive` or `irreversible`, with reasons and advice (`command`, optional `cwd`) | No |
| `airbag_snapshot` | Takes a save point now (`message`) | Only the snapshot store, never project files |
| `airbag_undo` | Previews undoing the last file-changing action(s) (`steps`). Restores only with `dry_run: false` + `confirm: true` | Yes, when confirmed |
| `airbag_rewind` | Previews restoring snapshot `#N` (`snapshot`). Restores only with `dry_run: false` + `confirm: true` | Yes, when confirmed |
| `recordings_list` | Finds `.vcr` recordings under a directory, newest first (skips `.git`, `node_modules`, virtualenvs) | No |
| `recording_summary` | One recording: frames, nodes, checkpoints, errors, tokens, cost, latency, models, integrity | No |

Each result comes back twice, as MCP recommends: as JSON text for the model, and as
`structuredContent` for clients that read it. Problems such as a bad argument, an unknown
snapshot, a missing file, or a refused restore come back as a result with `isError: true` and a
message the model can act on. The server keeps running.

## How it works

`vcr mcp` speaks MCP's stdio transport: one JSON-RPC 2.0 message per line on stdin, one per
line on stdout. It is written with the Python standard library only, and the official `mcp`
Python SDK client is used in the tests to check it.

- **Lifecycle.** `initialize` echoes the client's protocol version if it is one the server
  supports (`2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`), otherwise it answers with
  the newest. It declares only the `tools` capability. It also supports `ping`, `tools/list` and
  `tools/call`. Notifications get no reply. Unknown methods get JSON-RPC error `-32601`, unknown
  tools `-32602`, and malformed JSON `-32700`.
- **A clean stdout.** At startup the server moves the protocol streams to private file
  descriptors and points fd 1 at stderr and fd 0 at `/dev/null`. A stray `print` or a child
  process (git) cannot corrupt the JSON stream or swallow your client's messages. Logs go to
  stderr; set `AGENT_VCR_MCP_LOG=INFO` to see them.
- **Same engine as the CLI.** The tools call the same airbag code as `vcr airbag status`,
  `vcr airbag check`, `vcr airbag snapshot`, `vcr undo --dry-run`, `vcr undo` and `vcr rewind`,
  and the same `VCRPlayer` that opens recordings. The airbag's rules hold: it never grants a
  permission, and every restore snapshots the current state first, so `vcr redo` can undo the
  undo.
- **One request at a time**, in order. Snapshot and restore work is bounded by the store's git
  timeouts, and the recordings scan stops after 10 seconds or 200,000 entries
  (`scan_truncated: true`).

## Recipes

### "Check before you run" as a standing instruction

Add one line to your `CLAUDE.md`, `AGENTS.md` or `GEMINI.md`:

```markdown
Before running a shell command that deletes, overwrites, resets or pushes, call the agent-vcr
`airbag_check` tool; if it says `irreversible`, ask me first.
```

The server also sends a short version of this as its MCP `instructions`, for clients that show
them to the model.

### Ask your agent to undo, safely

> *"Undo what you just did."*

The agent calls `airbag_undo`, which by default is a preview. It shows you the plan:
`restore a.txt`, `remove scratch.py`. Once you say yes, it calls again with `dry_run: false` and
`confirm: true`. Without `confirm: true` the server refuses and returns the preview:

```text
Error: refusing to undo without confirm=true. This changes the user's files: call with
dry_run=false and confirm=true only when the user explicitly asked for it.
```

Changed your mind? `vcr redo` in a terminal.

### A save point before a big change

> *"Take an airbag snapshot called 'before the auth rewrite', then start."*

`airbag_snapshot` returns the number (`#7`). Later, *"rewind to #7"* goes through
`airbag_rewind`, preview first.

### What happened in that run?

> *"List the recordings in .vcr and summarize the newest."*

`recordings_list` then `recording_summary` give the agent frame counts, which nodes ran, the
error frames with messages, and tokens and cost, without loading the whole file into context.

## What it can't do

- **It doesn't install hooks.** `airbag_status` tells the agent (and you) when nothing is armed;
  run `vcr airbag install` yourself. The MCP server never changes agent configuration, and has no
  `vcr mcp install`. Use the agent's own `mcp add` command above.
- **It is not the airbag.** The tools let an agent *cooperate* with the safety net. Protection
  still comes from the hooks, which snapshot every tool call whether or not the agent remembers
  to call `airbag_check`.
- **Confirmation is a contract with the model, not a lock.** `confirm: true` is a deliberate
  second step, and the tool descriptions say to use it only when you asked. A model that ignores
  that can still restore files, so keep your agent's own tool-approval prompts on for
  `airbag_undo` and `airbag_rewind` if you want a human click every time. Even then, a restore can
  always be reversed with `vcr redo`.
- **No approvals over MCP.** Approving a stopped irreversible command stays a human action in
  your terminal (`vcr airbag allow`). An agent cannot approve its own command.
- **stdio only.** There is no HTTP transport; the server runs locally, launched by your agent.

## Cheat sheet

```bash
vcr mcp                                   # serve MCP on stdin/stdout (the agent starts this)
vcr mcp --project-dir /path/to/project    # pin the default project
claude mcp add agent-vcr -- vcr mcp       # Claude Code
codex mcp add agent-vcr -- vcr mcp        # Codex CLI
gemini mcp add agent-vcr vcr mcp          # Gemini CLI
AGENT_VCR_MCP_LOG=INFO vcr mcp            # protocol logs on stderr
```

Tools: `airbag_status` · `airbag_timeline` · `airbag_check` · `airbag_snapshot` ·
`airbag_undo` · `airbag_rewind` · `recordings_list` · `recording_summary`.

Related: [Airbag](airbag.md) · [Recording & time travel](recording-and-replay.md) ·
[MCP specification](https://modelcontextprotocol.io/specification/2025-11-25)
