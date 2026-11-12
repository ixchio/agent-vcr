# FAQ

Short answers to the questions people actually ask. Each one links to the page with the details.

## The basics

### What is Agent VCR?

An undo button for AI agents. The **airbag** snapshots your project before every tool call a
coding agent makes, shell commands included, so `vcr undo` can bring back what an `rm -rf` took.
For agents you build yourself there's more: record runs frame by frame, roll back files and
state together, replay verified runs without model calls, and fail CI when behaviour regresses.
Start with the [quickstart](quickstart.md).

### How is it different from LangSmith, Langfuse, Phoenix or AgentOps?

They *watch*: traces, dashboards, evals. Agent VCR *steps in*: it takes you back to before the
damage, and stops commands it can't undo so a human can decide. It's an intervention tool, not
an observability platform. They work fine side by side.

### Claude Code has `/rewind`, Cursor has checkpoints, Gemini CLI has `/restore`. Why do I need this?

Those restore edits made through the agent's **own file-edit tools**. Claude Code's docs say file
changes made by Bash commands aren't tracked. Gemini CLI takes its checkpoints before its
file-modifying tools. Cursor's docs describe checkpoints as covering files the Agent modifies,
and don't promise shell commands. So `rm -rf`, `git reset --hard`, `sed -i` or a script that
rewrites your SQLite file can slip past them. The airbag hooks in *before every tool call*,
shell commands included, and it also restores `.env` files and `.git`.
See [Airbag](airbag.md).

### LangGraph already has a checkpointer. What does this add?

The checkpointer saves your **graph state**. It doesn't touch files on disk, so an agent that
edits a repo can't be rolled back with it. Agent VCR records LangGraph runs
(`LangGraphCallback`, `VCRLangGraph`) and adds file plus state rollback
([ACID workspace](acid-workspace.md)), zero-token [Ghost Replay](recording-and-replay.md) and
[Agent CI](agent-ci.md). Setup is in [Integrations](integrations/README.md).

### Does it replace git?

No. Keep committing. The airbag covers the gap *between* commits, plus the things git ignores
on purpose (`.env`, local databases). Your own `.git` is never read or written by a snapshot.

## Privacy and cost

### Does it send my code or data anywhere?

No. There's no telemetry, and snapshots and recordings stay on your disk. The only network
calls are ones you set up yourself: `vcr cloud` pushing to a server *you* point it at, MCP tool
discovery, the optional LLM planner, or a workflow `callback_url`.

### Do I need an API key or an account?

No. The airbag, recording, replay, Agent CI and Sentinel all run without one. You only need a
model key for features that call a model on purpose, like the optional
[`LLMPlanner`](experimental/superagent.md).

### Does the airbag slow my agent down?

About 0.05–0.1 s per tool call, mostly Python start-up; read-only commands skip the snapshot
entirely. The first snapshot of a project costs the most: about 1 s for a 20,000-file repository
(it was 5–7 s before 1.7.1); after that, a snapshot after a small change takes about 0.1 s. CI
fails if these regress. Numbers: [Airbag → How it works](airbag.md#how-it-works).

## Snapshots

### Where are the snapshots, how big do they get, how do I clean up?

In a private git store **outside** your project, so deleting the project can't delete them:

| OS | Location |
|---|---|
| Linux | `~/.local/state/agent-vcr/airbag/` (or `$XDG_STATE_HOME`) |
| macOS | `~/Library/Application Support/agent-vcr/airbag/` |
| Windows | `%LOCALAPPDATA%\agent-vcr\airbag` |

`AGENT_VCR_AIRBAG_HOME` moves it. Snapshots are git objects, so identical files are stored once,
and the `.git` backup uses hardlinks. Skipped on purpose: dependency and build folders
(`node_modules/`, `.venv/`, `dist/`, …) and files over 25 MB. To see the store and tidy it:

```bash
vcr airbag status                       # where your project's store lives
vcr airbag gc --keep-days 7             # delete old snapshots (keeps the newest 50)
vcr airbag gc --keep-days 1 --keep-last 10
```

### Can it undo a pushed commit, a migration on a server, or an email?

**No.** A snapshot only covers files in your project. What happens instead depends on whether the
command classifier recognises the danger:

- **Recognised: stopped for a human.** For example `git push --force`, SQL `DROP`/`TRUNCATE` on a
  server database (localhost servers too), `terraform destroy`, `kubectl delete`,
  `npm publish`, `curl -X DELETE`, and `rm` outside the project. Claude Code, Gemini CLI and
  Copilot CLI ask you. Codex, Cursor, OpenCode and Devin block until `vcr airbag allow`.
- **Not recognised: runs as a normal write**, with a snapshot of your files and nothing more.
  For example a plain `git push`, `alembic upgrade head`, `python manage.py migrate`,
  `ssh prod '...'`, `sendmail` or `aws ses send-email`.

Check any command before you trust it:

```bash
vcr airbag check "alembic upgrade head"
# ✎ WRITE: changes files — snapshot first, undo with `vcr undo`
vcr airbag check "git push --force origin main"
# ⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
```

For side effects inside your own agent code, the experimental
[workflow engine](experimental/workflow.md) can run a compensating action ("cancel the booking"),
but only one you write.

### How do I turn the airbag off?

`AGENT_VCR_AIRBAG=off` disables it. `vcr airbag uninstall` removes its hook entries and leaves
your own hooks alone. Other settings are in [Airbag → Configuration](airbag.md#configuration).

## Platforms and versions

### Does it work on Windows? macOS?

Yes. Every change runs the full test suite on **Linux (Python 3.9–3.13), macOS and Windows**.
What's different on Windows:

- The airbag store lives in `%LOCALAPPDATA%\agent-vcr\airbag`, and it's locked against
  concurrent hook processes with `msvcrt` (`flock` on Linux and macOS).
- Hook commands are written with forward slashes, quoted only when needed, so they run in Git
  Bash and `cmd`. Copilot CLI's PowerShell hook uses `& '...'`.
- A `.git` backup is a copy, not a hardlink, so your own `.git` is never touched.
- `vcr airbag install cline` refuses to install on Windows; Cline hooks are POSIX shell scripts.
- The command classifier parses POSIX shell syntax. Windows paths (`C:\...`, backslashes) and
  `%TEMP%` are understood, but PowerShell- or `cmd`-only syntax isn't analysed.
- NTFS has no Unix permission bits, so "private files stay 0600 after restore" doesn't apply.
- `vcr bisect` and the local sandbox kill a timed-out run's whole process tree with
  `taskkill /T /F`.

### Which Python versions?

3.9 to 3.13. All five are tested in CI. The airbag and the ACID workspace also need `git` on
your `PATH`.

### Is it production-ready?

Some layers are, some aren't (the package as a whole is classified *Beta* on PyPI):

| Layer | Status |
|---|---|
| Recording kernel (`recorder`, `player`, `execution/v1` format) | Stable. Old recordings always load. |
| Airbag, ACID workspace, Sentinel | Stable API, actively developed |
| Ghost Replay, datasets, Agent CI, bundles | Stable |
| LangGraph, CrewAI, OpenHands, Claude Code integrations | Stable, tested against the real libraries |
| Dashboard and self-hosted store (`vcr-server`) | Supported |
| SuperAgent, workflows, sandbox, intelligence | **Experimental.** APIs may change |

The full map is in [Architecture](architecture.md).

## Agents and the project

### Which coding agents are supported? Can I add one?

Claude Code, OpenAI Codex CLI, Gemini CLI, Cursor, GitHub Copilot CLI, OpenHands, OpenCode 1.x,
Cline and Devin Local get real pre-tool hooks. Anything else can run under
`vcr airbag run -- <agent>`, which takes a snapshot every 2 s but can't stop a command before it
runs. The table is in [Airbag → Supported agents](airbag.md#supported-agents). To add an agent
(parse its payload, render its answer, install, test), follow
[CONTRIBUTING.md → Adding an agent to the airbag](../CONTRIBUTING.md#adding-an-agent-to-the-airbag).

### Can I use it in my own Python agent, not a coding agent?

Yes. Record with `VCRRecorder` ([Recording and replay](recording-and-replay.md)), roll back a repo
plus agent state with `ACIDWorkspace` ([ACID workspace](acid-workspace.md)), and gate releases
with [Agent CI](agent-ci.md).

### I ran `pip install agent-vcr` and got something else.

That's a different project by a different developer. This one is **`ai-agent-vcr`**:

```bash
pipx install ai-agent-vcr      # the vcr command
pip install ai-agent-vcr       # the Python library, in a venv
```

The import is still `agent_vcr`, and the CLI is `vcr`.

### `pip install` says "externally-managed-environment".

Your system Python (most current Linux distros, Homebrew) doesn't allow `pip install` into it.
For the `vcr` command, use `pipx install ai-agent-vcr` or `uv tool install ai-agent-vcr`: it
gets its own environment and works in every project. For the Python library, install it inside
your project's virtual environment.

### `vcr` fails with `ModuleNotFoundError: No module named 'agent_vcr'`.

An old `pip install --user` left a `vcr` launcher behind without its package. Remove the old
install and reinstall as a tool:

```bash
python3 -m pip uninstall --break-system-packages ai-agent-vcr
pipx install ai-agent-vcr
```

### What's the license?

[MIT](../LICENSE).

**Next:** [Quickstart](quickstart.md) · [Airbag](airbag.md) · [Architecture](architecture.md)
