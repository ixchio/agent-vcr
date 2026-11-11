# Airbag — undo anything your coding agent did, including Bash

The airbag snapshots your project **before every agent tool call** — shell commands
included — so one command brings everything back:

```bash
vcr airbag install          # hooks for every agent found on your PATH
# ...agent runs `rm -rf .`, `git reset --hard`, `sqlite3 app.db 'DROP TABLE users'`...
vcr undo                    # back to just before it
vcr redo                    # changed your mind
```

Built-in checkpoints in coding agents only track their own file-edit tools. The
airbag sees `rm`, `mv`, `sed -i`, redirects, `git clean`, migrations against a
local SQLite file, and edits made by subagents — anything that reaches a hook.

For commands a snapshot **cannot** undo — remote databases, `terraform destroy`,
`kubectl delete`, force-pushes, `npm publish`, writes outside the project — it
stops and asks a human first.

```bash
vcr airbag check "psql -h prod.db -c 'DROP TABLE users'"
# ⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
#    • SQL DROP on a remote database (prod.db) — server data is not in your project files…
```

## Supported agents

| Agent | Install | Snapshot before | Irreversible commands |
|---|---|---|---|
| Claude Code | `vcr airbag install claude` → `.claude/settings.json` | Bash, Write, Edit, MultiEdit, NotebookEdit | **asks** (permission prompt); headless `-p` runs see a denial |
| OpenAI Codex CLI | `vcr airbag install codex` → `.codex/hooks.json` | Bash, apply_patch | **blocked** (Codex has no "ask"); approve with `vcr airbag allow` |
| Gemini CLI | `vcr airbag install gemini` → `.gemini/settings.json` | run_shell_command, write_file, replace | **asks** on ≥ 0.36, otherwise blocked |
| Cursor (IDE + CLI) | `vcr airbag install cursor` → `.cursor/hooks.json` | shell commands, Write/Delete | **blocked** (Cursor's "ask" is not reliably enforced) |
| GitHub Copilot CLI | `vcr airbag install copilot` → `.github/hooks/agent-vcr-airbag.json` | bash, create, edit, apply_patch | **asks**; autopilot / cloud agent turn it into a denial |
| OpenHands (SDK) | `vcr airbag install openhands` → `.openhands/hooks.json`, or in-process (below) | terminal, file_editor, apply_patch | **blocked**, or **asks** with `protect(conversation, confirm=True)` |
| OpenCode 1.x | `vcr airbag install opencode` → `.opencode/plugin/agent-vcr-airbag.js` | bash, edit, write, apply_patch | **blocked** (plugins cannot ask); OpenCode 2.x plugin API not supported yet |
| Cline | `vcr airbag install cline` → `.clinerules/hooks/PreToolUse` | run_commands, editor, apply_patch | **task cancelled** (Cline has no "ask") |
| Devin Local / Devin Desktop (ex-Windsurf) | `vcr airbag install devin` → `.devin/hooks.v1.json` | exec, write, edit, apply_patch | **blocked** |
| Anything else (Aider, scripts, …) | `vcr airbag run -- aider` | periodic snapshot every 2 s | — (no interception) |

`--user` installs into the user-level config instead of the project.
`vcr airbag uninstall [agent]` removes only the airbag's entries; your own hooks stay.

Notes per agent:

* **Codex** asks you to review new hooks once: run `/hooks` inside Codex after installing.
* **Gemini CLI** only runs hooks in trusted folders.
* **Copilot CLI** loads hooks at startup — restart running sessions.
* **Cursor, Copilot CLI and Devin also run `.claude/settings.json` hooks.** Installing
  several is fine: one tool call reaching the airbag twice within 5 seconds is handled once.
* **OpenHands SDK 1.7.x** does not auto-load `.openhands/hooks.json`; pass
  `hook_config=HookConfig.load(working_dir=ws)` to `Conversation`, or use the in-process
  integration:

```python
from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect

airbag = AirbagCallback()
conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag])
protect(conversation, airbag)                # blocks irreversible actions
# protect(conversation, airbag, confirm=True)  # pause for a human instead — only if your
#                                              # loop handles WAITING_FOR_CONFIRMATION
```

## Commands

| Command | What it does |
|---|---|
| `vcr undo [N]` | Restore files to just before the last (N-th) agent action that changed something. `--dry-run` shows the plan. |
| `vcr redo` | Undo the last undo/rewind. |
| `vcr rewind` / `vcr airbag log` | Timeline of protected actions with snapshot numbers. |
| `vcr rewind 12` | Restore snapshot #12 (itself undoable with `vcr redo`). |
| `vcr airbag check "<cmd>"` | Classify a command without running it. |
| `vcr airbag allow` | Approve the last stopped command for one retry within 10 minutes. |
| `vcr airbag snapshot -m "msg"` | Manual snapshot. |
| `vcr airbag status` | Policy, store location, installed hooks. |
| `vcr airbag gc --keep-days 7` | Delete old snapshots. |

## How it works

* **Shadow git, outside your project.** Each project gets a private git directory under
  `~/.local/state/agent-vcr/airbag/` (macOS: `~/Library/Application Support/…`,
  Windows: `%LOCALAPPDATA%`; override with `AGENT_VCR_AIRBAG_HOME`). Deleting the whole
  project cannot delete its snapshots. Your own `.git` — HEAD, branches, index, stash,
  hooks — is never read or written.
* **What is protected:** every file git would see (tracked *and* untracked), honoring your
  `.gitignore`, plus small ignored "precious" files: `.env*`, `*.db`, `*.sqlite*`.
  Dependency trees and build output (`node_modules/`, `.venv/`, `dist/`, `target/`, …) and
  files over 25 MB are skipped — they are regenerable or too large.
* **SQLite** files are copied with SQLite's online-backup API, so a snapshot is consistent
  even while the database is open in WAL mode.
* **Your git history** is preserved for commands that would delete it (`rm -rf` of the
  project or of anything in `.git`): `.git` is copied as hardlinks (near-instant, no extra
  space). `vcr undo` puts it back when it is missing, broken, or replaced by a fresh
  `git init`; a damaged `.git` is moved into the store, never deleted. A worktree's `.git`
  pointer file is restored too. A working repository is never overwritten.
* **Bytes and permissions are exact:** line-ending/LFS attributes of the project do not
  apply inside the store, and private files (e.g. a `0600` `.env` or key) come back with
  their permissions.
* **Snapshots are cheap:** unchanged files are not re-read; identical states share one
  snapshot. Measured on a 180-file repo: 12 ms of airbag work when nothing changed,
  ~190 ms for the very first snapshot, and ~0.1 s per tool call for the whole hook
  process (most of it is Python start-up).
* **Never grants permission.** The airbag only adds a snapshot and a notice, or escalates
  (ask / block). It never answers "allow" in a way that skips your agent's own approval
  rules (Cursor requires an answer; there "allow" means "no objection").
* **Fails open for its own errors.** If a snapshot fails (disk full, git missing), the tool
  call proceeds and you see a warning — except irreversible commands, which are still
  escalated.

## What it cannot undo

* Anything outside the project directory (other folders, `~`, system files).
* Remote state: databases on a server (including a local Postgres/MySQL server — its data
  is not in your project), cloud resources, pushed commits, published packages, sent
  requests. These are classified *irreversible* and stopped for confirmation.
* Ignored dependency/build directories and files over the size limit (25 MB,
  `AGENT_VCR_AIRBAG_MAX_FILE_MB`). A restore never overwrites such a file — it is left as
  it is and listed as "left untouched".
* Nested git repositories (submodules, cloned dependencies): git records only a pointer,
  not their files. Deleting one is therefore treated as irreversible and stopped for
  confirmation.
* Empty directories (git does not track them).
* Commands an agent runs without a hook (use `vcr airbag run --` for those, which snapshots
  every 2 seconds and cannot intercept a command before it runs).

The classifier is a heuristic guard, not a sandbox: an agent determined to hide a command
(for example inside a script it just wrote) can get past it. The snapshot still happens,
so the project files remain undoable.

## Configuration

| Variable | Effect |
|---|---|
| `AGENT_VCR_AIRBAG=off` | Disable entirely. |
| `AGENT_VCR_AIRBAG_MODE=risky` | Snapshot only before destructive/irreversible commands (default `every`). |
| `AGENT_VCR_AIRBAG_ON_IRREVERSIBLE=ask\|deny\|warn` | What to do with commands a snapshot cannot undo (default `ask`). |
| `AGENT_VCR_AIRBAG_HOME=/path` | Where snapshot stores live. |
| `AGENT_VCR_AIRBAG_MAX_FILE_MB=25` | Files larger than this are not snapshotted. |
| `AGENT_VCR_AIRBAG_MAX_FILES=200000` | Refuse to protect trees with more files than this. |

## Sources for the hook contracts

Verified on 2026-11-11 against official docs and, where installed, the agents' own code:
Claude Code — code.claude.com/docs/en/hooks · Codex CLI 0.154 — developers.openai.com/codex and
`openai/codex` source · Gemini CLI 0.37.1 — installed package source and bundled docs ·
Cursor — cursor.com/docs/hooks · Copilot CLI — docs.github.com/en/copilot/reference/hooks-reference ·
OpenHands SDK 1.7.3 — installed source · Cline — `cline/cline` repository ·
Devin Local — docs.devin.ai. Payload formats for Cursor, Copilot, Cline and Devin were tested
with documented sample payloads, not live agent sessions.

## Safety review

Before release the restore path was attacked by an independent reviewer (file growth past the
size limit, overwriting "read-only" commands, nested repositories, permissions, line endings,
damaged `.git`, worktrees, odd file names, symlinks, concurrency, hostile hook payloads). Every
finding has a regression test in `tests/unit/test_airbag_core.py`.
