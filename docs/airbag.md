# 🛟 Airbag — undo anything your coding agent did, shell commands included

A save point before every tool call your coding agent makes, and one command to go back.

It's 6 p.m. You've spent the afternoon on a billing fix and haven't committed it yet. You ask
your agent to "clean up the build folder". It runs `rm -rf .` in the wrong directory. Your fix is
gone. Your `.env` is gone. `.git` is gone too, so there's nothing to `git checkout`.

You hit the agent's rewind button. Nothing happens. **Built-in checkpoints in coding agents only
track their own file-edit tools.** They never saw that shell command.

The airbag did see it. It took a snapshot just before the command ran:

```bash
vcr undo
# ↩ Undid: rm -rf /home/you/myapp  (claude, 4s ago)
#   restored 5 files — back to snapshot #1.
#     + .env
#     + src/billing.py
#     + .git/
```

**TL;DR**

- `vcr airbag install`, once, hooks every coding agent on your PATH at the user level. From
  then on, in every git repository you work in, every tool call (Bash, file edits, patches,
  subagents) is snapshotted first, into a store outside your project. Nothing is written into
  your repositories, so nothing can break a teammate.
- `vcr undo`, `vcr redo` and `vcr rewind` bring files, `.env`, local SQLite databases and even
  `.git` back exactly as they were.
- Commands a snapshot *can't* undo (a remote `DROP TABLE`, `terraform destroy`, force-push,
  `npm publish`) are stopped so a human can say yes or no first.

## Try it in 60 seconds

No agent, API key or account needed. The demo builds a project with an uncommitted fix and a
`.env`, feeds the airbag the exact hook payload Claude Code sends before `rm -rf`, deletes the
project, and runs `vcr undo`:

```bash
pip install ai-agent-vcr
bash examples/airbag_demo.sh      # from a clone of the repo
```

You'll see the code, the `.env` and the git history come back, and then a remote `DROP TABLE`
getting stopped. To arm it for real (once per machine, not per project):

```bash
pipx install ai-agent-vcr   # or: pip install ai-agent-vcr
vcr airbag install          # hooks for every supported agent found on your PATH
# 🛟 Airbag armed for 2 agents, user level: every project, nothing written into your repository.
#   ✓ Claude Code       ~/.claude/settings.json
#   ✓ OpenAI Codex CLI  ~/.codex/hooks.json
# …
# Required once:
#   • Codex: run /hooks in Codex once and trust the airbag hook (Codex skips new or changed hooks until you do).
# Try it: vcr airbag check "rm -rf ."   ·   undo anything: vcr undo   ·   health check: vcr doctor
vcr doctor                  # is everything really armed? prints the exact fix if not
```

Do the "Required once" steps it prints: Codex runs a new hook only after you trust it with
`/hooks`, and Gemini CLI runs hooks only in folders you trusted with `/trust`.

Ask "would this be stopped?" without running anything:

```bash
vcr airbag check "psql -h prod.db -c 'DROP TABLE users'"
# ⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
#    • SQL DROP on a remote database (prod.db) — server data is not in your project files, so the airbag cannot undo it
```

## Supported agents

Install for one agent by name, several at once (`vcr airbag install claude codex`), or all of
them (`--all`). With no names, it installs for every supported agent it finds on your PATH.

Every agent below reads hooks from a **user-level** file, and that's where the airbag goes by
default: one install covers every project, and nothing appears in `git status`.

| Agent | Installs into (user level) | Snapshot before | Irreversible commands |
|---|---|---|---|
| Claude Code | `~/.claude/settings.json` | Bash, Write, Edit, MultiEdit, NotebookEdit | **asks** (permission prompt); headless `-p` runs see a denial |
| OpenAI Codex CLI | `~/.codex/hooks.json` | Bash, apply_patch | **blocked** (Codex has no "ask"); approve with `vcr airbag allow` |
| Gemini CLI | `~/.gemini/settings.json` | run_shell_command, write_file, replace | **asks** on ≥ 0.36, otherwise blocked |
| Cursor (IDE + CLI) | `~/.cursor/hooks.json` | shell commands, Write/Delete | **blocked** (Cursor's "ask" is not reliably enforced) |
| GitHub Copilot CLI | `~/.copilot/hooks/agent-vcr-airbag.json` (`$COPILOT_HOME`) | bash, create, edit, apply_patch | **asks**; autopilot / cloud agent turn it into a denial |
| OpenHands (SDK) | `~/.openhands/hooks.json`, or in-process (below) | terminal, file_editor, apply_patch | **blocked**, or **asks** with `protect(conversation, airbag, confirm=True)` |
| OpenCode 1.x | `~/.config/opencode/plugin/agent-vcr-airbag.js` | bash, edit, write, apply_patch | **blocked** (plugins cannot ask); approve with `vcr airbag allow`. OpenCode 2.x plugin API not supported yet |
| Cline | `~/Documents/Cline/Hooks/PreToolUse` (your XDG Documents folder) | run_commands, editor, apply_patch | **task cancelled** (Cline has no "ask") |
| Devin Local / Devin Desktop (ex-Windsurf) | `~/.config/devin/config.json` (`%APPDATA%\devin\config.json` on Windows) | exec, write, edit, apply_patch | **blocked** |
| Anything else (Aider, scripts, …) | `vcr airbag run -- aider` | periodic snapshot every 2 s | — (no interception) |

`vcr airbag uninstall [agent]` removes only the airbag's entries, at both levels; your own
hooks and settings stay. `--user` or `--project` limits it to one level.

**Only this project, only for you: `--project`.** It writes the agent's project file instead
(`.codex/hooks.json`, `.cursor/hooks.json`, …; for Claude Code, its personal
`.claude/settings.local.json`, never the shared `.claude/settings.json`):

```bash
vcr airbag install --project claude
# 🛟 Airbag armed for 1 agent in this project only.
#   ✓ Claude Code  ~/myapp/.claude/settings.local.json
#   ⚠ These files are per-developer (they hold this machine's Python path): don't commit them; add them to your .gitignore.
```

The airbag never edits your `.gitignore` or `.git/info/exclude`; ignoring the file is your call.

### ⚠️ Agent gotchas

- **Codex** runs a new or changed hook only after you trust it: run `/hooks` inside Codex after
  installing (and again after `vcr doctor --fix` rewrote it).
- **Gemini CLI** only runs hooks in trusted folders (`/trust`), user-level hooks included.
- **Claude Code** holds back all hooks, user-level ones too, until you've accepted a folder's
  trust prompt (the one it shows the first time you open a folder).
- **Copilot CLI** loads hooks at startup, so restart sessions that are already running.
- **Cline** cancels the whole task on an irreversible command. Cline CLI reads
  `.cline/hooks/PreToolUse.sh` instead of the global hooks folder. Cline on Windows wants
  PowerShell `.ps1` hooks, which the airbag doesn't install; use `vcr airbag run --` there.
- **OpenHands** reads `~/.openhands/hooks.json` only when the project has no
  `.openhands/hooks.json` of its own. `vcr doctor` tells you when a project file hides yours.
- **Cursor, Copilot CLI and Devin also run `.claude/settings.json` hooks.** Installing several
  is fine: when one tool call reaches the airbag twice within 5 seconds, it's handled once.
- **The hook calls the Python that ran `vcr airbag install`** (its full path is in the config).
  On Linux and macOS the command is guarded (`test -x <python> && exec <python> … || true`): if
  that Python disappears (pipx reinstall, upgrade, a deleted virtualenv), the hook quietly does
  nothing instead of failing every tool call. `vcr doctor` notices, and `vcr doctor --fix` (or
  `vcr airbag install` again) rewrites the path. On Windows the command isn't guarded, because
  agents there run it through Git Bash, cmd or PowerShell; run `vcr doctor` after moving Python.
- **OpenHands SDK 1.7.x** does not load `.openhands/hooks.json` by itself. Pass
  `hook_config=HookConfig.load(working_dir=ws)` to `Conversation`, or use the in-process
  integration:

```python
from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect

airbag = AirbagCallback()
conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag])
protect(conversation, airbag)                  # blocks irreversible actions
# protect(conversation, airbag, confirm=True)  # pause for a human instead
```

`confirm=True` pauses the run in `WAITING_FOR_CONFIRMATION`. Only use it if your loop handles
that state: calling `conversation.run()` again **approves** the paused actions.

## How it works

```text
 agent wants to run a tool ──► hook ──► classify the command
                                          │
            read ───────────────────────► run it (no snapshot needed)
            write / destructive ────────► snapshot project ──► run it
            irreversible ───────────────► snapshot ──► ask / block a human
                                          │
                     shadow git store (outside your project, never your .git)
                                          ▲
                vcr undo · vcr redo · vcr rewind <#> restore from here
```

- **Shadow git, outside your project.** Each project gets a private git directory (mode `0700`)
  under `~/.local/state/agent-vcr/airbag/` on Linux (or `$XDG_STATE_HOME`),
  `~/Library/Application Support/agent-vcr/airbag/` on macOS, and `%LOCALAPPDATA%\agent-vcr\airbag`
  on Windows. Override it with `AGENT_VCR_AIRBAG_HOME`. Deleting the whole project can't delete
  its snapshots.
- **Your `.git` is never touched.** HEAD, branches, index, stash and hooks are never read or
  written by a snapshot.
- **Your git history survives `rm -rf`.** `.git` is backed up as hardlinks (near-instant, no extra
  space). `vcr undo` puts it back when it's missing, broken, or replaced by a fresh `git init`; a
  damaged `.git` is moved into the store, never deleted. A worktree's `.git` pointer file comes
  back too. A working repository is never overwritten.
- **SQLite is copied with SQLite's online-backup API**, so a snapshot is consistent even while
  the database is open in WAL mode.
- **Exact bytes and permissions.** The project's line-ending and LFS attributes don't apply inside
  the store, and private files (say a `0600` `.env` or key) come back with their permissions.
- **Cheap.** Unchanged files aren't re-read, and identical states share one snapshot. Measured on
  a 180-file repo: 12 ms of airbag work when nothing changed, ~190 ms for the very first snapshot,
  and ~0.1 s per tool call for the whole hook process (mostly Python start-up).
- **Never grants permission.** The airbag only adds a snapshot and a notice, or escalates (ask or
  block). It never answers "allow" in a way that skips your agent's own approval rules. (Cursor
  requires an answer; there "allow" means "no objection".)
- **Fails open on its own errors.** If a snapshot fails (disk full, git missing), the tool call goes
  ahead and you see a warning. Irreversible commands are still escalated.

### The four levels

Every shell command gets one of four levels. Try any command with `vcr airbag check`:

| Level | Meaning | Example |
|---|---|---|
| `read` | No side effects; no snapshot needed | `git status`, `ls`, `grep` |
| `write` | Changes project files; snapshot first | `cat a.txt \| grep a > out.txt` |
| `destructive` | Deletes or overwrites project data; snapshot makes it undoable | `rm -rf src`, `git reset --hard`, `sed -i`, `sqlite3 app.db 'DROP TABLE users'` |
| `irreversible` | A snapshot can't undo it; a human decides | remote `psql … DROP`, `terraform destroy`, `kubectl delete`, `git push --force`, `npm publish`, `rm -rf ~/Documents` |

```bash
vcr airbag check "git reset --hard"
# 🛟 DESTRUCTIVE: destroys project data — snapshot first, undo with `vcr undo`
#    • git reset --hard discards uncommitted changes

vcr airbag check "sqlite3 app.db 'DROP TABLE users'"
# 🛟 DESTRUCTIVE: destroys project data — snapshot first, undo with `vcr undo`
#    • SQL DROP on app.db (file snapshotted first)

vcr airbag check "rm -rf ~/Documents"
# ⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
#    • rm deletes files
#    • deletes ~/Documents outside the project (snapshots only cover the project)

vcr airbag check --json "git reset --hard"
# {"level": "destructive", "reasons": [...], "categories": ["files"], "undoable": true, "protected": true}
```

Run these inside a git repository. In a folder the airbag doesn't protect, deleting a
directory tree becomes `irreversible` (there's no snapshot to undo it with) and other
destructive commands go ahead without a snapshot ([which folders](#which-folders-are-protected)).

The classifier understands pipelines, `&&`, `sudo`, `bash -c "..."` and redirects, and leans
towards `write` (snapshot) when it isn't sure.

### What's protected

- Every file git would see in your project, tracked **and** untracked, honoring your `.gitignore`.
- Small ignored "precious" files on top: `.env`, `.env.*`, `*.env`, `*.db`, `*.sqlite`,
  `*.sqlite3`, `*.db3`.
- Your `.git` directory (via the hardlink backup above).

Skipped on purpose: dependency trees and build output (`node_modules/`, `.venv/`, `dist/`,
`target/`, …) because they're regenerable, and files over 25 MB because they're too large.

### Which folders are protected

User-level hooks fire wherever your agent runs: in a repository, but also in `~`, in `/` or in
a downloads folder. Snapshotting those would be slow and pointless, so the rule is:

| Where the agent works | Snapshots | Deleting a tree (`rm -rf old`, `git clean -fdx`) | Other changes (`mv`, `sed -i`, `>`, one `rm`, edits) | Irreversible command |
|---|---|---|---|---|
| A git repository (any subfolder) | ✓ before every change | runs, undo with `vcr undo` | runs, undo with `vcr undo` | stopped for a human |
| A folder you opted in with `vcr airbag protect` (or one below it) | ✓ | runs, undo with `vcr undo` | runs, undo with `vcr undo` | stopped for a human |
| A folder that already has snapshots (say, the agent deleted `.git`) | ✓ | runs, undo with `vcr undo` | runs, undo with `vcr undo` | stopped for a human |
| Anything else: a plain folder, `~`, `/`, `~/Desktop`, `~/Documents`, `~/Downloads` | never | **stopped for a human**: no snapshot, so it can't be undone | runs, with a notice: "not protected here (not a git repo) — `vcr airbag protect` to enable snapshots" | stopped for a human |

"Deleting a tree" means `rm -r`/`-rf` on a directory (or the folder itself), `rm` with a glob,
`git clean`, `git reset --hard`, `find … -delete` and `rsync --delete`. Agents that can't ask
(Codex, Cursor, Devin, OpenCode, Cline) block those until `vcr airbag allow`; everything else
in an unprotected folder goes through, so working outside a repository isn't a wall of prompts.

Your home directory, the filesystem root and Desktop/Documents/Downloads are never snapshotted
and can't be opted in. A dotfiles repository at `~` doesn't count as a project, but it doesn't
hide one either: a protected folder or an existing store below it still wins. An agent started
in `~` that works inside a repository is protected: the hook looks at the command's own folder
too.

`vcr airbag protect [DIR]` writes `DIR/.vcr/airbag.json`, and `vcr airbag unprotect` removes it.
A bare `.vcr/` folder doesn't count: the recorder and `vcr init` create one too, and it
shouldn't silently widen what gets snapshotted. `vcr doctor` points out a bare `.vcr/` that
isn't an opt-in. `vcr airbag check` and `vcr airbag status` follow the same rule:

```bash
cd ~/notes                     # not a git repository
vcr airbag check "rm -rf build"
# ⚠ IRREVERSIBLE: cannot be undone by a snapshot — the airbag asks a human first
#    • rm deletes files
#    • this folder is not protected by the airbag (not a git repo), so nothing was snapshotted and this cannot be undone (`vcr airbag protect` to enable snapshots)
vcr airbag check "mv a.txt b.txt"
# 🛟 DESTRUCTIVE: goes ahead, but this folder is not protected: no snapshot, no `vcr undo`
#    • mv can overwrite files
#    • not protected here (not a git repo) — `vcr airbag protect` to enable snapshots
vcr airbag status
# Airbag for /home/you/notes
#   ✗ not protected: /home/you/notes (not a git repository, not opted in). Nothing is snapshotted here; deleting a directory tree is stopped for a human, other changes go ahead with a notice. To protect it, run `vcr airbag protect` (or `git init`) in /home/you/notes.
#   …
vcr airbag protect
# 🛟 /home/you/notes is now protected: agents' changes there are snapshotted first, undo with `vcr undo`. Opt-in file: /home/you/notes/.vcr/airbag.json (`vcr airbag unprotect` removes it).
vcr airbag unprotect
# Removed /home/you/notes/.vcr/airbag.json.
```

If the folder already has snapshots, `vcr airbag unprotect` says so: it stays protected (so
`vcr undo` keeps working) until you delete its store, whose path it prints.

`vcr airbag allow` works in unprotected folders too: approvals there are kept in one shared
place in the airbag store.

## Everyday recipes

### Undo the last thing, or the last few

```bash
vcr undo             # back to just before the last agent action that changed something
vcr undo 3           # three actions back
vcr undo 2 --dry-run # show what would change, touch nothing
# Would undo: unlink a.txt  (claude, 0s ago) → snapshot #2
#     + a.txt
#     - c.txt
```

`--dry-run` previews exactly what the undo will do: `+` comes back, `-` gets removed (it was
created after that point), `~` is put back to its old content. Same signs as the real undo.

### Jump to any point in time

```bash
vcr rewind           # timeline (same as `vcr airbag log`)
#    snap      when  agent          command
#      #1    2m ago  claude       ✎ echo b > b.txt
#      #2    2m ago  claude       🛟 unlink a.txt
#      #3    1m ago  claude       ✎ echo c > c.txt
vcr rewind 1         # restore snapshot #1 (also accepts "#1")
```

### Changed your mind

```bash
vcr redo             # undoes the last undo or rewind
```

Undo and rewind are undoable themselves, so trying one is safe.

### Let one stopped command through

When an agent that can't "ask" (Codex, OpenCode, Cursor, …) gets blocked, check the command, then:

```bash
vcr airbag allow
# ✅ Approved once: git push --force origin main
#    The agent may retry this exact command in the next 10 minutes.
```

### Is it really armed? `vcr doctor`

```bash
vcr doctor
#   ✓ Agent VCR 1.7.1 on Python 3.12.3
#       pipx: /home/you/.local/share/pipx/venvs/ai-agent-vcr/bin/python
#   ✓ ~/.local/bin/vcr → Agent VCR 1.7.1
#       runs when you type `vcr`
#   ✓ git version 2.43.0
#   ✗ Claude Code: hook calls a Python that no longer exists
#       ~/.claude/settings.json → /home/you/.local/share/pipx/venvs/old/bin/python3
#       fix: vcr doctor --fix   (or re-run `vcr airbag install`)
#   ✓ OpenAI Codex CLI: protected (user-level)
#   · Codex runs new or changed hooks only after you trust them
#       fix: run /hooks in Codex and trust the Agent VCR hook
#   ✓ This folder is protected
#   · 12 snapshots for this project, 3.4 MB
# ✗ 1 problem; 1 can be fixed with `vcr doctor --fix`.
vcr doctor --fix
# 🔧 rewrote the Claude Code hook for /home/you/.local/share/pipx/venvs/ai-agent-vcr/bin/python3: ~/.claude/settings.json
# …
# ✓ Healthy.
```

It checks, each with the exact fix: the Python and Agent VCR version and how it's installed
(pipx, virtualenv, `pip --user`, editable); every `vcr` launcher on your PATH, each one *run*
with `--version`, so a stale one (say, a `~/.local/bin/vcr` left over from an old install whose
package is gone) is flagged, and removing a launcher is only suggested when running it fails;
git; for each agent on your PATH or with hooks: where the hooks are (user or project level),
whether the Python in the hook command still exists, and the one-time steps (Codex `/hooks`,
Gemini `/trust` and version ≥ 0.36 for "ask", Cline unsupported on Windows, an OpenHands
project file hiding yours); the snapshot store of the current project (location, size, count);
and whether the current folder is protected (a bare `.vcr/` that isn't an opt-in is pointed
out, with `vcr airbag protect` as the fix).

`vcr doctor --fix` only does what is safe to do automatically: it rewrites airbag hook
commands whose Python is gone (or that predate the self-healing form) and moves project-level
airbag hooks to the user level. A move writes the user-level hook first and checks it; only
then is the airbag's entry taken out of the project file. If your user-level file can't be
written (invalid JSON, read-only), it says which file and why, and changes nothing. It never
deletes a launcher or touches anyone else's hooks; it removes a folder like `.claude/` only
when the airbag's file was the last thing in it.
`--json` prints `{"healthy": …, "checks": [{"id", "status", "title", "detail", "fix",
"fixable"}], "fixed": [...]}`. Exit code: `0` healthy, `1` problems. `vcr airbag doctor` is the
same command.

### The airbag in Claude Code's status line

```bash
echo '{"workspace": {"current_dir": "'"$PWD"'"}}' | vcr airbag statusline
# 🛟 airbag · 12 snapshots · last: rm -rf build (2m ago)
cd ~ && vcr airbag statusline
# 🛟 airbag off here (your home directory)
```

It reads the JSON Claude Code passes on stdin, takes no snapshot and runs in about 60 ms. The
airbag never changes your status line by itself. Add it to `~/.claude/settings.json`:

```json
{"statusLine": {"type": "command", "command": "vcr airbag statusline"}}
```

…or let the installer add it, only if you have no `statusLine` yet (an existing one is never
overwritten):

```bash
vcr airbag install claude --statusline
# Status line: Claude Code now shows the airbag (`vcr airbag statusline`).
```

### Check health, take a manual save point, tidy up

```bash
vcr airbag status                        # protection here, policy, store location, hooks
vcr airbag snapshot -m "before refactor" # your own save point
vcr airbag gc --keep-days 7              # delete old snapshots (keeps the last 50 by default)
```

### Agent without hooks? Wrap it

```bash
vcr airbag run -- aider
```

This takes a snapshot when the agent starts, every 2 seconds while it runs (`--interval` to
change; a no-op when nothing changed), and when it exits. It can't stop a command *before* it
runs, so up to one interval of work can be lost. Use a real hook integration when one exists.

### Uninstall

```bash
vcr airbag uninstall codex   # one agent
vcr airbag uninstall         # every agent, user and project level; your own hooks are left alone
```

### Upgrading from 1.7.0

1.7.0 installed hooks into the project (`.claude/settings.json`, `.codex/hooks.json`, …) with
your machine's Python path in them. Committed, every teammate's agent then runs a hook that
fails on each tool call. Run the installer once after upgrading:

```bash
pipx upgrade ai-agent-vcr
cd your-project && vcr airbag install
# 🛟 Airbag armed for 2 agents, user level: every project, nothing written into your repository.
#   …
# Moved from the project to your user-level config (commit the removal, so teammates' agents stop calling a path from your machine):
#   ↪ moved the OpenAI Codex CLI hook from .codex/hooks.json to ~/.codex/hooks.json
```

It removes only the airbag's entries from the project files (your other hooks and settings
stay; a file that held nothing else is deleted) and installs the same agents at the user level.
Commit that change. `vcr doctor --fix` does the same move for every project you run it in.

Every command takes `--project-dir PATH` if you're not inside the project, and most take `--json`.

## Find the action that broke it: `vcr bisect`

Your tests passed at lunch. The agent has made 40 tool calls since, nothing is committed, and
now `pytest` is red. Which call did it? The airbag has a snapshot from just before every one of
them, so it can find out the way `git bisect` would: no model tokens, no guessing.

```bash
vcr bisect -- pytest -q
# 🔍 Bisecting the airbag timeline of /home/you/myapp
#    test: pytest -q   (each run in a temporary checkout)
#   #6 (current)   ✗ fail   (0.4s)
#   #1             ✓ pass   (0.4s)
#   #3             ✓ pass   (0.4s)
#   #4             ✗ fail   (0.4s)
#
# 🎯 First bad action: sed -i 's/round(sum(prices), 2)/int(sum(prices))/' billing.py
#    claude · Bash · 2026-11-12 12:50:24 (2s ago)
#    snapshot #3 passes → #4 fails   (4 test runs)
#    files it changed (+ added, - deleted, ~ modified):
#      ~ billing.py
#    failing run (last lines):
#      E       assert 3 == 3.75
#      …
#    Go back to just before it:  vcr rewind 3
```

`vcr rewind 3` puts the project back to just before that action, and `vcr redo` takes you
forward again if you'd rather fix it by hand. Exit code: `0` when it found the action, `1` when
there was nothing to find, `2` for a usage error.

**How it works.** The timeline is the one `vcr rewind` shows. Each action turns its
before-snapshot into the next action's before-snapshot (the last one into the current state,
which is snapshotted first). `vcr bisect` checks out a snapshot into a **temporary directory**,
runs your command there (cwd = that checkout), and binary-searches for the first action whose
before-state passes and whose after-state fails. Six states took four runs above; 100 actions
take about nine. States with identical files are tested once. Your project, its `.git` and
the existing snapshots are never modified, and the checkouts are deleted afterwards.

```bash
vcr bisect --good 2 --bad 5 -- pytest -q        # only search between snapshots #2 and #5
vcr bisect --timeout 60 -- pytest -q            # per run; a timeout counts as a failure
vcr bisect -- "pytest -q && ruff check ."       # one quoted string runs through the shell
vcr bisect --copy-ignored -- pytest -q          # bring node_modules/.venv into each checkout
vcr bisect --copy-ignored --copy-ignored dist -- pytest -q  # …and also link dist/
vcr bisect --json -- pytest -q                  # status, culprit, snapshots, runs, changes, rewind
```

Without `--good`, the search starts at the oldest recorded action; without `--bad`, it ends at
the current state. If the test already fails at the good end, or still passes at the bad end,
it tells you so and stops instead of guessing. A run that hits `--timeout` (default 300 s) is
killed with its whole process group, counted as a failure, and the report says so. Background
processes a test leaves behind (`server & pytest`) don't count against it: the run is over when
the command exits, and the leftovers are killed.

**Limits.**

- **Ignored dependencies aren't in snapshots** (`node_modules/`, `.venv/`, build output), so a
  bare checkout may not be able to run your tests. `--copy-ignored` symlinks the project's
  ignored *top-level dependency* directories into each checkout: `node_modules`, `.venv`,
  `venv`, `env`, `vendor`, `.tox`, `bower_components`. Build output (`dist/`, `build/`) is not
  linked unless you name it (`--copy-ignored dist`). Links are shared, not copied: a test that
  writes into them writes into the real ones, which aren't in any snapshot. Nested ones
  (`web/node_modules`) aren't linked.
- **Editable installs point back at your project.** If your package is installed with
  `pip install -e`, a test in the checkout may import the project's copy instead. Configure the
  import path in the test (`pythonpath = ["src"]` in pytest's config does this) or prefix
  `PYTHONPATH=src`.
- **Flaky tests confuse any bisect.** Make the command deterministic and fast; it runs about
  log2(actions) + 2 times.
- **Only what the airbag snapshotted is searched**: changes made between two hooked tool calls
  (by you, or a process without hooks) are attributed to the action before them.
- Each run sees `VCR_BISECT_SNAPSHOT=<n>` in its environment, and no `GIT_DIR`-style variables.

## Rewind your local Postgres too

A file snapshot can't undo `DROP TABLE` on a database server: the data isn't in your project.
For a **local development database you name explicitly**, the airbag can: it runs `pg_dump`
right before the destructive SQL, and `vcr undo` puts the database back.

```bash
export AGENT_VCR_AIRBAG_DATABASES=postgresql://localhost:5433/app_dev   # comma-separated
```

…or per project, in `.vcr/airbag-databases.json`:

```json
{"databases": ["postgresql://localhost:5433/app_dev"]}
```

Both are read. Keep passwords out of that file (it's in your project): use the environment
variable, `~/.pgpass`, or a local socket with peer auth. Then:

```bash
vcr airbag db list
#   ✓ app_dev (localhost:5433)
vcr airbag check "psql -h localhost -p 5433 -d app_dev -c 'DROP TABLE users'"
# 🛟 DESTRUCTIVE: destroys project data — snapshot first, undo with `vcr undo`
#    • SQL DROP on local database app_dev (dumped with pg_dump before it runs)
```

Without the config, the same command is *irreversible* and the agent has to ask you. With it,
the agent's `DROP TABLE` goes ahead, you see
`🛟 Agent VCR airbag: snapshot #9 saved before … (database app_dev snapshotted)`, and:

```bash
vcr undo
# ↩ Undid: psql -h localhost -p 5433 -d app_dev -c 'DROP TABLE users'  (claude, 0s ago)
#   restored database app_dev — back to snapshot #9.
#     ⟲ database app_dev
#   Changed your mind? `vcr redo`
```

`vcr rewind <#>` brings the database back to how it was at that snapshot too, and
`vcr undo --dry-run` lists `⟲ database …` among the changes. Dumped actions show
`[db: app_dev]` in `vcr rewind`.

**Only local databases, ever.** URLs must point at `localhost`, `127.0.0.1`, `::1` or a unix
socket (`postgresql:///app?host=/var/run/postgresql`). Anything else is refused, never dumped:

```bash
AGENT_VCR_AIRBAG_DATABASES=postgresql://admin:pw@db.prod.internal/app vcr airbag db list
#   ✗ refused: postgresql://admin:***@db.prod.internal/app: db.prod.internal is not a local database. …
```

**You're only spared the prompt when the airbag is sure.** That means a `psql` command whose SQL
is in `-c` and which states its host (`-h`, a conninfo/URL argument, or an inline `PGHOST=`),
database and port (or the default 5432), all pointing at a configured local database. When in
doubt it still dumps whatever configured database it can match, but it asks you anyway:

- a remote host named anywhere in the command, even if libpq would let a later one win;
- anything that can redirect the connection: `PGHOSTADDR`, `PGSERVICE`/`service=`, an
  `export PGHOST=…` earlier in the command, or no host at all (the default socket);
- SQL it can't see or that reaches further: `-f file.sql`, stdin/heredocs, `\connect`, `\i`,
  `\!`, `dblink`, foreign servers, `COPY … PROGRAM`, `DROP DATABASE`, `DROP ROLE`;
- ORM and framework commands (`prisma migrate reset`, `rails db:reset`,
  `manage.py flush`, …), which read their own config (`.env` files, `database.yml`, `--env`,
  linked projects); also `pgcli`, `dropdb`, `rails db:drop`;
- anything irreversible for another reason too (`… && git push --force`).

**How it works.** A `psql` target is resolved the way libpq does it (a conninfo or URL argument
overrides `-h`/`-p`/`-U`, which override `PG*` variables) and must match a configured database:
same database name and port, and both loopback TCP or the same socket directory. ORM commands
are matched through `DATABASE_URL` (environment or the project's `.env`) or the only configured
database, which is good enough for a dump but never for skipping the prompt.
`pg_dump --format=custom` writes into the airbag store (`<store>/db/<snapshot>-<db>.dump`, mode `0600`) and the dump is
recorded in that snapshot. The password goes to `pg_dump` through `PGPASSWORD`, never the
command line, and is never stored. Undo first dumps the database's *current* state (so the undo
is undoable), then, in one transaction, drops the user tables and views (their serial and
identity sequences go with them) and any other sequences, and replays the dump with `psql`.
`vcr undo` only restores a database when the action it reverts dumped it: undoing a later file
edit leaves the database alone. `vcr rewind <#>` puts every dumped database back to how it was
at that snapshot. If `psql` is missing or that fails, it falls back to
`pg_restore --clean --if-exists --no-owner`, which keeps tables created after the dump, and says
so.

```bash
vcr airbag db snapshot -m "before trying the migration"  # dump now (a save point)
# 🗄  Snapshot #11: dumped database app_dev. Restore with `vcr airbag db restore 11`.
vcr airbag db restore 11      # only the database back to #11; files untouched
# ⏪ Restored database app_dev to snapshot #11. Undo with `vcr airbag db restore 12`.
vcr airbag gc --keep-days 7   # also deletes the dumps of deleted snapshots
```

**Limits.**

- **Postgres only**, through `pg_dump`/`pg_restore`/`psql` on your PATH (or
  `AGENT_VCR_AIRBAG_PG_BIN=/usr/lib/postgresql/16/bin`). Use client tools at least as new as
  the server. If `pg_dump` is missing or fails, the command is stopped for confirmation as before,
  with the reason.
- **Dumps happen only before SQL the classifier flags as destructive** (`DROP`, `TRUNCATE`,
  `DELETE`/`UPDATE` without `WHERE`, `ALTER TABLE … DROP`, ORM resets). `psql -f migrate.sql`
  isn't inspected. Take a save point first with `vcr airbag db snapshot`.
- **The hook sees its own environment, not your agent's shell.** A `PGHOSTADDR` exported in an
  earlier tool call is invisible to it. Agents normally start each command from the same
  environment, but if yours keeps shell state between calls, keep `PGHOSTADDR`/`PGSERVICE` out of it.
- Types, functions and extensions created after the dump survive a restore; tables, views and
  sequences don't.
- **A restore puts the whole database back**, including rows your app wrote since the dump. The
  pre-restore dump means you can go forward again.
- **A dropped database isn't recreated.** After an approved `dropdb app_dev`, run
  `createdb app_dev` first, then `vcr airbag db restore <#>`.
- The dump runs inside the hook: big databases make tool calls slow. The default timeout is
  20 s (`AGENT_VCR_AIRBAG_DB_TIMEOUT`); past it, the command is stopped for confirmation.

## What it can't do

- **Undo in folders that aren't protected** (not a git repository, not opted in with
  `vcr airbag protect`; always `~` and `/`). Nothing is snapshotted there: deleting a directory
  tree is stopped for a human, other changes go ahead with a notice and can't be undone
  ([which folders](#which-folders-are-protected)).
- **Anything outside the project directory**: other folders, `~`, system files. Deletes and
  writes that land there (`rm -rf ~/Documents`, `echo x > /etc/foo.conf`, `cp a.txt ../x.txt`,
  `mv`/`rsync`/`tee`/`install` targets) are classified *irreversible* and stopped; only paths
  you name under a temp directory (`/tmp/...`) count as scratch space. It's a parser, not a
  sandbox: a script or a program that picks its own output path isn't seen.
- **Remote state**: databases on a server (including a Postgres/MySQL server on localhost — its
  data isn't in your project), cloud resources, pushed commits, published packages, sent
  requests. Classified *irreversible* and stopped for confirmation. The one exception is a
  local Postgres database you configure ([above](#rewind-your-local-postgres-too)).
- **Ignored dependency/build directories and files over the size limit** (25 MB,
  `AGENT_VCR_AIRBAG_MAX_FILE_MB`). A restore never overwrites such a file; it's left as-is and
  listed as "left untouched".
- **Nested git repositories** (submodules, cloned dependencies): git records only a pointer, not
  their files. Deleting one is therefore treated as irreversible and stopped.
- **Empty directories** (git doesn't track them).
- **Commands an agent runs without a hook.** Use `vcr airbag run --`, which snapshots every
  2 seconds but can't intercept anything.
- **It's a guard, not a sandbox.** An agent set on hiding a command (for example inside a script
  it just wrote) can get past the classifier. The snapshot still happens, so project files stay
  undoable.

## Configuration

| Variable | Effect |
|---|---|
| `AGENT_VCR_AIRBAG=off` | Disable entirely (`0`, `false`, `no`, `disabled` work too). |
| `AGENT_VCR_AIRBAG_MODE=risky` | Snapshot only before destructive/irreversible commands (default `every`). |
| `AGENT_VCR_AIRBAG_ON_IRREVERSIBLE=ask\|deny\|warn` | What to do with commands a snapshot can't undo (default `ask`). |
| `AGENT_VCR_AIRBAG_HOME=/path` | Where snapshot stores live. |
| `AGENT_VCR_AIRBAG_MAX_FILE_MB=25` | Files larger than this aren't snapshotted. |
| `AGENT_VCR_AIRBAG_MAX_FILES=200000` | Refuse to protect trees with more files than this (stops you from accidentally snapshotting your home directory). |
| `AGENT_VCR_AIRBAG_DATABASES=postgresql://localhost/app_dev` | Local Postgres databases to dump before destructive SQL (comma-separated; also `.vcr/airbag-databases.json`). |
| `AGENT_VCR_AIRBAG_PG_BIN=/usr/lib/postgresql/16/bin` | Where `pg_dump`, `pg_restore` and `psql` are (default: PATH). |
| `AGENT_VCR_AIRBAG_DB_TIMEOUT=20` | Seconds a `pg_dump` inside a hook may take. |

## Airbag FAQ

**Does it replace git?** No. Commit as usual. The airbag covers the gap between commits, plus the
things git ignores on purpose (`.env`, local databases).

**Will it slow my agent down?** About 0.1 s per tool call, most of it Python start-up.

**Where are my secrets going?** Nowhere off your machine. Snapshots of `.env` live in the
private store on your disk (mode `0700`) and keep their original permissions.

**I ran `vcr undo` by mistake.** Run `vcr redo`.

**Will my teammates notice I installed it?** No. Hooks go into your user-level agent configs,
snapshots into your own state directory; nothing is written into the repository (unless you
ask for `--project`).

**Two agents are installed and both run the Claude hook. Double snapshots?** No. A tool call
that reaches the airbag twice within 5 seconds is handled once.

**Is the classifier perfect?** No, it's a heuristic. When it isn't sure, it snapshots. Only
remote or outside-the-project damage needs a human, because that's the only kind a snapshot
can't fix.

## Sources for the hook contracts

Verified on 2026-11-11 against official docs and, where installed, the agents' own code:
Claude Code — code.claude.com/docs/en/hooks · Codex CLI 0.154 — developers.openai.com/codex and
`openai/codex` source · Gemini CLI 0.37.1 — installed package source and bundled docs ·
Cursor — cursor.com/docs/hooks · Copilot CLI — docs.github.com/en/copilot/reference/hooks-reference ·
OpenHands SDK 1.7.3 — installed source · Cline — `cline/cline` repository ·
Devin Local — docs.devin.ai. Payload formats for Cursor, Copilot, Cline and Devin were tested
with documented sample payloads, not live agent sessions.

User-level hook locations, verified on 2026-11-12: Claude Code `~/.claude/settings.json`
(code.claude.com/docs/en/hooks, settings, statusline) · Codex `~/.codex/hooks.json`
(Codex hooks docs; `/hooks` trust applies to user hooks too) · Gemini CLI `~/.gemini/settings.json`
(0.37.1 bundle: the trusted-folder check drops user hooks too) · Cursor `~/.cursor/hooks.json`
(cursor.com/docs/hooks) · Copilot CLI `~/.copilot/hooks/*.json` or `$COPILOT_HOME/hooks`
(Copilot CLI hooks reference) · OpenHands `~/.openhands/hooks.json`, read only when the project
has none (`software-agent-sdk` `HookConfig.load`) · Devin `~/.config/devin/config.json`, or
`%APPDATA%\devin\config.json` (docs.devin.ai CLI hooks) · OpenCode
`~/.config/opencode/plugin/` (opencode.ai/docs/plugins; the 1.19 binary loads `plugin/` and
`plugins/`) · Cline `<Documents>/Cline/Hooks/` (`cline/cline` source). These locations come from
docs and source; no live agent session has run the user-level files.

## Safety review

Before release, an independent reviewer attacked the restore path: file growth past the size
limit, "read-only" commands that overwrite files, nested repositories, permissions, line endings,
damaged `.git`, worktrees, odd file names, symlinks, concurrency, hostile hook payloads. Every
finding has a regression test in `tests/unit/test_airbag_core.py`.

## Cheat sheet

| Command | What it does |
|---|---|
| `vcr airbag install [agent…] [--all] [--project] [--statusline]` | Arm the hooks at the user level (default: every agent on your PATH); moves 1.7.0 project hooks. |
| `vcr doctor [--fix] [--json]` | Check the installation and print the exact fix; `--fix` repairs stale hooks. |
| `vcr airbag statusline` | One status line for Claude Code (reads its JSON on stdin). |
| `vcr undo [N] [--dry-run]` | Restore files to just before the last (N-th) agent action that changed something. |
| `vcr redo` | Undo the last undo/rewind. |
| `vcr rewind` / `vcr airbag log` | Timeline of protected actions with snapshot numbers (`-n` to limit). |
| `vcr rewind 12` | Restore snapshot #12 (undoable with `vcr redo`). |
| `vcr bisect -- <test cmd>` | Find the agent action that broke the tests (`--good N`, `--bad N`, `--timeout S`, `--copy-ignored [DIR]`, `--json`). |
| `vcr airbag db list` | Configured local Postgres databases and their dumps. |
| `vcr airbag db snapshot [--db NAME]` | Dump the configured database(s) now. |
| `vcr airbag db restore 12 [--db NAME]` | Put only the database(s) back to snapshot #12. |
| `vcr airbag check "<cmd>"` | Classify a command without running it. |
| `vcr airbag allow` | Approve the last stopped command for one retry within 10 minutes. |
| `vcr airbag snapshot -m "msg"` | Manual snapshot. |
| `vcr airbag status` | Whether this folder is protected, policy, store location, installed hooks. |
| `vcr airbag protect [DIR]` / `vcr airbag unprotect [DIR]` | Opt a folder that isn't a git repository in to snapshots (`DIR/.vcr/airbag.json`), or back out. |
| `vcr airbag gc --keep-days 7 [--keep-last 50]` | Delete old snapshots (and their database dumps). |
| `vcr airbag run -- <cmd>` | Periodic snapshots for agents without hooks. |
| `vcr airbag uninstall [agent…] [--user\|--project]` | Remove only the airbag's hook entries (both levels by default). |

**Next:** [Quickstart](quickstart.md) · [Recording and replay](recording-and-replay.md) ·
[ACID workspace](acid-workspace.md) (file rollback inside your own agent code)
