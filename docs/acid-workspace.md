# 🔁 ACID workspace

Save points for the agent *you* write: files on disk and agent state roll back together, on a
git branch of their own, and only the good attempt gets merged.

**TL;DR**

- `ACIDWorkspace` wraps your agent loop in a transaction: `begin()` → `savepoint()` →
  `rollback()` or `commit()` / `abort()`. Each savepoint is a git commit **and** a recorded frame.
- Files that git doesn't own, like an ignored `.env` or a local SQLite file, are copied into a
  private escrow, so rollback restores them too, and their bytes never land in a commit.
- It ships in the core package (`pip install ai-agent-vcr`). OpenHands isn't required.

## Try it in 60 seconds

Make a throwaway repo with a tracked file and an ignored secret:

```bash
mkdir acid-demo && cd acid-demo
git init -q -b main repo
echo 'print("v1")' > repo/app.py
echo '.env' > repo/.gitignore
echo 'API_KEY=secret' > repo/.env
git -C repo add .
git -C repo -c user.name=demo -c user.email=demo@example.com commit -qm init
```

Save this as `acid_demo.py` next to `repo/` and run `python acid_demo.py`:

```python
from pathlib import Path

from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

repo = Path("repo")
ws = ACIDWorkspace(str(repo), recorder=VCRRecorder(output_dir=".vcr"))
ws.begin(session_id="fix-bug-42")
ws.savepoint({"phase": "planned"}, node_name="planner")        # frame 0: a safe point

# The "agent" goes wild.
(repo / "app.py").write_text('print("v2 - broken")\n')
(repo / "junk.txt").write_text("hallucinated file\n")
(repo / ".env").unlink()
ws.savepoint({"phase": "coded"}, node_name="coder")             # frame 1

ws.rollback(to_frame_index=0)                                   # files AND state rewind
print((repo / "app.py").read_text().strip(), (repo / "junk.txt").exists(),
      (repo / ".env").read_text().strip())

(repo / "app.py").write_text('print("v2 - fixed")\n')            # the good attempt
ws.commit()                                                     # merge into main
```

```text
print("v1") False API_KEY=secret
```

The broken edit is gone, the hallucinated file is gone, and the deleted `.env` is back. Then
`git -C repo log --oneline --graph` shows one merge commit on `main` with the fixed `app.py`.
The `acid/fix-bug-42` branch and the escrow have been cleaned up.

## How it works

```text
            begin()                savepoint()            rollback(i)          commit()
 main ──●────────┐                                                        ┌──▶ ● merge
                 └─ acid/<id> ──● SAVEPOINT ──● SAVEPOINT ──◀ reset to i ─┘   (--no-ff)
                                │             │
            .git/vcr-escrow/    ├─ escrow     ├─ escrow     restore escrow i
            (private copies)    └─ frame 0    └─ frame 1    + fork recording
```

- **Git owns tracked files.** `begin()` creates an `acid/<session_id>` branch and points `HEAD`
  at it. That's a ref switch only, so no file changes. Each savepoint commits tracked edits and
  any new files the agent created onto that branch, and records a frame with the commit hash
  in `metadata.custom["acid_commit_hash"]`.
- **Escrow owns everything else.** At `begin()` and at every savepoint, untracked and ignored
  files are copied into `.git/vcr-escrow/<session>-<random>/blobs/<sha256>`. The folder is mode
  `0700`, the blobs `0600`, and identical bytes are stored once. That's outside your working
  tree, outside git objects and outside `.vcr` recordings. Files that existed before `begin()`
  are never handed to `git add`, so a secret can't even become an unreachable git object.
- **Rollback rewinds both worlds.** Tracked files are reset to the savepoint commit. Files created
  after it are deleted, along with empty folders that are new. Escrowed files that were
  modified, deleted, renamed, retyped, chmod-ed or re-pointed (symlinks) are restored. Restores go
  through a temp file and `os.replace`, with the original permission bits. Only after the files
  are back does the recording fork in place, so wrappers holding the recorder keep recording.
- **Commit locks it in.** `commit()` saves the recording, commits anything changed since the last
  savepoint, merges the branch into the branch you started on with a merge commit, deletes
  `acid/<id>` and removes the escrow. `abort()` puts the working tree **and** index back exactly as
  they were at `begin()`, including uncommitted edits you had before, and drops the branch.
- **It stays in its lane.** Only paths inside `workspace_dir` change, even when it's a
  subdirectory of a bigger repo. Every path is checked to be relative and to resolve inside the
  workspace, so `..`, absolute paths and directory-symlink escapes are rejected. Git runs with
  hooks and signing disabled, a private index, a timeout (`git_timeout`, 60 s by default) and
  captured output. Savepoints use an internal `agent-vcr` identity; the merge uses yours when
  git has one. `.vcr/`, `.venv/`, `node_modules/` and friends go into `.git/info/exclude`. Your
  `.gitignore` isn't touched.

## Everyday recipes

### Commit on success, abort on a crash

```python
from pathlib import Path

from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

repo = Path("repo")

try:
    with ACIDWorkspace(str(repo), recorder=VCRRecorder(output_dir=".vcr")) as ws:
        ws.savepoint({"phase": "start"}, node_name="boundary")
        (repo / "app.py").write_text("oops\n")
        raise RuntimeError("agent crashed")
except RuntimeError:
    pass
print("after abort:", (repo / "app.py").read_text().strip())
```

The `with` block calls `begin()` with a fresh session id, `commit()` on a clean exit and
`abort()` when an exception escapes. The exception still propagates. For production control
flow, explicit calls make the review step clearer: roll back, inspect and run your tests, *then*
call `commit()`.

### Start with uncommitted edits

By default `begin()` refuses a dirty tracked worktree, so a rollback can't discard work that
predates the agent:

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

ws = ACIDWorkspace("repo", recorder=VCRRecorder(output_dir=".vcr"),
                   dirty_worktree_policy="allow")
ws.begin("dirty-ok")
```

With `"allow"`, your edits become part of the transaction (a `commit()` merges them along with
the agent's work). `abort()` restores them as they were either way. Untracked files never count
as dirty: they go to escrow.

### Look at where you are

```python
print(ws.status())       # session_id, branch, base_ref, savepoints, workspace
print(ws.savepoints())   # [{"index", "node_name", "frame_id", "acid_commit_hash"}, ...]
ws.abort()
```

Run this right after the previous recipe. After a `rollback(i)`, the recording has forked: the
new session starts with one `fork:<node>` frame at index 0, so savepoint indices count from there.

### Tune what escrow copies

```python
from agent_vcr.integrations.openhands import ACIDWorkspace

ws = ACIDWorkspace(
    "repo",
    max_untracked_file_bytes=2 * 1024 * 1024,     # default 10 MiB per file
    untracked_exclude_dirs={"vendor", "artifacts"},
    git_timeout=120,
)
```

Your exclusions are added to the defaults: `.git`, `.venv`, `venv`, `node_modules`,
`__pycache__`, `.vcr`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, plus the recorder's own
output folder when it sits inside the workspace. Oversized files, excluded trees, sockets, FIFOs
and devices aren't copied. If one of those existed before `begin()`, it is left in place on
rollback, never deleted as if the agent had made it up.

### Plug in your agent's recorder

Pass the recorder your framework wrapper already uses (`ACIDWorkspace(path, recorder=rec)`), and
savepoints land in the same `.vcr` timeline as the agent's LLM and tool frames. Rollback forks
that recorder in place, so a `VCRLangGraph` or `VCRCrewAI` wrapper holding it keeps recording
into the new branch. See [integrations](integrations/README.md). Without a recorder, one is
created that writes to `.vcr/` in the current directory.

### Not a git repo yet?

Point it at any folder. If it isn't inside a repo, `begin()` runs `git init` (branch `main`) and
makes a baseline commit containing only an empty `.acid_init` marker. Every other file you
already had goes to escrow, not into git. On an existing repo with no commits, it adds the same
baseline commit.

## Airbag or ACIDWorkspace?

| | 🛟 [Airbag](airbag.md) | 🔁 ACIDWorkspace |
|---|---|---|
| **For** | Coding-agent CLIs you *use*: Claude Code, Codex, Cursor, Gemini CLI, Copilot CLI and more | Agents you *write* in Python |
| **Who decides when to snapshot** | Hooks, automatically, before every tool call (shell commands included) | Your code, by calling `savepoint()` |
| **Where snapshots live** | A shadow git store **outside** the project; your `.git` is never touched | An `acid/<id>` branch **in** your repo, plus escrow under `.git/vcr-escrow/` |
| **Undo** | `vcr undo` / `vcr rewind`, any time later | `rollback(i)` inside the running transaction |
| **Ends with** | Nothing to finish; snapshots pile up until pruned | `commit()` merges the result, `abort()` discards it |
| **Records agent state** | No, files only | Yes, every savepoint is a `.vcr` frame |

Rule of thumb: if you type to the agent, use the airbag. If you `import` the agent, use
ACIDWorkspace.

## What it can't do

- **It can't undo the outside world.** API calls, sent emails, payments and remote databases
  stay changed. Rollback covers files inside the workspace, nothing else.
- **Escrow is not a vault.** Blobs are local plaintext protected by file permissions, not
  encryption. Any process running as you can read them. A hard kill (power loss, `kill -9`) can
  leave an orphaned `.git/vcr-escrow/<session>-*` folder; delete it by hand.
- **Skipped files can't be rebuilt.** If an oversized, excluded or special file is changed by
  the agent, its old content is gone. New oversized files created after a savepoint also survive
  a rollback.
- **No detached HEAD, no half-finished merge.** `begin()` refuses both. Check out a branch first
  (`git switch -c agent-work`).
- **One transaction at a time per repo.** While it runs, `HEAD` points at `acid/<id>`, which your
  editor and other git tools will see. If the base branch moves meanwhile (say, from another
  worktree), `commit()` falls back to a real `git merge --no-ff`, which can conflict.
- **A leftover `acid/<id>` branch from a crash is left alone.** The next `begin()` with that id
  uses `acid/<id>-2` instead. `abort()` also keeps the branch if it holds commits that
  ACIDWorkspace didn't make.
- **Recordings still hold what you put in them.** The states you pass to `savepoint()` are written
  to `.vcr` as-is. Escrow keeps file *contents* out of recordings, not secrets you put in state.

## Cheat sheet

| Call | What it does |
|---|---|
| `ACIDWorkspace(dir, recorder=None, dirty_worktree_policy="fail", max_untracked_file_bytes=10 MiB, untracked_exclude_dirs=None, git_timeout=60.0)` | Set up (nothing changes on disk yet) |
| `begin(session_id=None)` | Validate the id, check the repo, escrow local files, create `acid/<id>` |
| `savepoint(state, node_name="openhands_action")` | Git commit + escrow + recorded frame; returns the `Frame` |
| `rollback(to_frame_index)` | Restore files and escrow to that savepoint, fork the recording in place |
| `commit()` | Save the recording, commit leftovers, merge into the base branch, delete `acid/<id>`, clean escrow |
| `abort()` | Restore the `begin()` state exactly, drop the branch, clean escrow |
| `status()` / `savepoints()` | Inspect the transaction |
| `with ACIDWorkspace(dir) as ws:` | `begin()`, then `commit()` on success or `abort()` on exception |
| `branch_name`, `base_ref`, `session` | The live branch, the branch you started on, the current recording session |

Want the full loop with a golden run at the end? `python examples/acid_golden_run.py` (it works
in `/tmp/acid_agent_workspace`). The tests live in `tests/unit/test_acid_golden.py` and
`tests/unit/test_acid_workspace_safety.py`. The original rollback-boundary question is issue
[#2](https://github.com/ixchio/agent-vcr/issues/2).

**Next:** [Airbag](airbag.md) (the same idea for coding-agent CLIs) ·
[Recording & time travel](recording-and-replay.md) (what's inside the `.vcr` file) ·
[Agent CI](agent-ci.md) (turn the good run into a regression test)
