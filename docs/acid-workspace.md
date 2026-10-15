# ACID workspace rollback

`ACIDWorkspace` joins an Agent VCR state frame to a physical filesystem savepoint. It is intended
for coding agents and other repository-writing agents where restoring only the Python state object
would leave the working tree polluted.

The implementation has two recovery layers:

1. **Git owns tracked content.** Each transaction runs on an isolated `acid/<session_id>` branch.
   Every savepoint commits tracked state and agent-created files that are safe to stage.
2. **Private escrow owns pre-existing local content.** Eligible untracked and ignored files are
   captured per savepoint in `.git/vcr-escrow/<session>/blobs/`, outside the working tree, Git
   objects, commits, and `.vcr` recordings.

This second layer makes a pre-existing ignored `.env` recoverable after an agent modifies, deletes,
renames, type-changes, or changes permissions on it.

## Install and prerequisites

The core package contains `ACIDWorkspace`; the `openhands` extra is needed only when your application
also imports OpenHands.

```bash
pip install ai-agent-vcr
```

Use a Git repository with a clean tracked worktree. By default, `begin()` refuses to run when
tracked files already have uncommitted changes because a later rollback would otherwise discard
work that predates the agent session.

## Transaction lifecycle

```python
from agent_vcr import VCRRecorder
from agent_vcr.integrations.openhands import ACIDWorkspace

recorder = VCRRecorder(output_dir=".vcr")
workspace = ACIDWorkspace(
    "/path/to/repository",
    recorder=recorder,
    max_untracked_file_bytes=10 * 1024 * 1024,
)

workspace.begin(session_id="issue-42")
workspace.savepoint({"phase": "planned"}, node_name="planner")

# Let the agent work inside the isolated branch.

workspace.savepoint({"phase": "implemented"}, node_name="coder")
workspace.rollback(to_frame_index=0)

# Inspect and verify the restored workspace before accepting it.
workspace.commit()
```

The context-manager form commits on a clean exit and aborts on an exception:

```python
from agent_vcr.integrations.openhands import ACIDWorkspace

with ACIDWorkspace("/path/to/repository") as workspace:
    workspace.savepoint({"phase": "before-agent"}, node_name="boundary")
    run_coding_agent()
```

For production control flow, explicit lifecycle calls make the review boundary clearer.

## Operation guarantees

### `begin()`

- validates the session ID before using it in Git commands;
- records the current branch or detached commit as the base ref;
- refuses a dirty tracked worktree unless `dirty_worktree_policy="allow"` is explicit;
- snapshots eligible pre-existing untracked and ignored files into private local escrow; and
- creates an isolated `acid/<session_id>` branch.

### `savepoint(state, node_name=...)`

- stages tracked mutations and agent-created untracked files;
- never presents pre-existing escrow-owned paths to `git add`, preventing secret bytes from being
  written even as unreachable Git objects;
- creates a Git savepoint commit;
- snapshots eligible current local files into the same content-addressed escrow; and
- records a VCR frame carrying the savepoint commit hash.

### `rollback(to_frame_index)`

- resets tracked content to the target savepoint commit;
- removes untracked/ignored files created after that boundary when they are inside the cleanup
  policy;
- removes newly created empty parent directories;
- restores missing, deleted, renamed, type-changed, content-changed, mode-changed, and
  symlink-target-changed local files from escrow; and
- forks the in-memory recording only after filesystem restoration succeeds.

Regular files are restored through a sibling temporary file followed by `os.replace`. Original
permission bits are reapplied. Symlinks are restored as symlinks with their exact target. Every
candidate path is validated as relative and its resolved parent must remain inside the workspace,
so `..`, absolute paths, and directory-symlink escapes are rejected.

Rollback may be called more than once while the transaction is active. Escrow blobs are immutable,
addressed as `blobs/<sha256>`, and remain available until `commit()` or `abort()` cleans the session.

### `commit()` and `abort()`

`commit()` saves the recording, returns to the base ref, merges the isolated branch with a merge
commit, and removes the escrow directory. `abort()` resets and discards the transaction branch,
restores the initial local snapshot, and removes the escrow directory.

## Local-file safety policy

The default maximum escrow size is 10 MiB per regular file. Configure it and add exclusions with:

```python
workspace = ACIDWorkspace(
    "/path/to/repository",
    max_untracked_file_bytes=2 * 1024 * 1024,
    untracked_exclude_dirs={"vendor", "artifacts"},
)
```

Custom exclusions are added to `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.vcr`, and
the supported tool-cache defaults. Oversized files, excluded trees, sockets, devices, FIFOs, and
other special files are not copied. Pre-existing skipped paths are preserved rather than mistaken
for hallucinated output and deleted. Their original content cannot be reconstructed if mutated.

Escrow directories use `0700` permissions and SHA-256-addressed blobs use `0600`. Identical bytes
within a transaction share one blob.

## Secret boundary

Escrow protects local-file payloads from Git staging and VCR serialization, but it is local
plaintext protected by filesystem permissions, not an encrypted secret store. A process running as
the repository owner can read it, and abrupt host termination can leave an orphaned session
directory for manual cleanup.

Prompts, states, tool arguments, and outputs explicitly passed to `VCRRecorder` are still written
to `.vcr`. Git/workspace rollback also cannot compensate external APIs or services.

## Verify locally

```bash
python examples/acid_golden_run.py
pytest tests/unit/test_acid_golden.py -v
```

Issue [#2](https://github.com/ixchio/agent-vcr/issues/2) contains the original rollback-boundary
question. Content-aware per-savepoint escrow shipped in `ai-agent-vcr==1.5.2`.

`commit()` does not require callers to create a final savepoint. Before it switches back to the
base ref, Agent VCR stages and commits every Git-owned change still present on the isolated ACID
branch. The merge therefore includes tracked edits, deletions, and generated non-ignored files made
after the last savepoint; if any tracked change remains dirty, commit stops instead of reporting a
false success.
