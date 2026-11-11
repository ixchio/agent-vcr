from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from agent_vcr.models import Frame, FrameMetadata, Session
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)

# Strict whitelist: only alphanumerics, hyphens, underscores, and dots.
_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
_DEFAULT_LOCAL_EXCLUDES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".vcr",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_COPY_CHUNK_BYTES = 1024 * 1024
_DEFAULT_GIT_TIMEOUT_SECONDS = 60.0
_ESCROW_ATTEMPTS = 3
_MAX_BRANCH_SUFFIX = 1000
_ADD_BATCH_SIZE = 128

# Identity used for ACIDWorkspace's own bookkeeping commits (savepoints, stash snapshots,
# the initial baseline). Never depends on the user's git configuration.
_INTERNAL_GIT_NAME = "agent-vcr"
_INTERNAL_GIT_EMAIL = "agent-vcr@localhost"

# Environment variables that would point git at a different repository/index than the
# workspace (for example when the host process itself runs inside a git hook).
_GIT_LOCATION_ENV = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_WORK_TREE",
    }
)


class _GitCommandError(subprocess.CalledProcessError):
    """``CalledProcessError`` whose message includes git's stderr."""

    def __str__(self) -> str:
        message = super().__str__()
        stderr = self.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        detail = stderr.strip() if isinstance(stderr, str) else ""
        return f"{message}\n{detail}" if detail else message


class _EscrowFileChanged(RuntimeError):
    """A local file changed (or changed type) while it was being escrowed."""


@dataclass(frozen=True)
class _LocalFileMetadata:
    """Content and filesystem metadata for one locally owned path."""

    kind: Literal["regular", "symlink"]
    sha256: str
    mode: int
    symlink_target: str | None
    size: int


@dataclass(frozen=True)
class _LocalSnapshot:
    """Rollback boundary for files which are not represented by Git commits."""

    manifest: dict[str, _LocalFileMetadata]
    skipped_paths: dict[str, str]
    directories: frozenset[str]


class ACIDWorkspace:
    """
    Provides ACID transaction semantics for agent execution.
    Combines Agent VCR state snapshotting with Git-backed filesystem savepoints and a
    private content escrow for untracked and ignored files.

    Safety guarantees:

    * Only paths inside ``workspace_dir`` are ever modified, even when the workspace is a
      subdirectory of a larger repository. During a transaction the repository's ``HEAD``
      points at the ``acid/<session_id>`` branch (a ref-only switch, no files change).
    * ``abort()`` restores the workspace's working tree *and* index exactly as they were
      at ``begin()``, including pre-existing uncommitted edits (``dirty_worktree_policy=
      "allow"``) and untracked/ignored files.
    * Savepoint commits are built with a private index via git plumbing: user hooks,
      commit signing, and missing ``user.name``/``user.email`` cannot break them, and
      changes the user staged outside the workspace are never included.
    * Every git invocation has a timeout (``git_timeout``) and its output is captured, so
      nothing is written to the host process's stdout.
    """

    def __init__(
        self,
        workspace_dir: str,
        recorder: VCRRecorder | None = None,
        dirty_worktree_policy: Literal["fail", "allow"] = "fail",
        max_untracked_file_bytes: int = 10 * 1024 * 1024,
        untracked_exclude_dirs: Iterable[str] | None = None,
        git_timeout: float = _DEFAULT_GIT_TIMEOUT_SECONDS,
    ) -> None:
        if max_untracked_file_bytes < 0:
            raise ValueError("max_untracked_file_bytes must be non-negative")
        if git_timeout <= 0:
            raise ValueError("git_timeout must be positive")

        self.workspace_dir = Path(workspace_dir).resolve()
        self.recorder = recorder or VCRRecorder(auto_save=True)
        self.dirty_worktree_policy = dirty_worktree_policy
        self.max_untracked_file_bytes = max_untracked_file_bytes
        self.git_timeout = float(git_timeout)
        self.untracked_exclude_dirs = frozenset(
            _DEFAULT_LOCAL_EXCLUDES | set(untracked_exclude_dirs or ())
        )
        self.session: Session | None = None
        self.branch_name: str | None = None
        self.base_ref: str | None = None
        self._baseline_local_manifest: dict[str, _LocalFileMetadata] = {}
        self._initial_local_snapshot: _LocalSnapshot | None = None
        self._local_snapshots: dict[str, _LocalSnapshot] = {}
        self._filesystem_savepoints: dict[str, str] = {}
        self._escrow_dir: Path | None = None
        self._escrow_blobs_dir: Path | None = None
        # Repository layout: git commands run from the top level; ``_prefix`` is the
        # workspace's path relative to it ("" or "some/dir/").
        self._toplevel: Path | None = None
        self._prefix = ""
        # Pre-transaction state captured at begin().
        self._base_commit: str | None = None
        self._begin_worktree_commit: str | None = None
        self._begin_index_commit: str | None = None
        self._internal_commits: set[str] = set()

        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> ACIDWorkspace:
        """Context-manager entry — calls :meth:`begin` with a fresh session."""
        self.begin()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Commit on clean exit, abort (rewind + branch cleanup) on exception."""
        if exc_type is None:
            self.commit()
        else:
            try:
                self.abort()
            except Exception:
                logger.warning("ACID abort on exit failed", exc_info=True)

    @staticmethod
    def _sanitize_session_id(session_id: str) -> str:
        """Validate session_id against a strict whitelist to prevent git argument injection.

        Git interprets strings starting with ``-`` as flags, so an
        unsanitised session ID like ``--orphan`` would be treated as a
        git option rather than a branch name.  This guard ensures only
        safe characters are allowed.
        """
        if not _SAFE_SESSION_ID.match(session_id):
            raise ValueError(
                f"Invalid session_id {session_id!r}: must match {_SAFE_SESSION_ID.pattern}. "
                f"Only alphanumerics, hyphens, underscores, and dots are allowed."
            )
        return session_id

    # ------------------------------------------------------------------
    #  Transaction API
    # ------------------------------------------------------------------

    def begin(self, session_id: str | None = None) -> Session:
        """
        BEGIN - Starts a session and snapshots local, non-git-owned files.

        Refuses to start on a detached HEAD (the final merge would land on an unnamed
        commit) or with unresolved merge conflicts. If a branch named
        ``acid/<session_id>`` already exists (e.g. left over from a crashed process) it is
        left untouched and a unique ``acid/<session_id>-<n>`` branch is used instead; see
        :attr:`branch_name`.
        """
        self.session = self.recorder.start_session(session_id)
        if self.session is None:
            raise RuntimeError("Session failed to start")

        # Validate session ID before using it in git commands
        self._sanitize_session_id(self.session.session_id)

        self._ensure_git_repo()

        base_branch = self._current_branch()
        if base_branch is None:
            raise RuntimeError(
                "ACIDWorkspace refuses to begin on a detached HEAD: the ACID COMMIT merge "
                "would land on an unnamed commit and could be lost. Check out a branch "
                "first (e.g. `git switch -c agent-work`)."
            )
        if base_branch.startswith("acid/"):
            logger.warning(
                "ACID BEGIN: base branch %s looks like a leftover ACID transaction branch.",
                base_branch,
            )
        if self._git_stdout(["ls-files", "--unmerged"]):
            raise RuntimeError(
                "ACIDWorkspace refuses to begin while the repository has unresolved merge "
                "conflicts. Resolve or abort the merge first."
            )
        self.base_ref = base_branch

        dirty_tracked = self._dirty_tracked_paths()
        if dirty_tracked and self.dirty_worktree_policy == "fail":
            sample = ", ".join(dirty_tracked[:5])
            more = "" if len(dirty_tracked) <= 5 else f", ... +{len(dirty_tracked) - 5} more"
            raise RuntimeError(
                "ACIDWorkspace refuses to start with dirty tracked files by default "
                f"because the transaction would include them: {sample}{more}. "
                "Commit/stash first, or pass dirty_worktree_policy='allow' "
                "(abort() restores them either way)."
            )

        base_commit = self._git_stdout(["rev-parse", "--verify", "HEAD^{commit}"])
        worktree_commit, index_commit = self._capture_tracked_state(base_commit)

        created_branch: str | None = None
        try:
            self._local_snapshots = {}
            self._filesystem_savepoints = {}
            self._create_escrow(self.session.session_id)
            self._initial_local_snapshot = self._snapshot_local_state()
            self._baseline_local_manifest = dict(self._initial_local_snapshot.manifest)

            # ISOLATION: Create an isolated branch for this agent's committed changes.
            # Only HEAD's symbolic ref moves; the index and working tree are untouched.
            branch = self._available_branch_name(f"acid/{self.session.session_id}")
            self._git(["branch", "--no-track", branch, base_commit])
            created_branch = branch
            self._git(
                [
                    "symbolic-ref",
                    "-m",
                    f"acid: begin {self.session.session_id}",
                    "HEAD",
                    f"refs/heads/{branch}",
                ]
            )
        except BaseException:
            if created_branch is not None:
                self._git(["symbolic-ref", "HEAD", f"refs/heads/{base_branch}"], check=False)
                self._git(["branch", "-D", created_branch], check=False)
            self._cleanup_escrow()
            self._initial_local_snapshot = None
            self._baseline_local_manifest = {}
            raise

        self.branch_name = created_branch
        self._base_commit = base_commit
        self._begin_worktree_commit = worktree_commit
        self._begin_index_commit = index_commit
        self._internal_commits = set()

        logger.info(
            "ACID BEGIN: Session %s started on branch %s",
            self.session.session_id,
            self.branch_name,
        )
        return self.session

    def savepoint(self, step_data: dict, node_name: str = "openhands_action") -> Frame:
        """
        SAVEPOINT - every frame is a savepoint, filesystem + memory state together
        """
        self._require_active_transaction()
        previous_tip = self._branch_tip()

        # 1. Synchronize the filesystem state via git.
        commit_hash = self._commit_workspace(f"SAVEPOINT: {node_name}", allow_empty=True)
        if commit_hash is None:  # pragma: no cover - allow_empty always commits
            raise RuntimeError("Savepoint commit was not created.")

        try:
            local_snapshot = self._snapshot_local_state()

            # 2. Record state map via agent-vcr with the commit hash already attached
            # so it is durable even if the recorder flushes every frame.
            frame = self.recorder.record_step(
                node_name=node_name,
                input_state={},
                output_state=step_data,
                metadata=FrameMetadata(custom={"acid_commit_hash": commit_hash}),
            )
        except BaseException:
            # Never leave a savepoint commit without a frame referencing it.
            self._discard_commit(commit_hash, previous_tip)
            raise

        self._local_snapshots[frame.frame_id] = local_snapshot
        self._filesystem_savepoints[frame.frame_id] = commit_hash
        logger.info("ACID SAVEPOINT: Filesystem & memory synced at frame %s", frame.frame_id)

        return frame

    def rollback(self, to_frame_index: int) -> None:
        """
        ROLLBACK - actually reverts the files on disk via git + restores agent-vcr state.
        The world rewinds, not just the object. Only paths inside the workspace change.
        """
        logger.info("ACID ROLLBACK: Reverting world to frame index %s...", to_frame_index)
        self._require_active_transaction()

        # 1. Rewind the VCRRecorder
        frames = self.recorder.get_frames()
        if to_frame_index < 0 or to_frame_index >= len(frames):
            raise ValueError(f"Frame index {to_frame_index} out of bounds")

        target_frame = frames[to_frame_index]
        target_snapshot = self._local_snapshots.get(target_frame.frame_id)
        if target_snapshot is None:
            raise RuntimeError(
                "Local-file escrow metadata is unavailable for the requested savepoint."
            )

        # 2. Rewind the git workspace
        target_hash = self._filesystem_savepoints.get(target_frame.frame_id)
        if target_hash is None:
            metadata_hash = target_frame.metadata.custom.get("acid_commit_hash")
            target_hash = metadata_hash if isinstance(metadata_hash, str) else None
        if not isinstance(target_hash, str) or not target_hash:
            commit_msg = f"SAVEPOINT: {target_frame.frame_id}"
            log_out = self._git_stdout(
                ["log", f"--grep={commit_msg}", "--format=%H", self._branch_ref()]
            )
            target_hash = log_out.splitlines()[0] if log_out else None

        if not target_hash:
            # Fallback for savepoints where FS didn't mutate, rewind to the closest previous hash
            logger.warning(
                "No direct file mutation at frame %s. Rolling back to latest FS state.",
                target_frame.frame_id,
            )
            self._restore_tracked(self._branch_tip(), staged=True, worktree=True)
        else:
            verified = self._git(
                ["rev-parse", "--verify", "-q", f"{target_hash}^{{commit}}"], check=False
            )
            if verified.returncode != 0:
                raise RuntimeError(f"Savepoint commit {target_hash} no longer exists.")
            target_hash = verified.stdout.strip()
            self._git(["update-ref", "-m", "acid: rollback", self._branch_ref(), target_hash])
            self._restore_tracked(target_hash, staged=True, worktree=True)

        self._restore_local_snapshot(target_snapshot)

        # 3. Fork the in-memory recording only after filesystem restoration succeeds.
        # Fork in place: the recorder object stays the same, so framework
        # wrappers (VCRLangGraph, VCRCrewAI, ...) holding it keep recording
        # into the new branch instead of the abandoned pre-rollback session.
        self.recorder.fork(from_frame=to_frame_index, in_place=True)
        self.session = self.recorder.get_session()
        fork_frames = self.recorder.get_frames()
        if fork_frames:
            fork_frame = fork_frames[0]
            self._local_snapshots[fork_frame.frame_id] = target_snapshot
            if target_hash:
                self._filesystem_savepoints[fork_frame.frame_id] = target_hash

        logger.info("ACID ROLLBACK SUCCESS: Filesystem and memory correctly rewound.")

    def commit(self) -> None:
        """
        COMMIT - successful path gets locked in, checkpointed permanently to main branch.

        Changes made after the last savepoint are committed first. The transaction
        branch is merged into the base branch with a ``--no-ff`` style merge commit and
        then deleted. The merge commit uses the user's git identity when one is
        configured, otherwise the internal ``agent-vcr`` identity.
        """
        if not self.session:
            raise RuntimeError("No active session to commit.")
        if self.branch_name is None:
            raise RuntimeError("Branch name not set — was begin() called?")
        if self._toplevel is None or not self.base_ref:
            raise RuntimeError("Transaction repository state is unavailable.")

        # Save any final memory states
        self.recorder.save()

        # (from upstream) Refuse to merge from anywhere but the transaction branch.
        current_branch = self._git_stdout(["branch", "--show-current"])
        if current_branch != self.branch_name:
            raise RuntimeError(
                "ACID commit must run on its transaction branch "
                f"{self.branch_name!r}, not {current_branch or 'detached HEAD'!r}."
            )
        # Lock in work done after the last savepoint (only if there is any).
        self._commit_workspace("SAVEPOINT: final changes before ACID COMMIT", allow_empty=False)

        base_ref = self.base_ref
        base_full_ref = f"refs/heads/{base_ref}"
        message = f"ACID COMMIT: {self.session.session_id}"
        acid_tip = self._branch_tip()
        base_tip_result = self._git(
            ["rev-parse", "--verify", "-q", f"{base_full_ref}^{{commit}}"], check=False
        )
        if base_tip_result.returncode != 0:
            raise RuntimeError(
                f"Base branch {base_ref!r} no longer exists; transaction left on "
                f"branch {self.branch_name!r}."
            )
        base_tip = base_tip_result.stdout.strip()
        use_internal_ident = not self._user_ident_available()

        if acid_tip == base_tip:
            self._switch_head(base_ref)
        elif self._is_ancestor(base_tip, acid_tip):
            # Fast path: build the merge commit directly. The merged tree equals the
            # transaction tip, which already matches the index and working tree, so no
            # file on disk has to change.
            tree = self._git_stdout(["rev-parse", f"{acid_tip}^{{tree}}"])
            merge_commit = self._git_stdout(
                [
                    "commit-tree",
                    "--no-gpg-sign",
                    tree,
                    "-p",
                    base_tip,
                    "-p",
                    acid_tip,
                    "-m",
                    message,
                ],
                internal_ident=use_internal_ident,
            )
            self._git(["update-ref", "-m", message, base_full_ref, merge_commit, base_tip])
            self._switch_head(base_ref)
        else:
            # The base branch advanced during the transaction (e.g. from another
            # worktree): fall back to a real merge.
            self._git(["checkout", "-q", base_ref, "--"])
            self._git(
                [
                    "-c",
                    "merge.verifySignatures=false",
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    "-m",
                    message,
                    self.branch_name,
                ],
                internal_ident=use_internal_ident,
            )

        self._delete_transaction_branch(force=True)
        self._cleanup_escrow()
        logger.info(
            "ACID COMMIT: Session %s successfully merged to %s.",
            self.session.session_id,
            base_ref,
        )
        self.branch_name = None

    def savepoints(self) -> list[dict]:
        """List filesystem savepoints (frame index + git commit hash)."""
        return [
            {
                "index": i,
                "node_name": frame.node_name,
                "frame_id": frame.frame_id,
                "acid_commit_hash": self._filesystem_savepoints.get(frame.frame_id)
                or frame.metadata.custom.get("acid_commit_hash"),
            }
            for i, frame in enumerate(self.recorder.get_frames())
        ]

    def status(self) -> dict:
        """Inspectable transaction status (session, branch, savepoint count)."""
        session = self.recorder.get_session()
        return {
            "session_id": session.session_id if session else None,
            "branch": self.branch_name,
            "base_ref": self.base_ref,
            "savepoints": len(self.recorder.get_frames()),
            "workspace": str(self.workspace_dir),
        }

    def abort(self) -> None:
        """ABORT — discard the transaction: rewind files, drop the branch.

        Safe to call after ``begin()`` even with zero savepoints. Never
        touches the base ref's committed history or anything outside the
        workspace. The workspace's working tree and index are restored to
        exactly their state at ``begin()``, including uncommitted edits that
        pre-date the transaction and untracked/ignored files.
        """
        if not self.session or self.branch_name is None:
            raise RuntimeError("No active transaction to abort.")

        if self._initial_local_snapshot is None:
            raise RuntimeError("Initial local-file escrow metadata is unavailable.")
        if self._toplevel is None or not self.base_ref or self._base_commit is None:
            raise RuntimeError("Transaction repository state is unavailable.")

        worktree_commit = self._begin_worktree_commit or self._base_commit
        index_commit = self._begin_index_commit or self._base_commit
        self._restore_tracked(worktree_commit, staged=True, worktree=True)
        if index_commit != worktree_commit:
            self._restore_tracked(index_commit, staged=True, worktree=False)
        self._switch_head(self.base_ref)
        self._restore_local_snapshot(self._initial_local_snapshot)
        self._delete_transaction_branch(force=False)
        self._cleanup_escrow()
        logger.info("ACID ABORT: Session %s discarded.", self.session.session_id)
        self.session = None
        self.branch_name = None

    # ------------------------------------------------------------------
    #  Git plumbing
    # ------------------------------------------------------------------

    @property
    def _git_cwd(self) -> Path:
        return self._toplevel or self.workspace_dir

    @property
    def _workspace_pathspec(self) -> str:
        """Pathspec (relative to the repository top level) covering the workspace."""
        return self._prefix.rstrip("/") or "."

    def _git_command(self, args: list[str], *, internal_ident: bool) -> list[str]:
        command = [
            "git",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "tag.gpgSign=false",
        ]
        if internal_ident:
            command += [
                "-c",
                f"user.name={_INTERNAL_GIT_NAME}",
                "-c",
                f"user.email={_INTERNAL_GIT_EMAIL}",
            ]
        return [*command, *args]

    @staticmethod
    def _git_env(extra: Mapping[str, str] | None, *, internal_ident: bool) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key not in _GIT_LOCATION_ENV}
        env["GIT_LITERAL_PATHSPECS"] = "1"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_OPTIONAL_LOCKS"] = "0"
        if internal_ident:
            env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = _INTERNAL_GIT_NAME
            env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = _INTERNAL_GIT_EMAIL
        if extra:
            env.update(extra)
        return env

    def _run_git(
        self,
        args: list[str],
        *,
        check: bool,
        env: Mapping[str, str] | None,
        internal_ident: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        command = self._git_command(args, internal_ident=internal_ident)
        try:
            result = subprocess.run(
                command,
                cwd=self._git_cwd,
                env=self._git_env(env, internal_ident=internal_ident),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=self.git_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "ACID: git %s timed out after %.1fs", " ".join(args[:2]), self.git_timeout
            )
            raise
        if result.stderr:
            logger.debug("git %s stderr: %s", " ".join(args[:2]), result.stderr[-2000:])
        if check and result.returncode != 0:
            raise _GitCommandError(result.returncode, command, result.stdout, result.stderr)
        return result

    def _git(
        self,
        args: list[str],
        *,
        capture_output: bool = True,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        internal_ident: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        # Output is always captured (``capture_output`` is kept for compatibility) so git
        # chatter never reaches this process's stdout (MCP/hook stdio protocols).
        del capture_output
        raw = self._run_git(args, check=check, env=env, internal_ident=internal_ident)
        return subprocess.CompletedProcess(
            raw.args,
            raw.returncode,
            raw.stdout.decode("utf-8", "surrogateescape"),
            raw.stderr.decode("utf-8", "surrogateescape"),
        )

    def _git_stdout(
        self,
        args: list[str],
        *,
        env: Mapping[str, str] | None = None,
        internal_ident: bool = False,
    ) -> str:
        return self._git(args, env=env, internal_ident=internal_ident).stdout.strip()

    def _git_bytes(self, args: list[str], *, env: Mapping[str, str] | None = None) -> bytes:
        return self._run_git(args, check=True, env=env, internal_ident=False).stdout

    def _ensure_git_repo(self) -> None:
        """Initialize git and create a baseline commit when the workspace is new."""
        self._toplevel = None
        self._prefix = ""
        is_repo = self._git(["rev-parse", "--is-inside-work-tree"], check=False)
        if is_repo.returncode != 0:
            self._git(["init", "-q"])
            self._git(["symbolic-ref", "HEAD", "refs/heads/main"])
            self._git(["config", "user.name", "Agent VCR"])
            self._git(["config", "user.email", "vcr@example.com"])
        elif is_repo.stdout.strip() != "true":
            raise RuntimeError(
                f"ACIDWorkspace needs a git work tree, but {self.workspace_dir} is inside a "
                "git directory."
            )

        toplevel = self._git_stdout(["rev-parse", "--show-toplevel"])
        prefix = self._git(["rev-parse", "--show-prefix"]).stdout.strip("\n")
        self._toplevel = Path(toplevel).resolve()
        self._prefix = prefix if not prefix or prefix.endswith("/") else f"{prefix}/"

        self._ensure_local_git_excludes()

        if self._git(["rev-parse", "--verify", "-q", "HEAD"], check=False).returncode != 0:
            self._create_initial_commit()

    def _create_initial_commit(self) -> None:
        init_file = self.workspace_dir / ".acid_init"
        if not init_file.exists():
            init_file.touch()
        init_path = self._to_repo_relative(init_file.name)
        # A new workspace may already contain local credentials or configuration.
        # Seed Git with only Agent VCR's marker; begin() will escrow every other path.
        with self._temporary_index(None) as index_env:
            self._git(["add", "-f", "--", init_path], env=index_env)
            tree = self._git_stdout(["write-tree"], env=index_env)
        commit = self._git_stdout(
            ["commit-tree", "--no-gpg-sign", tree, "-m", "Initial ACID baseline"],
            internal_ident=True,
        )
        self._git(["update-ref", "-m", "acid: initial baseline", "HEAD", commit])
        self._git(["add", "-f", "--", init_path])

    def _ensure_local_git_excludes(self) -> None:
        """Keep VCR audit files out of git commits without editing user .gitignore."""
        exclude_path = self._git_private_path("info/exclude")
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text() if exclude_path.exists() else ""
        additions = [pattern for pattern in self._git_clean_excludes() if pattern not in existing]
        if not additions:
            return
        content = existing + "".join(f"\n{pattern}" for pattern in additions)
        temp_path = exclude_path.with_name(f".{exclude_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(content)
            os.replace(temp_path, exclude_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _current_ref(self) -> str:
        branch = self._current_branch()
        if branch:
            return branch
        return self._git_stdout(["rev-parse", "HEAD"])

    def _current_branch(self) -> str | None:
        """Return the checked-out branch name, or ``None`` for a detached HEAD."""
        result = self._git(["symbolic-ref", "-q", "HEAD"], check=False)
        ref = result.stdout.strip()
        if result.returncode != 0 or not ref.startswith("refs/heads/"):
            return None
        return ref[len("refs/heads/") :]

    def _branch_ref(self) -> str:
        if self.branch_name is None:
            raise RuntimeError("No active ACID transaction branch.")
        return f"refs/heads/{self.branch_name}"

    def _branch_tip(self) -> str:
        return self._git_stdout(["rev-parse", "--verify", f"{self._branch_ref()}^{{commit}}"])

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self._git(["merge-base", "--is-ancestor", ancestor, descendant], check=False)
        return result.returncode == 0

    def _user_ident_available(self) -> bool:
        return all(
            self._git(["var", variable], check=False).returncode == 0
            for variable in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT")
        )

    def _available_branch_name(self, name: str) -> str:
        if not self._branch_exists(name):
            return name
        for suffix in range(2, _MAX_BRANCH_SUFFIX):
            candidate = f"{name}-{suffix}"
            if not self._branch_exists(candidate):
                logger.warning(
                    "ACID BEGIN: branch %s already exists (stale transaction?); "
                    "leaving it untouched and using %s instead.",
                    name,
                    candidate,
                )
                return candidate
        raise RuntimeError(
            f"Too many stale ACID branches named {name}-<n>; delete them with "
            f"`git branch -D {name}...`."
        )

    def _branch_exists(self, name: str) -> bool:
        result = self._git(["rev-parse", "--verify", "-q", f"refs/heads/{name}"], check=False)
        return result.returncode == 0

    def _require_active_transaction(self) -> None:
        if self.session is None or self.branch_name is None or self._toplevel is None:
            raise RuntimeError("No active ACID transaction; call begin() first.")

    def _capture_tracked_state(self, base_commit: str) -> tuple[str, str]:
        """Snapshot dirty tracked state (worktree, index) as commits without touching it."""
        stash = self._git_stdout(
            ["stash", "create", "agent-vcr: pre-transaction state"], internal_ident=True
        )
        if not stash:
            return base_commit, base_commit
        index_commit = self._git_stdout(["rev-parse", "--verify", f"{stash}^2"])
        return stash, index_commit

    @contextlib.contextmanager
    def _temporary_index(self, seed_commit: str | None) -> Iterator[dict[str, str]]:
        """Yield env for a private index (seeded from ``seed_commit``, else empty)."""
        index_path = self._git_private_path(f"vcr-index-{uuid.uuid4().hex}")
        env = {"GIT_INDEX_FILE": str(index_path)}
        try:
            if seed_commit is not None:
                # One-tree merge into a separate file: reuses the real index's stat cache
                # but drops whatever the user staged, and never writes the real index.
                self._git(["read-tree", "--reset", f"--index-output={index_path}", seed_commit])
            yield env
        finally:
            for path in (index_path, index_path.with_name(f"{index_path.name}.lock")):
                path.unlink(missing_ok=True)

    def _build_workspace_tree(self, seed_commit: str) -> str:
        """Tree of ``seed_commit`` with the workspace's current, transaction-owned files."""
        pathspec = self._workspace_pathspec
        with self._temporary_index(seed_commit) as index_env:
            # Resetting a path after ``git add -A`` is too late: Git has already written its
            # content as an unreachable object. Never present baseline-local bytes to Git.
            self._git(["add", "-u", "--", pathspec], env=index_env)
            untracked = self._git_bytes(
                ["ls-files", "-z", "--others", "--exclude-standard", "--", pathspec],
                env=index_env,
            )
            generated: list[str] = []
            for raw in untracked.split(b"\0"):
                if not raw or raw.endswith(b"/"):
                    continue  # empty record, or a nested repository
                repo_path = os.fsdecode(raw)
                relative_path = self._to_workspace_relative(repo_path)
                if not relative_path:
                    continue
                if self._is_baseline_local_path(relative_path) or self._is_clean_excluded(
                    relative_path
                ):
                    continue
                generated.append(repo_path)
            for offset in range(0, len(generated), _ADD_BATCH_SIZE):
                self._git(
                    ["add", "--", *generated[offset : offset + _ADD_BATCH_SIZE]],
                    env=index_env,
                )
            return self._git_stdout(["write-tree"], env=index_env)

    def _commit_workspace(self, message: str, *, allow_empty: bool) -> str | None:
        """Commit the workspace onto the transaction branch; return the new commit."""
        branch_ref = self._branch_ref()
        parent = self._branch_tip()
        tree = self._build_workspace_tree(parent)
        if not allow_empty and tree == self._git_stdout(["rev-parse", f"{parent}^{{tree}}"]):
            return None
        commit = self._git_stdout(
            ["commit-tree", "--no-gpg-sign", tree, "-p", parent, "-m", message],
            internal_ident=True,
        )
        self._git(["update-ref", "-m", message, branch_ref, commit, parent])
        self._internal_commits.add(commit)
        self._sync_index(commit)
        return commit

    def _discard_commit(self, commit: str, previous_tip: str) -> None:
        try:
            self._git(
                ["update-ref", "-m", "acid: discard savepoint", self._branch_ref(), previous_tip]
            )
            self._sync_index(previous_tip)
            self._internal_commits.discard(commit)
        except Exception:
            logger.warning("ACID: failed to discard incomplete savepoint %s", commit, exc_info=True)

    def _sync_index(self, commit: str) -> None:
        """Make the real index match ``commit`` for workspace paths only."""
        self._git(["reset", "-q", commit, "--", self._workspace_pathspec])

    def _restore_tracked(self, source: str, *, staged: bool, worktree: bool) -> None:
        """Restore tracked workspace paths from ``source``; never touches other paths."""
        pathspec = self._workspace_pathspec
        args = ["restore", "-q", f"--source={source}"]
        if staged:
            args.append("--staged")
        if worktree:
            args.append("--worktree")
        result = self._run_git([*args, "--", pathspec], check=False, env=None, internal_ident=False)
        if result.returncode == 0:
            return
        # ``git restore`` fails when the pathspec matches nothing in either the index or
        # the source tree; that simply means there is nothing to restore.
        in_index = self._git_bytes(["ls-files", "-z", "--", pathspec])
        in_source = self._git_bytes(["ls-tree", "-z", "--name-only", source, "--", pathspec])
        if in_index or in_source:
            raise _GitCommandError(result.returncode, result.args, result.stdout, result.stderr)

    def _switch_head(self, branch: str) -> None:
        """Point HEAD at ``branch`` without touching the index or working tree."""
        self._git(["symbolic-ref", "-m", "acid: restore base", "HEAD", f"refs/heads/{branch}"])

    def _delete_transaction_branch(self, *, force: bool) -> None:
        if self.branch_name is None:
            return
        branch_ref = self._branch_ref()
        if not force and self._base_commit is not None:
            listed = self._git_stdout(["rev-list", f"{self._base_commit}..{branch_ref}"])
            foreign = [sha for sha in listed.split() if sha not in self._internal_commits]
            if foreign:
                logger.warning(
                    "ACID: keeping branch %s because it contains %d commit(s) not created "
                    "by ACIDWorkspace.",
                    self.branch_name,
                    len(foreign),
                )
                return
        result = self._git(["branch", "-D", self.branch_name], check=False)
        if result.returncode != 0:
            logger.warning(
                "ACID: could not delete branch %s: %s", self.branch_name, result.stderr.strip()
            )

    def _dirty_tracked_paths(self) -> list[str]:
        """Workspace-relative tracked paths with uncommitted (staged or unstaged) changes."""
        status = self._git_bytes(
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=no",
                "--",
                self._workspace_pathspec,
            ]
        )
        paths: list[str] = []
        for record in self._iter_status_records(status):
            code, repo_path = record
            if code in ("??", "!!") or not code.strip():
                continue
            relative_path = self._to_workspace_relative(repo_path)
            if relative_path:
                paths.append(relative_path)
        return paths

    @staticmethod
    def _iter_status_records(status: bytes) -> Iterator[tuple[str, str]]:
        """Parse ``git status --porcelain=v1 -z`` into (XY, repo-relative path)."""
        records = status.split(b"\0")
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if len(record) < 4:
                continue
            code = record[:2].decode("ascii", "replace")
            yield code, os.fsdecode(record[3:])
            if code[0] in "RC":
                index += 1  # skip the rename/copy source path

    def _local_untracked_or_ignored_paths(self) -> set[str]:
        """Return local paths not tracked by git, including ignored files."""
        status = self._git_bytes(
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored",
                "--",
                self._workspace_pathspec,
            ]
        )
        paths: set[str] = set()
        for code, repo_path in self._iter_status_records(status):
            if code not in ("??", "!!"):
                continue
            relative_path = self._to_workspace_relative(repo_path)
            if relative_path is None:
                continue
            if relative_path == "":
                # The whole workspace is untracked/ignored in the enclosing repository.
                with os.scandir(self.workspace_dir) as entries:
                    paths.update(entry.name for entry in entries if entry.name != ".git")
                continue
            paths.add(self._normalize_relative_path(relative_path))
        return paths

    def _to_workspace_relative(self, repo_path: str) -> str | None:
        """Map a repository-relative path to a workspace-relative one (None if outside)."""
        path = repo_path.rstrip("/")
        if not self._prefix:
            return path
        if path == self._prefix.rstrip("/"):
            return ""
        if path.startswith(self._prefix):
            return path[len(self._prefix) :]
        return None

    def _to_repo_relative(self, relative_path: str) -> str:
        return f"{self._prefix}{self._normalize_relative_path(relative_path)}"

    # ------------------------------------------------------------------
    #  Local-file escrow
    # ------------------------------------------------------------------

    def _create_escrow(self, session_id: str) -> None:
        root = self._git_private_path("vcr-escrow")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(root, 0o700)

        session_dir = root / f"{session_id}-{uuid.uuid4().hex}"
        session_dir.mkdir(mode=0o700)
        blobs_dir = session_dir / "blobs"
        blobs_dir.mkdir(mode=0o700)
        self._escrow_dir = session_dir
        self._escrow_blobs_dir = blobs_dir

    def _cleanup_escrow(self) -> None:
        """Remove this transaction's private escrow; safe to call repeatedly."""
        escrow_dir = self._escrow_dir
        self._escrow_dir = None
        self._escrow_blobs_dir = None
        if escrow_dir is not None and escrow_dir.exists():
            shutil.rmtree(escrow_dir)

    def _snapshot_local_state(self) -> _LocalSnapshot:
        """Escrow restorable untracked/ignored content for one rollback boundary.

        Files that disappear while being snapshotted are left out; files that keep
        changing are recorded as ``unstable`` (preserved, never deleted, on restore).
        Neither aborts the snapshot.
        """
        candidates, skipped = self._scan_local_paths()
        manifest: dict[str, _LocalFileMetadata] = {}
        vanished: list[str] = []

        for relative_path in sorted(candidates):
            path = self._workspace_path(relative_path)
            try:
                entry = self._escrow_local_path(path)
            except FileNotFoundError:
                vanished.append(relative_path)
                continue
            except _EscrowFileChanged:
                skipped[relative_path] = "unstable"
                continue
            except PermissionError:
                skipped[relative_path] = "unreadable"
                continue
            if entry is None:
                skipped[relative_path] = "oversized"
            else:
                manifest[relative_path] = entry

        if vanished:
            logger.warning(
                "ACID escrow skipped %d local path(s) that vanished while being snapshotted: %s",
                len(vanished),
                ", ".join(vanished[:5]),
            )
        non_exclusion_skips = {
            path: reason for path, reason in skipped.items() if reason != "excluded"
        }
        if non_exclusion_skips:
            logger.warning(
                "ACID escrow preserved %d local path(s) without content snapshots: %s",
                len(non_exclusion_skips),
                ", ".join(
                    f"{path} ({non_exclusion_skips[path]})"
                    for path in sorted(non_exclusion_skips)[:5]
                ),
            )

        return _LocalSnapshot(
            manifest=manifest,
            skipped_paths=skipped,
            directories=frozenset(self._existing_directory_paths()),
        )

    def _escrow_local_path(self, path: Path) -> _LocalFileMetadata | None:
        """Escrow one regular file or symlink, retrying if it changes mid-copy."""
        last_error: Exception | None = None
        for _attempt in range(_ESCROW_ATTEMPTS):
            try:
                file_stat = path.lstat()
                if stat.S_ISLNK(file_stat.st_mode):
                    target = os.readlink(path)
                    target_bytes = os.fsencode(target)
                    return _LocalFileMetadata(
                        kind="symlink",
                        sha256=hashlib.sha256(target_bytes).hexdigest(),
                        mode=stat.S_IMODE(file_stat.st_mode),
                        symlink_target=target,
                        size=len(target_bytes),
                    )
                if not stat.S_ISREG(file_stat.st_mode):
                    raise _EscrowFileChanged(f"Local path is no longer a regular file: {path}")
                return self._escrow_regular_file(path)
            except FileNotFoundError:
                raise
            except _EscrowFileChanged as exc:
                last_error = exc
            except OSError as exc:
                # The path was swapped for a symlink/directory between lstat and open.
                if exc.errno not in (errno.ELOOP, errno.EISDIR, errno.ENOTDIR, errno.EINVAL):
                    raise
                last_error = exc
        raise _EscrowFileChanged(str(last_error))

    def _scan_local_paths(self) -> tuple[dict[str, os.stat_result], dict[str, str]]:
        candidates: dict[str, os.stat_result] = {}
        skipped: dict[str, str] = {}

        def visit(relative_path: str) -> None:
            path = self._workspace_path(relative_path)
            try:
                file_stat = path.lstat()
            except FileNotFoundError:
                return

            if self._is_clean_excluded(relative_path):
                skipped[relative_path] = "excluded"
                return
            if stat.S_ISDIR(file_stat.st_mode):
                try:
                    with os.scandir(path) as entries:
                        names = sorted(entry.name for entry in entries)
                except (FileNotFoundError, NotADirectoryError):
                    return
                for name in names:
                    visit(f"{relative_path}/{name}" if relative_path else name)
                return
            if stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
                candidates[relative_path] = file_stat
                return
            skipped[relative_path] = "special-file"

        for relative_path in sorted(self._local_untracked_or_ignored_paths()):
            visit(relative_path)
        return candidates, skipped

    def _escrow_regular_file(self, source: Path) -> _LocalFileMetadata | None:
        if self._escrow_blobs_dir is None:
            raise RuntimeError("Local-file escrow has not been initialized.")

        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        temp_descriptor, temp_name = tempfile.mkstemp(
            prefix=".vcr-blob-", dir=self._escrow_blobs_dir
        )
        temp_path = Path(temp_name)
        digest = hashlib.sha256()
        total = 0
        try:
            before = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb") as input_file, os.fdopen(
                temp_descriptor, "wb"
            ) as output_file:
                descriptor = -1
                temp_descriptor = -1
                while chunk := input_file.read(_COPY_CHUNK_BYTES):
                    total += len(chunk)
                    if total > self.max_untracked_file_bytes:
                        return None
                    digest.update(chunk)
                    output_file.write(chunk)
                output_file.flush()
                os.fsync(output_file.fileno())
                after = os.fstat(input_file.fileno())

            before_signature = (
                before.st_size,
                before.st_mtime_ns,
                stat.S_IMODE(before.st_mode),
            )
            after_signature = (
                after.st_size,
                after.st_mtime_ns,
                stat.S_IMODE(after.st_mode),
            )
            if before_signature != after_signature or total != after.st_size:
                raise _EscrowFileChanged(f"Local file changed while being escrowed: {source}")

            sha256 = digest.hexdigest()
            blob_path = self._escrow_blobs_dir / sha256
            os.chmod(temp_path, 0o600)
            if blob_path.exists():
                temp_path.unlink()
            else:
                os.replace(temp_path, blob_path)
                os.chmod(blob_path, 0o600)
            return _LocalFileMetadata(
                kind="regular",
                sha256=sha256,
                mode=stat.S_IMODE(after.st_mode),
                symlink_target=None,
                size=total,
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temp_descriptor >= 0:
                os.close(temp_descriptor)
            temp_path.unlink(missing_ok=True)

    def _restore_local_snapshot(self, snapshot: _LocalSnapshot) -> None:
        """Prune generated local files and atomically restore escrowed paths."""
        candidates, _ = self._scan_local_paths()
        for relative_path, file_stat in sorted(candidates.items()):
            if relative_path in snapshot.manifest or self._snapshot_preserves(
                snapshot, relative_path
            ):
                continue
            if stat.S_ISREG(file_stat.st_mode) and (
                file_stat.st_size > self.max_untracked_file_bytes
            ):
                continue
            self._remove_path(self._workspace_path(relative_path))

        current_directories = self._existing_directory_paths()
        for relative_path in sorted(
            current_directories - set(snapshot.directories),
            key=lambda value: (value.count("/"), value),
            reverse=True,
        ):
            if self._is_clean_excluded(relative_path) or self._snapshot_preserves(
                snapshot, relative_path
            ):
                continue
            with contextlib.suppress(OSError):
                self._workspace_path(relative_path).rmdir()

        for relative_path in sorted(snapshot.directories, key=lambda value: value.count("/")):
            self._ensure_directory(self._workspace_path(relative_path))

        for relative_path, entry in sorted(snapshot.manifest.items()):
            destination = self._workspace_path(relative_path)
            if not self._local_entry_matches(destination, entry):
                self._restore_local_entry(destination, entry)

    def _restore_local_entry(self, destination: Path, entry: _LocalFileMetadata) -> None:
        self._ensure_directory(destination.parent)
        if destination.exists() and destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)

        if entry.kind == "symlink":
            if entry.symlink_target is None:
                raise RuntimeError(f"Escrow metadata is missing a symlink target: {destination}")
            temp_path = destination.parent / f".vcr-restore-{uuid.uuid4().hex}"
            try:
                os.symlink(entry.symlink_target, temp_path)
                os.replace(temp_path, destination)
            finally:
                if os.path.lexists(temp_path):
                    temp_path.unlink()
            return

        if self._escrow_blobs_dir is None:
            raise RuntimeError("Local-file escrow has already been cleaned up.")
        blob_path = self._escrow_blobs_dir / entry.sha256
        if not blob_path.is_file() or self._sha256_file(blob_path) != entry.sha256:
            raise RuntimeError(f"Escrow blob is missing or corrupt: {entry.sha256}")

        descriptor, temp_name = tempfile.mkstemp(prefix=".vcr-restore-", dir=destination.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "wb") as output_file, blob_path.open("rb") as input_file:
                shutil.copyfileobj(input_file, output_file, length=_COPY_CHUNK_BYTES)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.chmod(temp_path, entry.mode)
            os.replace(temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)

    def _local_entry_matches(self, path: Path, entry: _LocalFileMetadata) -> bool:
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            return False
        if entry.kind == "symlink":
            if not stat.S_ISLNK(file_stat.st_mode):
                return False
            target = os.readlink(path)
            return (
                target == entry.symlink_target
                and hashlib.sha256(os.fsencode(target)).hexdigest() == entry.sha256
            )
        if not stat.S_ISREG(file_stat.st_mode):
            return False
        return (
            file_stat.st_size == entry.size
            and stat.S_IMODE(file_stat.st_mode) == entry.mode
            and self._sha256_file(path) == entry.sha256
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as input_file:
            while chunk := input_file.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    def _existing_directory_paths(self) -> set[str]:
        directories: set[str] = set()

        def visit(directory: Path, relative_parent: str) -> None:
            try:
                with os.scandir(directory) as entries:
                    names = sorted(entry.name for entry in entries)
            except (FileNotFoundError, NotADirectoryError):
                return
            for name in names:
                relative_path = f"{relative_parent}/{name}" if relative_parent else name
                path = directory / name
                try:
                    file_stat = path.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISDIR(file_stat.st_mode) or self._is_clean_excluded(relative_path):
                    continue
                directories.add(relative_path)
                visit(path, relative_path)

        visit(self.workspace_dir, "")
        return directories

    def _ensure_directory(self, directory: Path) -> None:
        if directory == self.workspace_dir:
            return
        relative = directory.relative_to(self.workspace_dir)
        current = self.workspace_dir
        for part in relative.parts:
            current /= part
            if os.path.lexists(current):
                if current.is_symlink() or not current.is_dir():
                    self._remove_path(current)
                    current.mkdir()
            else:
                current.mkdir()

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not os.path.lexists(path):
            return
        with contextlib.suppress(FileNotFoundError):
            if path.is_symlink() or not path.is_dir():
                path.unlink()
            else:
                shutil.rmtree(path)

    def _workspace_path(self, relative_path: str) -> Path:
        normalized = self._normalize_relative_path(relative_path)
        candidate = self.workspace_dir.joinpath(*PurePosixPath(normalized).parts)

        # Validate the resolved parent rather than the final path so a symlink
        # itself may be escrowed without following its target. This also blocks
        # a nested path from escaping through a pre-existing directory symlink.
        try:
            candidate.parent.resolve().relative_to(self.workspace_dir)
        except ValueError as err:
            raise ValueError(
                f"Workspace path resolves outside the repository: {relative_path!r}"
            ) from err
        return candidate

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        normalized = PurePosixPath(path.rstrip("/"))
        if normalized.is_absolute() or not normalized.parts or ".." in normalized.parts:
            raise ValueError(f"Unsafe workspace-relative path: {path!r}")
        return normalized.as_posix()

    def _git_private_path(self, relative_path: str) -> Path:
        git_path = Path(self._git_stdout(["rev-parse", "--git-path", relative_path]))
        if not git_path.is_absolute():
            git_path = self._git_cwd / git_path
        return git_path

    def _is_clean_excluded(self, path: str) -> bool:
        normalized = self._normalize_relative_path(path)
        parts = PurePosixPath(normalized).parts
        for exclude in self.untracked_exclude_dirs:
            exclude_path = PurePosixPath(exclude.strip("/"))
            if len(exclude_path.parts) == 1 and exclude_path.as_posix() in parts:
                return True
            exclude_normalized = exclude_path.as_posix()
            if normalized == exclude_normalized or normalized.startswith(f"{exclude_normalized}/"):
                return True

        for exclude in self._dynamic_clean_excludes():
            if normalized == exclude or normalized.startswith(f"{exclude}/"):
                return True
        return False

    def _is_baseline_local_path(self, path: str) -> bool:
        if self._initial_local_snapshot is None:
            return False
        normalized = self._normalize_relative_path(path)
        return normalized in self._baseline_local_manifest or self._snapshot_preserves(
            self._initial_local_snapshot, normalized
        )

    @staticmethod
    def _snapshot_preserves(snapshot: _LocalSnapshot, path: str) -> bool:
        return any(
            path == skipped or path.startswith(f"{skipped}/")
            for skipped in snapshot.skipped_paths
        )

    def _git_clean_excludes(self) -> list[str]:
        """Return patterns stored in the repository-local Git exclude file.

        Single-component names match at any depth; multi-component workspace paths are
        anchored to the workspace inside the repository.
        """
        excludes: list[str] = []
        for exclude in sorted(self.untracked_exclude_dirs - {".git"}):
            exclude_path = PurePosixPath(exclude.strip("/"))
            if len(exclude_path.parts) == 1:
                excludes.append(f"{exclude_path.as_posix()}/")
            else:
                excludes.append(f"{self._prefix}{exclude_path.as_posix()}/")

        excludes.extend(f"{self._prefix}{path}/" for path in self._dynamic_clean_excludes())
        return list(dict.fromkeys(excludes))

    def _dynamic_clean_excludes(self) -> list[str]:
        excludes: list[str] = []

        recorder_dir = self.recorder.output_dir
        if not recorder_dir.is_absolute():
            recorder_dir = recorder_dir.resolve()

        try:
            relative = recorder_dir.relative_to(self.workspace_dir)
        except ValueError:
            return excludes

        pattern = relative.as_posix().rstrip("/")
        if pattern and pattern != "." and pattern not in excludes:
            excludes.append(pattern)
        return excludes
