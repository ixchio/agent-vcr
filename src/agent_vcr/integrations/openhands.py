from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Iterable
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
    """
    def __init__(
        self,
        workspace_dir: str,
        recorder: VCRRecorder | None = None,
        dirty_worktree_policy: Literal["fail", "allow"] = "fail",
        max_untracked_file_bytes: int = 10 * 1024 * 1024,
        untracked_exclude_dirs: Iterable[str] | None = None,
    ) -> None:
        if max_untracked_file_bytes < 0:
            raise ValueError("max_untracked_file_bytes must be non-negative")

        self.workspace_dir = Path(workspace_dir).resolve()
        self.recorder = recorder or VCRRecorder(auto_save=True)
        self.dirty_worktree_policy = dirty_worktree_policy
        self.max_untracked_file_bytes = max_untracked_file_bytes
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

    def begin(self, session_id: str | None = None) -> Session:
        """
        BEGIN - Starts a session and snapshots local, non-git-owned files.
        """
        self.session = self.recorder.start_session(session_id)
        if self.session is None:
            raise RuntimeError("Session failed to start")

        # Validate session ID before using it in git commands
        self._sanitize_session_id(self.session.session_id)

        self._ensure_git_repo()
        self.base_ref = self._current_ref()

        dirty_tracked = self._dirty_tracked_paths()
        if dirty_tracked and self.dirty_worktree_policy == "fail":
            sample = ", ".join(dirty_tracked[:5])
            more = "" if len(dirty_tracked) <= 5 else f", ... +{len(dirty_tracked) - 5} more"
            raise RuntimeError(
                "ACIDWorkspace refuses to start with dirty tracked files by default "
                f"because rollback would discard them: {sample}{more}. "
                "Commit/stash first, or pass dirty_worktree_policy='allow'."
            )

        try:
            self._create_escrow(self.session.session_id)
            self._initial_local_snapshot = self._snapshot_local_state()
            self._baseline_local_manifest = dict(self._initial_local_snapshot.manifest)

            # ISOLATION: Create an isolated branch for this agent's committed changes.
            self.branch_name = f"acid/{self.session.session_id}"
            self._git(["checkout", "-b", self.branch_name])
        except Exception:
            self._cleanup_escrow()
            self._initial_local_snapshot = None
            self._baseline_local_manifest = {}
            raise

        logger.info(f"ACID BEGIN: Session {self.session.session_id} started on branch {self.branch_name}")
        return self.session

    def savepoint(self, step_data: dict, node_name: str = "openhands_action") -> Frame:
        """
        SAVEPOINT - every frame is a savepoint, filesystem + memory state together
        """
        # 1. Synchronize the filesystem state via git.
        self._stage_transaction_files()
        commit_msg = f"SAVEPOINT: {node_name}"
        self._git(["commit", "--allow-empty", "-m", commit_msg])
        commit_hash = self._git_stdout(["rev-parse", "HEAD"])
        local_snapshot = self._snapshot_local_state()

        # 2. Record state map via agent-vcr with the commit hash already attached
        # so it is durable even if the recorder flushes every frame.
        frame = self.recorder.record_step(
            node_name=node_name,
            input_state={},
            output_state=step_data,
            metadata=FrameMetadata(custom={"acid_commit_hash": commit_hash}),
        )
        self._local_snapshots[frame.frame_id] = local_snapshot
        self._filesystem_savepoints[frame.frame_id] = commit_hash
        logger.info(f"ACID SAVEPOINT: Filesystem & memory synced at frame {frame.frame_id}")

        return frame

    def rollback(self, to_frame_index: int) -> None:
        """
        ROLLBACK - actually reverts the files on disk via git + restores agent-vcr state.
        The world rewinds, not just the object.
        """
        logger.info(f"ACID ROLLBACK: Reverting world to frame index {to_frame_index}...")

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
            log_out = self._git_stdout(["log", f"--grep={commit_msg}", "--format=%H"])
            target_hash = log_out.splitlines()[0] if log_out else None

        if not target_hash:
            # Fallback for savepoints where FS didn't mutate, rewind to the closest previous hash
            logger.warning(f"No direct file mutation at frame {target_frame.frame_id}. Rolling back to latest FS state.")
            self._git(["reset", "--hard"])
        else:
            self._git(["reset", "--hard", target_hash])

        self._restore_local_snapshot(target_snapshot)

        # 3. Fork the in-memory recording only after filesystem restoration succeeds.
        self.recorder = self.recorder.fork(from_frame=to_frame_index)
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
        """
        if not self.session:
            raise RuntimeError("No active session to commit.")
        if self.branch_name is None:
            raise RuntimeError("Branch name not set — was begin() called?")

        # Save any final memory states
        self.recorder.save()

        # A savepoint is not required immediately before commit. Finalize every
        # Git-owned change on the isolated transaction branch before switching
        # refs; otherwise Git carries dirty tracked files across checkout and a
        # successful merge can omit the agent's latest work.
        current_branch = self._git_stdout(["branch", "--show-current"])
        if current_branch != self.branch_name:
            raise RuntimeError(
                "ACID commit must run on its transaction branch "
                f"{self.branch_name!r}, not {current_branch or 'detached HEAD'!r}."
            )
        self._stage_transaction_files()
        staged = self._git(["diff", "--cached", "--quiet"], check=False)
        if staged.returncode == 1:
            self._git(["commit", "-m", f"ACID FINALIZE: {self.session.session_id}"])
        elif staged.returncode != 0:
            staged.check_returncode()

        dirty_tracked = self._dirty_tracked_paths()
        if dirty_tracked:
            sample = ", ".join(dirty_tracked[:5])
            raise RuntimeError(
                f"ACID commit could not finalize all tracked workspace changes: {sample}"
            )

        # Merge the branch back to the original branch/ref.
        # Branch names are safe — validated by _sanitize_session_id().
        self._git(["checkout", self.base_ref or "main"])
        self._git(["merge", "--no-ff", "-m", f"ACID COMMIT: {self.session.session_id}", self.branch_name])
        self._cleanup_escrow()
        logger.info(
            "ACID COMMIT: Session %s successfully merged to %s.",
            self.session.session_id,
            self.base_ref or "main",
        )

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
        touches the base ref's committed history; only the acid branch
        (plus generated untracked files) is removed.
        """
        if not self.session or self.branch_name is None:
            raise RuntimeError("No active transaction to abort.")

        if self._initial_local_snapshot is None:
            raise RuntimeError("Initial local-file escrow metadata is unavailable.")

        self._git(["reset", "--hard"])
        self._git(["checkout", self.base_ref or "main"])
        self._restore_local_snapshot(self._initial_local_snapshot)
        self._git(["branch", "-D", self.branch_name], check=False)
        self._cleanup_escrow()
        logger.info("ACID ABORT: Session %s discarded.", self.session.session_id)
        self.session = None
        self.branch_name = None

    def _git(
        self,
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.workspace_dir,
            capture_output=capture_output,
            text=True,
            check=check,
        )

    def _git_stdout(self, args: list[str]) -> str:
        return self._git(args, capture_output=True).stdout.strip()

    def _git_bytes(self, args: list[str]) -> bytes:
        result = subprocess.run(
            ["git", *args],
            cwd=self.workspace_dir,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def _ensure_git_repo(self) -> None:
        """Initialize git and create a baseline commit when the workspace is new."""
        is_repo = self._git(
            ["rev-parse", "--is-inside-work-tree"], capture_output=True, check=False
        )
        if is_repo.returncode != 0:
            self._git(["init"])
            self._git(["config", "user.name", "Agent VCR"])
            self._git(["config", "user.email", "vcr@example.com"])
            self._git(["branch", "-m", "main"])

        self._ensure_local_git_excludes()

        if self._git(["rev-parse", "--verify", "HEAD"], check=False).returncode != 0:
            init_file = self.workspace_dir / ".acid_init"
            if not init_file.exists():
                init_file.touch()
            # A new workspace may already contain local credentials or configuration.
            # Seed Git with only Agent VCR's marker; begin() will escrow every other path.
            self._git(["add", "--", init_file.name])
            self._git(["commit", "-m", "Initial ACID baseline"])

    def _ensure_local_git_excludes(self) -> None:
        """Keep VCR audit files out of git commits without editing user .gitignore."""
        exclude_path = self._git_private_path("info/exclude")
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_path.read_text() if exclude_path.exists() else ""
        additions = [pattern for pattern in self._git_clean_excludes() if pattern not in existing]
        if additions:
            with open(exclude_path, "a") as f:
                for pattern in additions:
                    f.write(f"\n{pattern}")

    def _current_ref(self) -> str:
        branch = self._git_stdout(["branch", "--show-current"])
        if branch:
            return branch
        return self._git_stdout(["rev-parse", "HEAD"])

    def _dirty_tracked_paths(self) -> list[str]:
        status = self._git_stdout(["status", "--porcelain=v1", "--untracked-files=no"])
        paths = []
        for line in status.splitlines():
            if not line or not line[:2].strip():
                continue
            paths.append(line[3:].strip())
        return paths

    def _local_untracked_or_ignored_paths(self) -> set[str]:
        """Return local paths not tracked by git, including ignored files."""
        status = self._git_bytes(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored"]
        )
        paths: set[str] = set()
        for record in status.split(b"\0"):
            if record.startswith((b"?? ", b"!! ")):
                paths.add(self._normalize_relative_path(os.fsdecode(record[3:])))
        return paths

    def _stage_transaction_files(self) -> None:
        # Resetting a path after ``git add -A`` is too late: Git has already written its
        # content as an unreachable object. Never present baseline-local bytes to Git.
        self._git(["add", "-u"])
        untracked = self._git_bytes(["ls-files", "--others", "--exclude-standard", "-z"])
        generated = [
            path
            for path in (
                self._normalize_relative_path(os.fsdecode(raw))
                for raw in untracked.split(b"\0")
                if raw
            )
            if not self._is_baseline_local_path(path) and not self._is_clean_excluded(path)
        ]
        for offset in range(0, len(generated), 128):
            self._git(["add", "--", *generated[offset : offset + 128]])

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
        """Escrow restorable untracked/ignored content for one rollback boundary."""
        candidates, skipped = self._scan_local_paths()
        manifest: dict[str, _LocalFileMetadata] = {}

        for relative_path in sorted(candidates):
            path = self._workspace_path(relative_path)
            file_stat = path.lstat()
            if stat.S_ISLNK(file_stat.st_mode):
                target = os.readlink(path)
                target_bytes = os.fsencode(target)
                manifest[relative_path] = _LocalFileMetadata(
                    kind="symlink",
                    sha256=hashlib.sha256(target_bytes).hexdigest(),
                    mode=stat.S_IMODE(file_stat.st_mode),
                    symlink_target=target,
                    size=len(target_bytes),
                )
                continue

            entry = self._escrow_regular_file(path)
            if entry is None:
                skipped[relative_path] = "oversized"
            else:
                manifest[relative_path] = entry

        non_exclusion_skips = {
            path: reason for path, reason in skipped.items() if reason != "excluded"
        }
        if non_exclusion_skips:
            logger.warning(
                "ACID escrow preserved %d local path(s) without content snapshots: %s",
                len(non_exclusion_skips),
                ", ".join(sorted(non_exclusion_skips)[:5]),
            )

        return _LocalSnapshot(
            manifest=manifest,
            skipped_paths=skipped,
            directories=frozenset(self._existing_directory_paths()),
        )

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
                with os.scandir(path) as entries:
                    names = sorted(entry.name for entry in entries)
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
                raise RuntimeError(f"Local file changed while being escrowed: {source}")

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
            with os.scandir(directory) as entries:
                names = sorted(entry.name for entry in entries)
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
            git_path = self.workspace_dir / git_path
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
        """Return patterns stored in the repository-local Git exclude file."""
        excludes = [
            exclude.strip("/") + "/"
            for exclude in sorted(self.untracked_exclude_dirs - {".git"})
        ]

        excludes.extend(f"{path}/" for path in self._dynamic_clean_excludes())
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
        if pattern not in excludes:
            excludes.append(pattern)
        return excludes
