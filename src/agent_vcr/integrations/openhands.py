import logging
import re
import subprocess
from pathlib import Path
from typing import Literal, Optional

from agent_vcr.models import Frame, FrameMetadata, Session
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)

# Strict whitelist: only alphanumerics, hyphens, underscores, and dots.
_SAFE_SESSION_ID = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class ACIDWorkspace:
    """
    Provides ACID transaction semantics for agent execution.
    Combines Agent VCR state snapshotting with git-backed filesystem freezing to allow
    full-world rollbacks.
    """
    def __init__(
        self,
        workspace_dir: str,
        recorder: Optional[VCRRecorder] = None,
        dirty_worktree_policy: Literal["fail", "allow"] = "fail",
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.recorder = recorder or VCRRecorder(auto_save=True)
        self.dirty_worktree_policy = dirty_worktree_policy
        self.session: Optional[Session] = None
        self.branch_name: Optional[str] = None
        self.base_ref: Optional[str] = None
        self._baseline_local_paths: set[str] = set()

        self.workspace_dir.mkdir(parents=True, exist_ok=True)

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

    def begin(self, session_id: Optional[str] = None) -> Session:
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
        self._baseline_local_paths = self._local_untracked_or_ignored_paths()

        dirty_tracked = self._dirty_tracked_paths()
        if dirty_tracked and self.dirty_worktree_policy == "fail":
            sample = ", ".join(dirty_tracked[:5])
            more = "" if len(dirty_tracked) <= 5 else f", ... +{len(dirty_tracked) - 5} more"
            raise RuntimeError(
                "ACIDWorkspace refuses to start with dirty tracked files by default "
                f"because rollback would discard them: {sample}{more}. "
                "Commit/stash first, or pass dirty_worktree_policy='allow'."
            )

        # ISOLATION: Create an isolated branch for this agent's uncommitted file changes.
        # The session ID is whitelist-validated by _sanitize_session_id() above,
        # so git argument injection (e.g. --orphan) is not possible here.
        self.branch_name = f"acid/{self.session.session_id}"
        self._git(["checkout", "-b", self.branch_name])

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

        # 2. Record state map via agent-vcr with the commit hash already attached
        # so it is durable even if the recorder flushes every frame.
        frame = self.recorder.record_step(
            node_name=node_name,
            input_state={},
            output_state=step_data,
            metadata=FrameMetadata(custom={"acid_commit_hash": commit_hash}),
        )
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
        self.recorder = self.recorder.fork(from_frame=to_frame_index)
        self.session = self.recorder.get_session()

        # 2. Rewind the git workspace
        target_hash = target_frame.metadata.custom.get("acid_commit_hash")
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

        self._clean_generated_local_paths()

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

        # Merge the branch back to the original branch/ref.
        # Branch names are safe — validated by _sanitize_session_id().
        self._git(["checkout", self.base_ref or "main"])
        self._git(["merge", "--no-ff", "-m", f"ACID COMMIT: {self.session.session_id}", self.branch_name])
        logger.info(
            "ACID COMMIT: Session %s successfully merged to %s.",
            self.session.session_id,
            self.base_ref or "main",
        )

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

    def _ensure_git_repo(self) -> None:
        """Initialize git and create a baseline commit when the workspace is new."""
        if not (self.workspace_dir / ".git").exists():
            self._git(["init"])
            self._git(["config", "user.name", "Agent VCR"])
            self._git(["config", "user.email", "vcr@example.com"])
            self._git(["branch", "-m", "main"])

        self._ensure_local_git_excludes()

        if self._git(["rev-parse", "--verify", "HEAD"], check=False).returncode != 0:
            init_file = self.workspace_dir / ".acid_init"
            if not init_file.exists():
                init_file.touch()
            self._git(["add", "-A"])
            self._git(["commit", "-m", "Initial ACID baseline"])

    def _ensure_local_git_excludes(self) -> None:
        """Keep VCR audit files out of git commits without editing user .gitignore."""
        exclude_path = self.workspace_dir / ".git" / "info" / "exclude"
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
        status = self._git_stdout(
            ["status", "--porcelain=v1", "--untracked-files=all", "--ignored"]
        )
        paths: set[str] = set()
        for line in status.splitlines():
            if line.startswith("?? ") or line.startswith("!! "):
                paths.add(line[3:].strip())
        return paths

    def _stage_transaction_files(self) -> None:
        self._git(["add", "-A"])
        staged = self._git_stdout(["diff", "--cached", "--name-only"])
        protected = [
            path
            for path in staged.splitlines()
            if self._is_baseline_local_path(path)
        ]
        if protected:
            self._git(["reset", "-q", "--", *protected])

    def _clean_generated_local_paths(self) -> None:
        """Delete only local files introduced after begin(), preserving user-owned files."""
        current_paths = self._local_untracked_or_ignored_paths()
        generated = sorted(
            path
            for path in current_paths - self._baseline_local_paths
            if not self._is_clean_excluded(path)
        )
        if not generated:
            return
        self._git(["clean", "-fdx", "--", *generated])

    def _is_clean_excluded(self, path: str) -> bool:
        normalized = path.rstrip("/") + ("/" if path.endswith("/") else "")
        for exclude in self._git_clean_excludes():
            pattern = exclude.rstrip("/") + "/"
            if normalized == pattern or normalized.startswith(pattern):
                return True
        return False

    def _is_baseline_local_path(self, path: str) -> bool:
        normalized = path.rstrip("/")
        for baseline_path in self._baseline_local_paths:
            baseline = baseline_path.rstrip("/")
            if normalized == baseline or normalized.startswith(f"{baseline}/"):
                return True
        return False

    def _git_clean_excludes(self) -> list[str]:
        """Exclude VCR audit files from ignored-file cleanup when they live in the workspace."""
        excludes = [".vcr/"]

        recorder_dir = self.recorder.output_dir
        if not recorder_dir.is_absolute():
            recorder_dir = recorder_dir.resolve()

        try:
            relative = recorder_dir.relative_to(self.workspace_dir)
        except ValueError:
            return excludes

        pattern = relative.as_posix().rstrip("/") + "/"
        if pattern not in excludes:
            excludes.append(pattern)
        return excludes
