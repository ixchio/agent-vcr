import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from agent_vcr.models import Frame, Session
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
    def __init__(self, workspace_dir: str, recorder: Optional[VCRRecorder] = None) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.recorder = recorder or VCRRecorder(auto_save=True)
        self.session: Optional[Session] = None
        self.branch_name: Optional[str] = None

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
        BEGIN - Starts a session and git stashes the workspace state as a snapshot.
        """
        self.session = self.recorder.start_session(session_id)
        if self.session is None:
            raise RuntimeError("Session failed to start")

        # Validate session ID before using it in git commands
        self._sanitize_session_id(self.session.session_id)

        # Git initialize workspace if not already
        if not (self.workspace_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "config", "user.name", "Agent VCR"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "config", "user.email", "vcr@example.com"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "branch", "-m", "main"], cwd=self.workspace_dir, check=True)

            # Initial baseline commit
            init_file = self.workspace_dir / ".acid_init"
            if not init_file.exists():
                init_file.touch()
            subprocess.run(["git", "add", "-A"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial ACID baseline"], cwd=self.workspace_dir, check=True)

        # ISOLATION: Create an isolated branch for this agent's uncommitted file changes.
        # The session ID is whitelist-validated by _sanitize_session_id() above,
        # so git argument injection (e.g. --orphan) is not possible here.
        self.branch_name = f"acid/{self.session.session_id}"
        subprocess.run(["git", "checkout", "-b", self.branch_name], cwd=self.workspace_dir, check=True)

        logger.info(f"ACID BEGIN: Session {self.session.session_id} started on branch {self.branch_name}")
        return self.session

    def savepoint(self, step_data: dict, node_name: str = "openhands_action") -> Frame:
        """
        SAVEPOINT - every frame is a savepoint, filesystem + memory state together
        """
        # 1. Record state map via agent-vcr
        frame = self.recorder.record_step(
            node_name=node_name,
            input_state={},
            output_state=step_data
        )

        # 2. Synchronize the filesystem state via git
        subprocess.run(["git", "add", "-A"], cwd=self.workspace_dir, check=True)
        commit_msg = f"SAVEPOINT: {frame.frame_id}"

        subprocess.run(["git", "commit", "--allow-empty", "-m", commit_msg], cwd=self.workspace_dir, check=True)
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
        if to_frame_index >= len(frames):
            raise ValueError(f"Frame index {to_frame_index} out of bounds")

        target_frame = frames[to_frame_index]
        self.recorder = self.recorder.fork(from_frame=to_frame_index)
        self.session = self.recorder.get_session()

        # 2. Rewind the git workspace
        commit_msg = f"SAVEPOINT: {target_frame.frame_id}"
        log_out = subprocess.run(
            ["git", "log", f"--grep={commit_msg}", "--format=%H"],
            cwd=self.workspace_dir,
            capture_output=True,
            text=True,
            check=True
        )

        target_hash = log_out.stdout.strip().splitlines()[0] if log_out.stdout.strip() else None

        if not target_hash:
            # Fallback for savepoints where FS didn't mutate, rewind to the closest previous hash
            logger.warning(f"No direct file mutation at frame {target_frame.frame_id}. Rolling back to latest FS state.")
            subprocess.run(["git", "reset", "--hard"], cwd=self.workspace_dir, check=True)
        else:
            subprocess.run(["git", "reset", "--hard", target_hash], cwd=self.workspace_dir, check=True)

        clean_cmd = ["git", "clean", "-fdx"]
        for exclude in self._git_clean_excludes():
            clean_cmd.extend(["-e", exclude])
        subprocess.run(clean_cmd, cwd=self.workspace_dir, check=True)

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

        # Merge the branch back to main.
        # Branch names are safe — validated by _sanitize_session_id().
        subprocess.run(["git", "checkout", "main"], cwd=self.workspace_dir, check=True)
        subprocess.run(
            ["git", "merge", "--no-ff", "-m", f"ACID COMMIT: {self.session.session_id}", self.branch_name],
            cwd=self.workspace_dir,
            check=True,
        )
        logger.info(f"ACID COMMIT: Session {self.session.session_id} successfully merged to main.")

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
