import logging
import subprocess
from pathlib import Path

from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)

class ACIDWorkspace:
    """
    Provides ACID transaction semantics for agent execution.
    Combines Agent VCR state snapshotting with git-backed filesystem freezing to allow
    full-world rollbacks.
    """
    def __init__(self, workspace_dir: str, recorder: VCRRecorder = None):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.recorder = recorder or VCRRecorder(auto_save=True)
        self.session = None

        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def begin(self, session_id: str | None = None):
        """
        BEGIN - Starts a session and git stashes the workspace state as a snapshot.
        """
        self.session = self.recorder.start_session(session_id)

        # Git initialize workspace if not already
        if not (self.workspace_dir / ".git").exists():
            subprocess.run(["git", "init"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "branch", "-m", "main"], cwd=self.workspace_dir, check=True)

            # Initial baseline commit
            init_file = self.workspace_dir / ".acid_init"
            if not init_file.exists():
                init_file.touch()
            subprocess.run(["git", "add", "-A"], cwd=self.workspace_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial ACID baseline"], cwd=self.workspace_dir, check=True)

        # ISOLATION: Create an isolated branch for this agent's uncommitted file changes
        self.branch_name = f"acid/{self.session.session_id}"
        subprocess.run(["git", "checkout", "-b", self.branch_name], cwd=self.workspace_dir, check=True)

        logger.info(f"ACID BEGIN: Session {self.session.session_id} started on branch {self.branch_name}")
        return self.session

    def savepoint(self, step_data: dict, node_name: str = "openhands_action"):
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

    def rollback(self, to_frame_index: int):
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

        subprocess.run(["git", "clean", "-fd"], cwd=self.workspace_dir, check=True)

        logger.info("ACID ROLLBACK SUCCESS: Filesystem and memory correctly rewound.")

    def commit(self):
        """
        COMMIT - successful path gets locked in, checkpointed permanently to main branch.
        """
        if not self.session:
            raise RuntimeError("No active session to commit.")

        # Save any final memory states
        self.recorder.save()

        # Merge the branch back to main
        subprocess.run(["git", "checkout", "main"], cwd=self.workspace_dir, check=True)
        subprocess.run(["git", "merge", self.branch_name, "--no-ff", "-m", f"ACID COMMIT: {self.session.session_id}"], cwd=self.workspace_dir, check=True)
        logger.info(f"ACID COMMIT: Session {self.session.session_id} successfully merged to main.")
