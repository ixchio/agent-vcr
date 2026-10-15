"""Production tests for the ACID Workspace and Golden Run Cache."""

import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent_vcr.golden_cache import CostLedger, GoldenRunCache
from agent_vcr.integrations.openhands import ACIDWorkspace
from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder

# ──────────────────────────────────────────────
#  ACIDWorkspace Tests
# ──────────────────────────────────────────────


class TestACIDWorkspace:
    """Tests for transactional agent execution."""

    def setup_method(self):
        self.workspace = tempfile.mkdtemp(prefix="acid_test_")

    def teardown_method(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _init_repo(self) -> Path:
        workspace = Path(self.workspace)
        subprocess.run(["git", "init"], cwd=workspace, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Agent VCR"], cwd=workspace, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "vcr@example.com"], cwd=workspace, check=True
        )
        (workspace / "tracked.py").write_text("VALUE = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=workspace, check=True)
        return workspace

    def test_begin_creates_git_repo(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-001")

        assert os.path.exists(os.path.join(self.workspace, ".git"))
        assert acid.session is not None
        assert acid.session.session_id == "test-001"

    def test_new_workspace_does_not_write_preexisting_secret_to_git_objects(self):
        workspace = Path(self.workspace)
        env_file = workspace / ".env"
        env_file.write_text("VCR_ESCROW_SENTINEL=baseline-only\n")

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="new-workspace-local-file")
        acid.savepoint({"step": "baseline"})

        tracked = subprocess.run(
            ["git", "ls-files", "--", ".env"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        git_objects = subprocess.run(
            ["git", "cat-file", "--batch-all-objects", "--batch"],
            cwd=workspace,
            capture_output=True,
            check=True,
        ).stdout
        assert tracked == ""
        assert b"VCR_ESCROW_SENTINEL=baseline-only" not in git_objects

    def test_begin_creates_isolation_branch(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-002")

        assert acid.branch_name == "acid/test-002"

    def test_savepoint_creates_git_commit(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-003")

        frame = acid.savepoint({"action": "wrote file"}, node_name="coder")

        assert frame is not None
        assert frame.node_name == "coder"
        assert "acid_commit_hash" in frame.metadata.custom

    def test_savepoint_commit_hash_is_persisted_when_recorder_flushes_immediately(self):
        recorder = VCRRecorder(
            output_dir=os.path.join(self.workspace, ".vcr"),
            buffer_size=1,
            auto_save=True,
        )
        acid = ACIDWorkspace(self.workspace, recorder=recorder)
        acid.begin(session_id="flush-hash")

        acid.savepoint({"step": "durable hash"})
        recorder.save()

        vcr_file = os.path.join(self.workspace, ".vcr", "flush-hash.vcr")
        with open(vcr_file) as f:
            contents = f.read()
        assert "acid_commit_hash" in contents

    def test_rollback_reverts_filesystem(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-004")

        # Savepoint 0: clean state
        acid.savepoint({"step": "init"})

        # Create a file that shouldn't exist after rollback
        bad_file = os.path.join(self.workspace, "bad_code.py")
        with open(bad_file, "w") as f:
            f.write("# This should be rolled back\n" * 100)

        # Savepoint 1: bad file exists
        acid.savepoint({"step": "wrote bad code"})
        assert os.path.exists(bad_file)

        # Rollback to savepoint 0
        acid.rollback(to_frame_index=0)

        # The file must be physically gone
        assert not os.path.exists(bad_file), "Rollback failed to remove file from disk"

    def test_rollback_preserves_earlier_files(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-005")

        # Create a good file
        good_file = os.path.join(self.workspace, "good.py")
        with open(good_file, "w") as f:
            f.write("# Good code\n")

        acid.savepoint({"step": "good file"})

        # Create a bad file
        bad_file = os.path.join(self.workspace, "bad.py")
        with open(bad_file, "w") as f:
            f.write("# Bad code\n")

        acid.savepoint({"step": "bad file"})

        # Rollback to savepoint 0 (good file)
        acid.rollback(to_frame_index=0)

        assert os.path.exists(good_file), "Good file should survive rollback"
        assert not os.path.exists(bad_file), "Bad file should be removed"

    def test_rollback_deletes_ignored_generated_files(self):
        recorder = VCRRecorder(output_dir=os.path.join(self.workspace, ".vcr"), auto_save=False)
        acid = ACIDWorkspace(self.workspace, recorder=recorder)
        acid.begin(session_id="ignored-files")

        gitignore = os.path.join(self.workspace, ".gitignore")
        with open(gitignore, "w") as f:
            f.write("ignored.txt\n")
        acid.savepoint({"step": "ignore rule"})

        ignored_file = os.path.join(self.workspace, "ignored.txt")
        with open(ignored_file, "w") as f:
            f.write("agent generated ignored junk\n")
        acid.savepoint({"step": "ignored junk"})
        assert os.path.exists(ignored_file)

        acid.rollback(to_frame_index=0)

        assert not os.path.exists(ignored_file), "Ignored generated files must be removed"
        assert os.path.isdir(os.path.join(self.workspace, ".vcr")), "VCR audit directory must survive"

    def test_rollback_preserves_preexisting_ignored_files(self):
        gitignore = os.path.join(self.workspace, ".gitignore")
        with open(gitignore, "w") as f:
            f.write(".env\nignored-generated.txt\n")

        env_file = os.path.join(self.workspace, ".env")
        with open(env_file, "w") as f:
            f.write("TOKEN=keep-me\n")

        recorder = VCRRecorder(output_dir=os.path.join(self.workspace, ".vcr"), auto_save=False)
        acid = ACIDWorkspace(self.workspace, recorder=recorder)
        acid.begin(session_id="preserve-env")
        acid.savepoint({"step": "baseline"})

        generated = os.path.join(self.workspace, "ignored-generated.txt")
        with open(generated, "w") as f:
            f.write("agent junk\n")
        acid.savepoint({"step": "generated ignored junk"})

        acid.rollback(to_frame_index=0)

        assert os.path.exists(env_file), "Pre-existing ignored files must survive rollback"
        assert not os.path.exists(generated), "Agent-generated ignored files must be removed"

    def test_rollback_preserves_preexisting_untracked_files_without_committing_them(self):
        subprocess.run(["git", "init"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.name", "Agent VCR"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.email", "vcr@example.com"], cwd=self.workspace, check=True)

        tracked = os.path.join(self.workspace, "tracked.py")
        with open(tracked, "w") as f:
            f.write("VALUE = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=self.workspace, check=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=self.workspace, check=True)

        local_notes = os.path.join(self.workspace, "notes.local")
        with open(local_notes, "w") as f:
            f.write("keep this local\n")

        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="preserve-untracked")
        acid.savepoint({"step": "baseline"})

        generated = os.path.join(self.workspace, "generated.local")
        with open(generated, "w") as f:
            f.write("agent junk\n")
        acid.savepoint({"step": "generated untracked junk"})

        acid.rollback(to_frame_index=0)
        acid.commit()

        tracked_local = subprocess.run(
            ["git", "ls-files", "--", "notes.local"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert os.path.exists(local_notes), "Pre-existing untracked files must survive rollback"
        assert not os.path.exists(generated), "Agent-generated untracked files must be removed"
        assert tracked_local == "", "Pre-existing untracked files must not be committed"

    def test_rollback_restores_modified_ignored_file_without_leaking_secret(self):
        workspace = self._init_repo()
        local_payload = b"VCR_ESCROW_SENTINEL=baseline-only\n\x00binary-tail\n"
        env_file = workspace / ".env"
        env_file.write_bytes(local_payload)

        recorder = VCRRecorder(output_dir=str(workspace / ".vcr"), auto_save=False)
        acid = ACIDWorkspace(str(workspace), recorder=recorder)
        acid.begin(session_id="restore-env")

        assert acid._escrow_dir is not None
        assert stat.S_IMODE(acid._escrow_dir.stat().st_mode) == 0o700
        blobs = list((acid._escrow_dir / "blobs").iterdir())
        assert blobs
        assert all(stat.S_IMODE(blob.stat().st_mode) == 0o600 for blob in blobs)

        acid.savepoint({"step": "baseline"})
        env_file.write_bytes(b"VCR_ESCROW_SENTINEL=agent-overwrite\n")
        acid.savepoint({"step": "mutated local file"})

        acid.rollback(to_frame_index=0)
        assert env_file.read_bytes() == local_payload

        acid.recorder.save()
        git_log = subprocess.run(
            ["git", "log", "-p", "--all"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "VCR_ESCROW_SENTINEL=baseline-only" not in git_log
        assert "VCR_ESCROW_SENTINEL=agent-overwrite" not in git_log
        git_objects = subprocess.run(
            ["git", "cat-file", "--batch-all-objects", "--batch"],
            cwd=workspace,
            capture_output=True,
            check=True,
        ).stdout
        assert b"VCR_ESCROW_SENTINEL=baseline-only" not in git_objects
        assert b"VCR_ESCROW_SENTINEL=agent-overwrite" not in git_objects
        for recording in (workspace / ".vcr").glob("*.vcr"):
            contents = recording.read_bytes()
            assert b"VCR_ESCROW_SENTINEL=baseline-only" not in contents
            assert b"VCR_ESCROW_SENTINEL=agent-overwrite" not in contents

    def test_rollback_restores_deleted_untracked_file_content_and_permissions(self):
        workspace = self._init_repo()
        local_file = workspace / "notes.local"
        original = b"private local notes\n"
        local_file.write_bytes(original)
        local_file.chmod(0o640)

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="restore-deleted")
        acid.savepoint({"step": "baseline"})
        local_file.unlink()
        acid.savepoint({"step": "deleted local file"})

        acid.rollback(to_frame_index=0)

        assert local_file.read_bytes() == original
        assert stat.S_IMODE(local_file.stat().st_mode) == 0o640

    def test_rollback_removes_renamed_untracked_path_and_restores_original(self):
        workspace = self._init_repo()
        original = workspace / "original.local"
        original.write_text("original content\n")

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="restore-rename")
        acid.savepoint({"step": "baseline"})

        moved_directory = workspace / "agent-output"
        moved_directory.mkdir()
        moved = moved_directory / "renamed.local"
        original.rename(moved)
        acid.savepoint({"step": "renamed local file"})

        acid.rollback(to_frame_index=0)

        assert original.read_text() == "original content\n"
        assert not moved.exists()
        assert not moved_directory.exists()

    def test_rollback_restores_nested_ignored_symlink_and_executable(self):
        workspace = self._init_repo()
        subprocess.run(
            ["git", "config", "--local", "core.fileMode", "true"], cwd=workspace, check=True
        )
        (workspace / ".gitignore").write_text("scratch/\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "ignore scratch"], cwd=workspace, check=True)

        nested = workspace / "scratch" / "nested"
        nested.mkdir(parents=True)
        executable = nested / "tool.sh"
        executable.write_bytes(b"#!/bin/sh\necho original\n")
        executable.chmod(0o751)
        link = nested / "current-tool"
        link.symlink_to("tool.sh")

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="restore-nested")
        acid.savepoint({"step": "baseline"})

        executable.write_bytes(b"#!/bin/sh\necho changed\n")
        executable.chmod(0o600)
        link.unlink()
        link.symlink_to("missing-tool.sh")
        acid.savepoint({"step": "mutated nested files"})

        acid.rollback(to_frame_index=0)

        assert executable.read_bytes() == b"#!/bin/sh\necho original\n"
        assert stat.S_IMODE(executable.stat().st_mode) == 0o751
        assert link.is_symlink()
        assert os.readlink(link) == "tool.sh"

    def test_rollback_restores_local_file_and_removes_hallucinated_sibling(self):
        workspace = self._init_repo()
        (workspace / ".gitignore").write_text("local/\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "ignore local files"], cwd=workspace, check=True)

        local_directory = workspace / "local"
        local_directory.mkdir()
        existing = local_directory / "settings.json"
        existing.write_text('{"safe": true}\n')

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="mixed-local-files")
        acid.savepoint({"step": "baseline"})

        existing.write_text('{"safe": false}\n')
        hallucinated = local_directory / "hallucinated.json"
        hallucinated.write_text('{"invented": true}\n')
        acid.savepoint({"step": "bad local edits"})

        acid.rollback(to_frame_index=0)

        assert existing.read_text() == '{"safe": true}\n'
        assert not hallucinated.exists()

    def test_rollback_preserves_oversized_files_and_excluded_directories(self):
        workspace = self._init_repo()
        (workspace / ".gitignore").write_text("*.large\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "ignore large files"], cwd=workspace, check=True)

        preexisting_large = workspace / "preexisting.large"
        preexisting_large.write_bytes(b"a" * 65)

        acid = ACIDWorkspace(str(workspace), max_untracked_file_bytes=64)
        acid.begin(session_id="preserve-skipped")
        acid.savepoint({"step": "baseline"})

        preexisting_large.write_bytes(b"changed but preserved" * 4)
        generated_large = workspace / "generated.large"
        generated_large.write_bytes(b"b" * 65)
        venv_file = workspace / ".venv" / "secret.txt"
        node_file = workspace / "node_modules" / "package" / "index.js"
        venv_file.parent.mkdir(parents=True)
        node_file.parent.mkdir(parents=True)
        venv_file.write_text("keep virtualenv data\n")
        node_file.write_text("keep dependency data\n")
        acid.savepoint({"step": "large and excluded output"})

        acid.rollback(to_frame_index=0)

        assert preexisting_large.exists()
        assert generated_large.read_bytes() == b"b" * 65
        assert venv_file.read_text() == "keep virtualenv data\n"
        assert node_file.read_text() == "keep dependency data\n"

    def test_savepoint_local_snapshots_support_repeated_rollback_and_cleanup(self):
        workspace = self._init_repo()
        (workspace / ".gitignore").write_text(".env\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=workspace, check=True)
        subprocess.run(["git", "commit", "-m", "ignore env"], cwd=workspace, check=True)
        env_file = workspace / ".env"
        env_file.write_text("VALUE=begin\n")

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="repeat-rollback")
        env_file.write_text("VALUE=savepoint-zero\n")
        acid.savepoint({"step": 0})
        escrow_dir = acid._escrow_dir

        env_file.write_text("VALUE=bad-change\n")
        acid.rollback(to_frame_index=0)
        assert env_file.read_text() == "VALUE=savepoint-zero\n"

        env_file.write_text("VALUE=another-bad-change\n")
        acid.rollback(to_frame_index=0)
        assert env_file.read_text() == "VALUE=savepoint-zero\n"

        acid.commit()
        assert escrow_dir is not None
        assert not escrow_dir.exists()

    def test_savepoints_share_immutable_content_addressed_escrow_blobs(self):
        workspace = self._init_repo()
        env_file = workspace / ".env"
        baseline = b"TOKEN=baseline\n"
        env_file.write_bytes(baseline)

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="content-addressed-blobs")
        acid.savepoint({"step": 0})

        assert acid._escrow_blobs_dir is not None
        baseline_hash = hashlib.sha256(baseline).hexdigest()
        baseline_blob = acid._escrow_blobs_dir / baseline_hash
        assert baseline_blob.read_bytes() == baseline

        env_file.write_bytes(b"TOKEN=mutated\n")
        acid.savepoint({"step": 1})
        blob_names = {path.name for path in acid._escrow_blobs_dir.iterdir()}
        assert baseline_hash in blob_names
        assert all(len(name) == 64 for name in blob_names)

        acid.rollback(to_frame_index=0)
        assert env_file.read_bytes() == baseline
        assert baseline_blob.read_bytes() == baseline

        env_file.write_bytes(b"TOKEN=mutated-again\n")
        acid.rollback(to_frame_index=0)
        assert env_file.read_bytes() == baseline
        assert baseline_blob.read_bytes() == baseline

    def test_workspace_paths_reject_traversal_absolute_and_symlink_parent_escape(self):
        workspace = self._init_repo()
        outside = workspace.parent / f"{workspace.name}-outside"
        outside.mkdir()
        (workspace / "escape").symlink_to(outside, target_is_directory=True)

        acid = ACIDWorkspace(str(workspace))
        with pytest.raises(ValueError, match="Unsafe workspace-relative path"):
            acid._workspace_path("../outside-secret")
        with pytest.raises(ValueError, match="Unsafe workspace-relative path"):
            acid._workspace_path(str(outside / "secret"))
        with pytest.raises(ValueError, match="resolves outside"):
            acid._workspace_path("escape/secret")

        # Escrowing the symlink object itself is safe; its target is metadata.
        assert acid._workspace_path("escape") == workspace / "escape"
        shutil.rmtree(outside)

    def test_begin_rejects_dirty_tracked_files_by_default(self):
        subprocess.run(["git", "init"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.name", "Agent VCR"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.email", "vcr@example.com"], cwd=self.workspace, check=True)

        tracked = os.path.join(self.workspace, "tracked.py")
        with open(tracked, "w") as f:
            f.write("VALUE = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=self.workspace, check=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=self.workspace, check=True)

        with open(tracked, "w") as f:
            f.write("VALUE = 2\n")

        acid = ACIDWorkspace(self.workspace)
        with pytest.raises(RuntimeError, match="dirty tracked files"):
            acid.begin(session_id="dirty-tracked")

    def test_commit_returns_to_original_branch(self):
        subprocess.run(["git", "init"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.name", "Agent VCR"], cwd=self.workspace, check=True)
        subprocess.run(["git", "config", "user.email", "vcr@example.com"], cwd=self.workspace, check=True)
        subprocess.run(["git", "branch", "-m", "trunk"], cwd=self.workspace, check=True)

        tracked = os.path.join(self.workspace, "tracked.py")
        with open(tracked, "w") as f:
            f.write("VALUE = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=self.workspace, check=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=self.workspace, check=True)

        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="original-branch")
        acid.savepoint({"step": "baseline"})
        acid.commit()

        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert branch == "trunk"

    def test_commit_merges_to_main(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-006")

        result_file = os.path.join(self.workspace, "result.py")
        with open(result_file, "w") as f:
            f.write("# Final result\n")

        acid.savepoint({"step": "final"})
        acid.commit()

        # After commit, we should be on main and the file should exist
        assert os.path.exists(result_file)

    def test_commit_finalizes_changes_made_after_last_savepoint(self):
        workspace = self._init_repo()
        tracked = workspace / "tracked.py"

        acid = ACIDWorkspace(str(workspace))
        acid.begin(session_id="finalize-after-savepoint")
        acid.savepoint({"step": "last savepoint"})

        tracked.write_text("VALUE = 2\n")
        generated = workspace / "generated.py"
        generated.write_text("FINAL = True\n")
        acid.commit()

        committed = subprocess.run(
            ["git", "show", "HEAD:tracked.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        committed_generated = subprocess.run(
            ["git", "show", "HEAD:generated.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert tracked.read_text() == "VALUE = 2\n"
        assert committed == "VALUE = 2\n"
        assert generated.read_text() == "FINAL = True\n"
        assert committed_generated == "FINAL = True\n"
        assert status == ""

    def test_full_acid_lifecycle(self):
        """The complete BEGIN → SAVEPOINT → ROLLBACK → RESUME → COMMIT cycle."""
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="lifecycle-test")

        # Phase 1: initial work
        acid.savepoint({"phase": "init"})

        # Phase 2: bad work
        bad = os.path.join(self.workspace, "monolith.py")
        with open(bad, "w") as f:
            f.write("class GodObject:\n" + "    pass\n" * 200)
        acid.savepoint({"phase": "bad"})

        # Rollback
        acid.rollback(to_frame_index=0)
        assert not os.path.exists(bad)

        # Phase 3: good work
        good = os.path.join(self.workspace, "clean.py")
        with open(good, "w") as f:
            f.write("class CleanService:\n    pass\n")
        acid.savepoint({"phase": "good"})

        # Commit
        acid.commit()
        assert os.path.exists(good)
        assert not os.path.exists(bad)


# ──────────────────────────────────────────────
#  GoldenRunCache Tests
# ──────────────────────────────────────────────


class TestGoldenRunCache:
    """Tests for the golden run cache — never pay twice."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="golden_test_")
        self.vcr_dir = os.path.join(self.tmpdir, "vcr")
        self.golden_dir = os.path.join(self.tmpdir, "golden")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_recorder_with_frames(self, task_name="test task"):
        """Helper to create a recorder with realistic frames."""
        recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        recorder.start_session(session_id="golden-test")

        recorder.record_step(
            "planner", {"task": task_name}, {"plan": "do stuff"},
            metadata=FrameMetadata(tokens_used=1000, cost_usd=0.003, latency_ms=500),
        )
        recorder.record_step(
            "coder", {"plan": "do stuff"}, {"code": "written"},
            metadata=FrameMetadata(tokens_used=2000, cost_usd=0.006, latency_ms=900),
        )
        recorder.record_step(
            "tester", {"code": "written"}, {"result": "pass"},
            metadata=FrameMetadata(tokens_used=500, cost_usd=0.0015, latency_ms=300),
        )

        recorder.save()
        return recorder

    def test_save_golden_run(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)

        fp = cache.save_golden_run("build a todo app", recorder)

        assert fp is not None
        assert len(fp) == 16  # sha256 hex truncated to 16

    def test_has_golden_run(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)

        cache.save_golden_run("build a todo app", recorder)

        assert cache.has_golden_run("build a todo app") is True
        assert cache.has_golden_run("build a blog") is False

    def test_replay_returns_all_outputs(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        outputs, ledger = cache.replay("build a todo app", recorder=replay_recorder)

        assert len(outputs) == 3
        assert outputs[0]["plan"] == "do stuff"
        assert outputs[1]["code"] == "written"
        assert outputs[2]["result"] == "pass"

    def test_replay_zero_cost(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        _, ledger = cache.replay("build a todo app", recorder=replay_recorder)

        assert ledger.replay_tokens == 0
        assert ledger.replay_cost_usd == 0.0
        assert ledger.steps_replayed == 3
        assert ledger.steps_rerun == 0

    def test_replay_with_changed_steps(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        def mock_executor(node_name, input_state):
            return {"re_executed": True, "node": node_name, "input": input_state}

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        outputs, ledger = cache.replay(
            "build a todo app",
            step_executor=mock_executor,
            changed_steps={1},  # Only re-run the coder step
            recorder=replay_recorder,
        )

        assert len(outputs) == 3
        assert ledger.steps_replayed == 1  # planner from cache
        assert ledger.steps_rerun == 2     # coder + downstream tester re-executed
        assert outputs[1]["re_executed"] is True
        assert outputs[2]["node"] == "tester"
        assert ledger.step_sources == ["golden_cache", "changed_step", "downstream_invalidated"]

    def test_replay_partial_mode_keeps_old_downstream_behavior(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        def mock_executor(node_name, input_state):
            return {"re_executed": True}

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        _, ledger = cache.replay(
            "build a todo app",
            step_executor=mock_executor,
            changed_steps={1},
            recorder=replay_recorder,
            allow_partial_replay=True,
        )

        assert ledger.steps_replayed == 2
        assert ledger.steps_rerun == 1

    def test_changed_steps_require_executor(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        with pytest.raises(ValueError, match="changed_steps requires step_executor"):
            cache.replay("build a todo app", changed_steps={1})

    def test_identity_participates_in_fingerprint(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)

        identity = GoldenRunCache.build_identity(
            model="gpt-4o",
            prompt_hash="prompt-a",
            code_commit="abc123",
            tool_schema_hash="tools-v1",
        )
        cache.save_golden_run("build a todo app", recorder, identity=identity)

        assert cache.has_golden_run("build a todo app", identity=identity) is True
        assert cache.has_golden_run(
            "build a todo app",
            identity={**identity, "prompt_hash": "prompt-b"},
        ) is False

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        _, ledger = cache.replay(
            "build a todo app",
            recorder=replay_recorder,
            identity=identity,
        )
        assert ledger.cache_hit_reason is not None
        assert "task_and_identity_match" in ledger.cache_hit_reason

    def test_nested_identity_key_order_is_canonical(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        first = {
            "tools": [
                {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    },
                }
            ],
            "prompt": {"system": "be precise", "temperature": 0},
        }
        reordered = {
            "prompt": {"temperature": 0, "system": "be precise"},
            "tools": [
                {
                    "parameters": {
                        "properties": {"limit": {"type": "integer"}, "query": {"type": "string"}},
                        "type": "object",
                    },
                    "name": "search",
                }
            ],
        }

        cache.save_golden_run("canonical identity", recorder, identity=first)
        assert cache.has_golden_run("canonical identity", identity=reordered)

    def test_cost_savings_calculation(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        _, ledger = cache.replay("build a todo app", recorder=replay_recorder)

        assert ledger.original_tokens == 3500
        assert ledger.original_cost_usd == 0.0105
        assert ledger.tokens_saved == 3500
        assert ledger.savings_percent == 100.0

    def test_invalidate_golden_run(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder)

        assert cache.has_golden_run("build a todo app") is True

        result = cache.invalidate("build a todo app")

        assert result is True
        assert cache.has_golden_run("build a todo app") is False

    def test_list_golden_runs(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder, tags=["web"])

        runs = cache.list_golden_runs()

        assert len(runs) == 1
        assert runs[0]["task"] == "build a todo app"
        assert "web" in runs[0]["tags"]

    def test_list_runs_alias(self):
        recorder = self._make_recorder_with_frames()
        cache = GoldenRunCache(cache_dir=self.golden_dir)
        cache.save_golden_run("build a todo app", recorder, tags=["web"])

        assert cache.list_runs() == cache.list_golden_runs()

    def test_fingerprint_is_case_insensitive(self):
        cache = GoldenRunCache(cache_dir=self.golden_dir)

        fp1 = cache._fingerprint("Build a Todo App")
        fp2 = cache._fingerprint("build a todo app")

        assert fp1 == fp2

    def test_replay_nonexistent_task_raises(self):
        cache = GoldenRunCache(cache_dir=self.golden_dir)

        with pytest.raises(KeyError):
            cache.replay("does not exist")


# ──────────────────────────────────────────────
#  CostLedger Tests
# ──────────────────────────────────────────────


class TestCostLedger:
    """Tests for cost tracking."""

    def test_empty_ledger(self):
        ledger = CostLedger()
        assert ledger.tokens_saved == 0
        assert ledger.cost_saved_usd == 0.0
        assert ledger.savings_percent == 100.0  # 0/0 defaults to 100%

    def test_full_savings(self):
        ledger = CostLedger()
        ledger.original_tokens = 5000
        ledger.original_cost_usd = 0.015
        ledger.original_latency_ms = 3000
        ledger.replay_tokens = 0
        ledger.replay_cost_usd = 0.0
        ledger.replay_latency_ms = 1

        assert ledger.tokens_saved == 5000
        assert ledger.cost_saved_usd == 0.015
        assert ledger.savings_percent == 100.0

    def test_partial_savings(self):
        ledger = CostLedger()
        ledger.original_tokens = 5000
        ledger.original_cost_usd = 0.015
        ledger.replay_tokens = 2000
        ledger.replay_cost_usd = 0.006

        assert ledger.tokens_saved == 3000
        assert ledger.savings_percent == 60.0

    def test_summary_dict(self):
        ledger = CostLedger()
        ledger.original_tokens = 1000
        ledger.original_cost_usd = 0.003

        summary = ledger.summary()
        assert "original" in summary
        assert "replay" in summary
        assert "saved" in summary
