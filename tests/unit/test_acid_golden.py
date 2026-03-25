"""Production tests for the ACID Workspace and Golden Run Cache."""

import os
import shutil
import tempfile

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

    def test_begin_creates_git_repo(self):
        acid = ACIDWorkspace(self.workspace)
        acid.begin(session_id="test-001")

        assert (os.path.join(self.workspace, ".git"))
        assert acid.session is not None
        assert acid.session.session_id == "test-001"

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
            return {"re_executed": True}

        replay_recorder = VCRRecorder(output_dir=self.vcr_dir, auto_save=False)
        outputs, ledger = cache.replay(
            "build a todo app",
            step_executor=mock_executor,
            changed_steps={1},  # Only re-run the coder step
            recorder=replay_recorder,
        )

        assert len(outputs) == 3
        assert ledger.steps_replayed == 2  # planner + tester from cache
        assert ledger.steps_rerun == 1     # coder re-executed
        assert outputs[1]["re_executed"] is True

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
