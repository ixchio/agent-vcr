"""Phase 1 tests: replay bundle, ghost modes, ACID abort/ctx, bundle CLI."""

from __future__ import annotations

import json
import subprocess

import pytest

import agent_vcr
from agent_vcr.bundle import (
    export_bundle,
    import_bundle,
    reproduce,
    verify_bundle,
)
from agent_vcr.golden_cache import GhostReplayMode, GoldenRunCache
from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder


def _recorded_session(tmp_path, session_id="bundle-src", n=5):
    rec = VCRRecorder(output_dir=str(tmp_path / "vcr"), auto_save=False)
    rec.start_session(session_id)
    for i in range(n):
        rec.record_step(
            f"step_{i}",
            {"i": i},
            {"result": i * 2},
            metadata=FrameMetadata(model="gpt-4o", tokens_used=100, cost_usd=0.001),
        )
    rec.record_tool_call("search", {"q": "x"}, {"hits": 1}, latency_ms=5.0)
    rec.record_checkpoint({"result": 8}, label="mid")
    return rec.save()


class TestReplayBundle:
    def test_export_verify_import_roundtrip(self, tmp_path):
        src = _recorded_session(tmp_path)
        bundle = export_bundle(src)
        assert bundle.suffix == ".vcrb"

        report = verify_bundle(bundle)
        assert report["ok"] is True
        assert report["frame_count"] == 7
        assert report["schema_version"] == "1.0.0"
        assert "gpt-4o" in report["models"]
        assert "search" in report["tools"]

        out = import_bundle(bundle, dest_dir=tmp_path / "in")
        assert out.exists()

        from agent_vcr.player import VCRPlayer

        player = VCRPlayer.load(out)
        assert len(player.frames) == 7
        assert player.goto_checkpoint("mid") == {"result": 8}

    def test_reproduce_report(self, tmp_path):
        src = _recorded_session(tmp_path)
        bundle = export_bundle(src)
        report = reproduce(bundle)
        assert report["reproduced"] is True
        assert report["frame_count"] == 7
        assert report["total_tokens"] == 500
        assert report["checkpoints"] == ["mid"]

    def test_tamper_detected(self, tmp_path):
        import tarfile

        from agent_vcr.bundle import EXECUTION_NAME, MANIFEST_NAME

        src = _recorded_session(tmp_path)
        bundle = export_bundle(src)

        # Surgical tamper: modify execution payload, keep original manifest
        work = tmp_path / "surgery"
        work.mkdir()
        with tarfile.open(bundle, "r:gz") as tar:
            try:
                tar.extractall(path=work, filter="data")
            except TypeError:  # Python < 3.12 has no filter argument
                tar.extractall(path=work)
        exec_path = work / EXECUTION_NAME
        exec_path.write_bytes(exec_path.read_bytes() + b'{"type":"frame","data":{}}' + b"\n")
        tampered = tmp_path / "tampered.vcrb"
        with tarfile.open(tampered, "w:gz") as tar:
            tar.add(str(exec_path), arcname=EXECUTION_NAME)
            tar.add(str(work / MANIFEST_NAME), arcname=MANIFEST_NAME)

        report = verify_bundle(tampered)
        assert report["ok"] is False
        assert any("sha256" in issue for issue in report["issues"])
        with pytest.raises(ValueError, match="verification failed"):
            import_bundle(tampered, dest_dir=tmp_path / "tampered-out")

    def test_corrupt_bytes_detected(self, tmp_path):
        src = _recorded_session(tmp_path)
        bundle = export_bundle(src)
        with open(bundle, "r+b") as fh:
            fh.seek(200)
            byte = fh.read(1)
            fh.seek(200)
            fh.write(b"X" if byte != b"X" else b"Y")
        report = verify_bundle(bundle)
        assert report["ok"] is False  # unreadable gzip -> clean failure, never a raise

    def test_refuses_corrupt_session(self, tmp_path):
        bad = tmp_path / "bad.vcr"
        bad.write_text("not json\n")
        with pytest.raises(ValueError):
            export_bundle(bad)

    def test_public_api_exports(self):
        assert agent_vcr.__version__ == "1.5.1"
        for name in ("export_bundle", "import_bundle", "verify_bundle", "reproduce",
                     "GhostReplayMode", "BundleManifest"):
            assert name in agent_vcr.__all__


class TestGhostReplayModes:
    def _golden(self, tmp_path, task="ghost task"):
        rec = VCRRecorder(output_dir=str(tmp_path / "vcr"), auto_save=False)
        rec.start_session("golden-src")
        for i in range(4):
            rec.record_step(
                f"step_{i}", {"i": i}, {"r": i},
                metadata=FrameMetadata(tokens_used=50, cost_usd=0.001, latency_ms=10.0),
            )
        rec.save()
        cache = GoldenRunCache(cache_dir=str(tmp_path / "golden"))
        cache.save_golden_run(task, rec)
        return cache

    def test_full_mode_zero_cost(self, tmp_path):
        cache = self._golden(tmp_path)
        outputs, ledger = cache.replay_with_mode(
            "ghost task", GhostReplayMode.FULL,
            recorder=VCRRecorder(output_dir=str(tmp_path / "r1"), auto_save=False),
        )
        assert len(outputs) == 4
        assert ledger.steps_replayed == 4 and ledger.steps_rerun == 0
        assert ledger.replay_tokens == 0

    def test_hybrid_reruns_downstream(self, tmp_path):
        cache = self._golden(tmp_path)
        calls = []

        def executor(node, state):
            calls.append(node)
            return {"r": "fresh"}

        outputs, ledger = cache.replay_with_mode(
            "ghost task", "hybrid", step_executor=executor, changed_steps={1},
            recorder=VCRRecorder(output_dir=str(tmp_path / "r2"), auto_save=False),
        )
        assert ledger.steps_rerun == 3  # frame 1 + downstream 2,3
        assert ledger.steps_replayed == 1
        assert outputs[1] == {"r": "fresh"}

    def test_live_reruns_only_listed(self, tmp_path):
        cache = self._golden(tmp_path)
        outputs, ledger = cache.replay_with_mode(
            "ghost task", GhostReplayMode.LIVE,
            step_executor=lambda n, s: {"r": "live"},
            changed_steps={1},
            recorder=VCRRecorder(output_dir=str(tmp_path / "r3"), auto_save=False),
        )
        assert ledger.steps_rerun == 1
        assert ledger.steps_replayed == 3

    def test_branch_forks_at_frame(self, tmp_path):
        cache = self._golden(tmp_path)
        outputs, ledger = cache.replay_with_mode(
            "ghost task", GhostReplayMode.BRANCH,
            step_executor=lambda n, s: {"r": "branched"},
            from_frame=2,
            recorder=VCRRecorder(output_dir=str(tmp_path / "r4"), auto_save=False),
        )
        assert outputs[:2] == [{"r": 0}, {"r": 1}]
        assert outputs[2] == {"r": "branched"}
        assert ledger.steps_replayed == 2

    def test_mode_validation(self, tmp_path):
        cache = self._golden(tmp_path)
        with pytest.raises(ValueError):
            cache.replay_with_mode("ghost task", "hybrid")  # no executor
        with pytest.raises(ValueError):
            cache.replay_with_mode(
                "ghost task", "branch",
                step_executor=lambda n, s: {},
            )  # no from_frame


class TestACIDHardening:
    def _repo(self, tmp_path):
        work = tmp_path / "work"
        work.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=work, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=work, check=True)
        (work / "base.txt").write_text("base")
        subprocess.run(["git", "add", "-A"], cwd=work, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=work, check=True)
        return work

    def test_savepoints_status_and_abort(self, tmp_path):
        from agent_vcr.integrations.openhands import ACIDWorkspace

        work = self._repo(tmp_path)
        ws = ACIDWorkspace(str(work), recorder=VCRRecorder(
            output_dir=str(tmp_path / "vcr"), auto_save=False))
        ws.begin("aborttest")
        (work / "agent.txt").write_text("v1")
        ws.savepoint({"v": 1}, node_name="write")
        assert len(ws.savepoints()) == 1
        assert ws.savepoints()[0]["acid_commit_hash"]
        assert ws.status()["savepoints"] == 1

        ws.abort()
        assert not (work / "agent.txt").exists()
        assert (work / "base.txt").read_text() == "base"

    def test_context_manager_commits_and_aborts(self, tmp_path):
        from agent_vcr.integrations.openhands import ACIDWorkspace

        work = self._repo(tmp_path)
        with ACIDWorkspace(str(work), recorder=VCRRecorder(
                output_dir=str(tmp_path / "vcr1"), auto_save=False)) as ws:
            (work / "keep.txt").write_text("keep")
            ws.savepoint({"v": 1})
        assert (work / "keep.txt").read_text() == "keep"

        with pytest.raises(RuntimeError, match="boom"), ACIDWorkspace(
            str(work),
            recorder=VCRRecorder(output_dir=str(tmp_path / "vcr2"), auto_save=False),
        ) as ws:
            (work / "drop.txt").write_text("drop")
            ws.savepoint({"v": 2})
            raise RuntimeError("boom")
        assert not (work / "drop.txt").exists()


class TestBundleCLI:
    def test_bundle_export_verify_reproduce(self, tmp_path, capsys):
        from agent_vcr.cli import main

        src = _recorded_session(tmp_path)
        bundle = tmp_path / "cli.vcrb"
        main(["bundle", "export", str(src), "-o", str(bundle)])
        assert bundle.exists()
        capsys.readouterr()  # drain export confirmation line

        main(["bundle", "verify", str(bundle)])
        out = capsys.readouterr().out
        assert json.loads(out)["ok"] is True

        main(["reproduce", str(bundle)])
        report = json.loads(capsys.readouterr().out)
        assert report["reproduced"] is True

        main(["bundle", "import", str(bundle), "-d", str(tmp_path / "cli-in")])
        assert (tmp_path / "cli-in" / f"{report['session_id']}.vcr").exists()
