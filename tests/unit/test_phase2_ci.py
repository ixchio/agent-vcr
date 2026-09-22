"""Phase 2 tests: datasets, regression verdicts, risk, CI gates."""

from __future__ import annotations

import json
import shutil

import pytest

import agent_vcr
from agent_vcr.datasets import Expectation, ReplayDataset
from agent_vcr.golden_cache import GoldenRunCache
from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.regression import (
    CaseVerdict,
    RiskLevel,
    compare_sessions,
    evaluate_candidate,
    run_regression,
)


def _record(tmp_path, session_id, cost=0.01, tokens=100, latency=50.0,
            state=None, tool=None, error=None, checkpoint=None):
    rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    rec.start_session(session_id)
    if error is not None:
        rec.record_error("n", {}, error, latency_ms=latency)
    elif tool is not None:
        rec.record_tool_call(tool, {"q": "x"}, {"hits": 1}, latency_ms=latency)
    else:
        rec.record_step(
            "n", {}, state or {"ok": True},
            metadata=FrameMetadata(model="gpt-4o", cost_usd=cost,
                                   tokens_used=tokens, latency_ms=latency),
        )
    if checkpoint:
        rec.record_checkpoint(state or {"ok": True}, label=checkpoint)
    return rec, rec.save()


def _dataset(tmp_path, name="ci-demo", **expect_kwargs):
    ds_dir = tmp_path / "datasets"
    g_dir = tmp_path / "golden"
    ds = ReplayDataset.create(name, datasets_dir=ds_dir)
    cache = GoldenRunCache(cache_dir=str(g_dir))
    return ds, cache


class TestReplayDataset:
    def test_create_add_select_remove(self, tmp_path):
        ds, cache = _dataset(tmp_path)
        assert ds.version == 1
        rec, _ = _record(tmp_path / "r1", "s1")
        case = ds.add_case("task one", rec, tags=["refunds"],
                           expect=Expectation(max_cost_usd=0.30),
                           golden_cache=cache)
        assert ds.version == 2
        assert case.fingerprint
        assert ds.select(["refunds"]) and not ds.select(["other"])
        assert ds.get_case(case.name).task == "task one"
        assert ReplayDataset.list_datasets(ds.dir) == ["ci-demo"]
        assert ds.remove_case(case.name) is True
        assert ds.remove_case(case.name) is False

    def test_duplicate_dataset_and_case_rejected(self, tmp_path):
        ds, cache = _dataset(tmp_path)
        rec, _ = _record(tmp_path / "r1", "s1")
        ds.add_case("t", rec, golden_cache=cache)
        with pytest.raises(ValueError):
            ReplayDataset.create("ci-demo", datasets_dir=tmp_path / "datasets")
        with pytest.raises(ValueError):
            ds.add_case("t", rec, golden_cache=cache)
        with pytest.raises(FileNotFoundError):
            ReplayDataset.load("nope", datasets_dir=tmp_path / "datasets")

    def test_corrupt_dataset_file(self, tmp_path):
        d = tmp_path / "datasets"
        d.mkdir()
        (d / "bad.json").write_text("{broken")
        with pytest.raises(ValueError, match="Corrupt dataset"):
            ReplayDataset.load("bad", datasets_dir=d)


class TestCompareAndVerdicts:
    def _players(self, tmp_path):
        _, b = _record(tmp_path / "b", "base", cost=0.10, tokens=200, latency=100.0)
        _, c = _record(tmp_path / "c", "cand", cost=0.15, tokens=200, latency=100.0)
        return VCRPlayer.load(b), VCRPlayer.load(c)

    def test_compare_deltas(self, tmp_path):
        base, cand = self._players(tmp_path)
        diff = compare_sessions(base, cand)
        assert diff.states_match is True
        assert diff.cost_delta_usd == pytest.approx(0.05)
        assert diff.cost_delta_pct == pytest.approx(0.5)
        assert diff.token_delta == 0

    def test_state_mismatch_and_new_tool(self, tmp_path):
        _, b = _record(tmp_path / "b", "base", state={"ok": True})
        _, c = _record(tmp_path / "c", "cand", state={"ok": False}, tool="browser")
        diff = compare_sessions(VCRPlayer.load(b), VCRPlayer.load(c))
        assert diff.states_match is False
        assert diff.added_tools == ["browser"]
        assert diff.removed_models == ["gpt-4o"]

    def test_expectation_gates(self, tmp_path):
        from agent_vcr.datasets import TestCase

        _, b = _record(tmp_path / "b", "base", cost=0.10)
        base = VCRPlayer.load(b)
        case = TestCase(name="c", task="t", fingerprint="x",
                        expect=Expectation(max_cost_usd=0.30,
                                           expect_state={"ok": True},
                                           forbid_tools=["payments"],
                                           require_checkpoints=["ship"]))

        _, ok_path = _record(tmp_path / "ok", "ok", cost=0.10)
        res = evaluate_candidate(case, VCRPlayer.load(ok_path), base)
        assert res.verdict == CaseVerdict.FAIL  # missing checkpoint
        assert any("checkpoint" in r for r in res.reasons)

        _, bad_path = _record(tmp_path / "bad", "bad", cost=0.99)
        res = evaluate_candidate(case, VCRPlayer.load(bad_path), base)
        assert res.verdict == CaseVerdict.FAIL
        assert any("budget" in r for r in res.reasons)

        _, err_path = _record(tmp_path / "err", "err", error=ValueError("boom"))
        res = evaluate_candidate(case, VCRPlayer.load(err_path), base)
        assert res.verdict == CaseVerdict.FAIL

    def test_degraded_on_cost_spike(self, tmp_path):
        from agent_vcr.datasets import TestCase

        _, b = _record(tmp_path / "b", "base", cost=0.10)
        _, c = _record(tmp_path / "c", "cand", cost=0.20)
        case = TestCase(name="c", task="t", fingerprint="x")
        res = evaluate_candidate(case, VCRPlayer.load(c), VCRPlayer.load(b))
        assert res.verdict == CaseVerdict.DEGRADED
        assert any("cost" in r for r in res.reasons)


class TestRunRegression:
    def _setup(self, tmp_path):
        ds, cache = _dataset(tmp_path)
        rec1, _ = _record(tmp_path / "g1", "g1", cost=0.10)
        ds.add_case("pass-task", rec1, golden_cache=cache,
                    expect=Expectation(expect_state={"ok": True}))
        rec2, _ = _record(tmp_path / "g2", "g2", cost=0.10)
        ds.add_case("fail-task", rec2, golden_cache=cache)
        return ds, cache

    def test_pass_fail_missing_and_risk(self, tmp_path):
        ds, cache = self._setup(tmp_path)
        _, ok_path = _record(tmp_path / "ok", "ok", cost=0.10)
        _, bad_path = _record(tmp_path / "bad", "bad", error=RuntimeError("x"))
        report = run_regression(
            ds,
            {"pass-task": str(ok_path), "fail-task": str(bad_path)},
            cache,
        )
        assert report.passed == 1 and report.failed == 1 and report.missing == 0
        assert report.risk == RiskLevel.HIGH
        md = report.to_markdown()
        assert "1 passed" in md and "HIGH" in md and "pass-task" in md

    def test_missing_candidate(self, tmp_path):
        ds, cache = self._setup(tmp_path)
        report = run_regression(ds, {}, cache)
        assert report.missing == 2 and report.missing_cases
        assert report.risk == RiskLevel.MEDIUM
        assert "missing" in report.to_markdown()

    def test_tag_filter(self, tmp_path):
        ds, cache = _dataset(tmp_path)
        rec, _ = _record(tmp_path / "g", "g", cost=0.10)
        ds.add_case("tagged", rec, tags=["smoke"], golden_cache=cache)
        ds.add_case("untagged", rec, golden_cache=cache)
        _, ok_path = _record(tmp_path / "ok", "ok", cost=0.10)
        report = run_regression(ds, {"tagged": str(ok_path)}, cache, tags=["smoke"])
        assert report.passed == 1 and report.total == 1

    def test_report_save_roundtrip(self, tmp_path):
        ds, cache = self._setup(tmp_path)
        _, ok_path = _record(tmp_path / "ok", "ok", cost=0.10)
        report = run_regression(ds, {"pass-task": str(ok_path)}, cache)
        out = report.save(tmp_path / "report.json")
        assert json.loads(out.read_text())["dataset"] == "ci-demo"

    def test_public_api(self):
        assert agent_vcr.__version__ == "1.5.0"
        for name in ("ReplayDataset", "TestCase", "Expectation", "run_regression",
                     "compare_sessions", "evaluate_candidate", "RegressionReport",
                     "CaseVerdict", "RiskLevel"):
            assert name in agent_vcr.__all__


class TestDatasetAndCICLI:
    def _run_cli(self, tmp_path, monkeypatch, *argv):
        from agent_vcr.cli import main

        monkeypatch.chdir(tmp_path)
        main(list(argv))

    def test_dataset_crud_cli(self, tmp_path, monkeypatch, capsys):
        from agent_vcr.cli import main

        monkeypatch.chdir(tmp_path)
        _, src = _record(tmp_path / "src", "s", cost=0.05)
        self._run_cli(tmp_path, monkeypatch, "dataset", "create", "shop")
        self._run_cli(tmp_path, monkeypatch, "dataset", "add", "shop",
                       "--task", "refund", "--vcr", str(src),
                       "--max-cost", "0.30", "--expect-state", '{"ok": true}')
        capsys.readouterr()
        main(["dataset", "show", "shop"])
        shown = json.loads(capsys.readouterr().out)
        assert shown["version"] == 2
        assert shown["cases"][0]["expect"]["max_cost_usd"] == 0.30
        main(["dataset", "list"])
        assert "shop" in capsys.readouterr().out
        main(["dataset", "remove", "shop", "refund"])
        assert "removed" in capsys.readouterr().out

    def test_ci_run_gates(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        _, golden = _record(tmp_path / "g", "g", cost=0.10)
        self._run_cli(tmp_path, monkeypatch, "dataset", "create", "shop")
        self._run_cli(tmp_path, monkeypatch, "dataset", "add", "shop",
                       "--task", "refund", "--vcr", str(golden),
                       "--expect-state", '{"ok": true}')
        capsys.readouterr()
        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        _, ok = _record(tmp_path / "ok", "ok", cost=0.10)
        shutil.copy(ok, cand_dir / "refund.vcr")

        # passing gate exits 0 and prints the PR-style report
        self._run_cli(tmp_path, monkeypatch, "ci", "run", "--dataset", "shop",
                       "--candidates", str(cand_dir), "--report", "r.json")
        out = capsys.readouterr().out
        assert "1 passed" in out and "Risk: LOW" in out
        assert json.loads((tmp_path / "r.json").read_text())["passed"] == 1

        # error-frame candidate breaches the default failed gate
        _, bad = _record(tmp_path / "bad", "bad", error=RuntimeError("boom"))
        shutil.copy(bad, cand_dir / "refund.vcr")
        with pytest.raises(SystemExit) as exc:
            self._run_cli(tmp_path, monkeypatch, "ci", "run", "--dataset", "shop",
                           "--candidates", str(cand_dir), "--quiet")
        assert exc.value.code == 1

        # ...but --fail-on never always passes
        self._run_cli(tmp_path, monkeypatch, "ci", "run", "--dataset", "shop",
                       "--candidates", str(cand_dir), "--quiet", "--fail-on", "never")

    def test_ci_github_summary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        summary = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        _, golden = _record(tmp_path / "g", "g", cost=0.10)
        self._run_cli(tmp_path, monkeypatch, "dataset", "create", "shop")
        self._run_cli(tmp_path, monkeypatch, "dataset", "add", "shop",
                       "--task", "refund", "--vcr", str(golden))
        cand_dir = tmp_path / "cand"
        cand_dir.mkdir()
        _, ok = _record(tmp_path / "ok", "ok", cost=0.10)
        shutil.copy(ok, cand_dir / "refund.vcr")
        self._run_cli(tmp_path, monkeypatch, "ci", "run", "--dataset", "shop",
                       "--candidates", str(cand_dir), "--quiet")
        assert "Agent CI" in summary.read_text()
