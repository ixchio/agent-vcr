"""Medium/low fixes: CI regression, datasets, bundle export, TUI and CLI DX."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import cli
from agent_vcr.datasets import Expectation, ReplayDataset, TestCase
from agent_vcr.golden_cache import GoldenRunCache
from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.regression import (
    CaseVerdict,
    RiskLevel,
    case_slug,
    emit_github_annotations,
    evaluate_candidate,
    github_annotation,
    run_regression,
)


def _record(
    out_dir: Path,
    session_id: str,
    *,
    cost: float = 0.01,
    tokens: int = 100,
    latency: float = 50.0,
    tool: str | None = None,
    frames: bool = True,
) -> tuple[VCRRecorder, Path]:
    rec = VCRRecorder(output_dir=str(out_dir), auto_save=False)
    rec.start_session(session_id)
    if tool is not None:
        rec.record_tool_call(tool, {"q": "x"}, {"hits": 1}, latency_ms=latency)
    if frames:
        rec.record_step(
            "n", {}, {"ok": True},
            metadata=FrameMetadata(model="gpt-4o", cost_usd=cost, tokens_used=tokens,
                                   latency_ms=latency),
        )
    return rec, Path(rec.save())


def _strict_loads(text: str) -> Any:
    def _reject(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    return json.loads(text, parse_constant=_reject)


# -- 1. zero baselines --------------------------------------------------------


class TestZeroBaseline:
    def test_zero_to_zero_is_unchanged(self, tmp_path: Path) -> None:
        _, b = _record(tmp_path / "b", "b", cost=0.0, tokens=0, latency=10.0)
        _, c = _record(tmp_path / "c", "c", cost=0.0, tokens=0, latency=10.0)
        case = TestCase(name="c", task="t", fingerprint="x")
        res = evaluate_candidate(case, VCRPlayer.load(c), VCRPlayer.load(b))
        assert res.verdict == CaseVerdict.PASS, res.reasons
        assert res.diff.cost_delta_pct is None
        assert res.diff.token_delta_pct is None

    def test_zero_to_positive_is_finite_regression(self, tmp_path: Path) -> None:
        _, b = _record(tmp_path / "b", "b", cost=0.0, tokens=0, latency=10.0)
        _, c = _record(tmp_path / "c", "c", cost=0.05, tokens=40, latency=10.0)
        case = TestCase(name="c", task="t", fingerprint="x")
        res = evaluate_candidate(case, VCRPlayer.load(c), VCRPlayer.load(b))
        assert res.verdict == CaseVerdict.DEGRADED
        assert res.diff.cost_delta_pct is None
        assert res.diff.cost_delta_usd == pytest.approx(0.05)
        assert "cost +$0.0500 vs zero baseline" in res.reasons
        assert "tokens +40 vs zero baseline" in res.reasons
        assert not any("inf" in r for r in res.reasons)

    def test_zero_baseline_ignored_when_threshold_disabled(self, tmp_path: Path) -> None:
        _, b = _record(tmp_path / "b", "b", cost=0.0, tokens=100, latency=10.0)
        _, c = _record(tmp_path / "c", "c", cost=0.05, tokens=100, latency=10.0)
        case = TestCase(name="c", task="t", fingerprint="x",
                        expect=Expectation(max_cost_delta=None))
        res = evaluate_candidate(case, VCRPlayer.load(c), VCRPlayer.load(b))
        assert res.verdict == CaseVerdict.PASS, res.reasons

    def test_report_json_is_strict(self, tmp_path: Path) -> None:
        ds = ReplayDataset.create("z", datasets_dir=tmp_path / "ds")
        cache = GoldenRunCache(cache_dir=str(tmp_path / "g"))
        golden, _ = _record(tmp_path / "g0", "g0", cost=0.0, tokens=0)
        ds.add_case("t", golden, golden_cache=cache)
        _, cand = _record(tmp_path / "c", "c", cost=0.5, tokens=10)
        report = run_regression(ds, {"t": str(cand)}, cache)
        report.results[0].diff.latency_delta_pct = float("inf")  # belt and braces
        out = report.save(tmp_path / "out" / "report.json")
        data = _strict_loads(out.read_text())
        assert data["degraded"] == 1
        assert data["results"][0]["diff"]["latency_delta_pct"] is None
        assert not list(out.parent.glob("*.tmp"))
        assert "+0.5000" in report.to_markdown()


# -- 2. one bad file never aborts the run ---------------------------------------


def _two_case_dataset(tmp_path: Path) -> tuple[ReplayDataset, GoldenRunCache]:
    ds = ReplayDataset.create("shop", datasets_dir=tmp_path / "ds")
    cache = GoldenRunCache(cache_dir=str(tmp_path / "g"))
    for name in ("good", "bad"):
        rec, _ = _record(tmp_path / f"g-{name}", f"g-{name}")
        ds.add_case(name, rec, golden_cache=cache)
    return ds, cache


class TestPerCaseErrors:
    def test_corrupt_candidate_is_a_failed_case(self, tmp_path: Path) -> None:
        ds, cache = _two_case_dataset(tmp_path)
        _, ok = _record(tmp_path / "ok", "ok")
        corrupt = tmp_path / "corrupt.vcr"
        corrupt.write_text("this is not jsonl\n")
        report = run_regression(ds, {"good": str(ok), "bad": str(corrupt)}, cache)
        verdicts = {r.case: r for r in report.results}
        assert verdicts["good"].verdict == CaseVerdict.PASS
        assert verdicts["bad"].verdict == CaseVerdict.FAIL
        assert "cannot load candidate" in verdicts["bad"].reasons[0]
        assert report.failed == 1 and report.passed == 1
        assert report.risk == RiskLevel.HIGH

    def test_deleted_golden_is_a_failed_case(self, tmp_path: Path) -> None:
        ds, cache = _two_case_dataset(tmp_path)
        info = cache.get_golden_info("bad", None)
        assert info is not None
        Path(info["file"]).unlink()
        _, ok = _record(tmp_path / "ok", "ok")
        report = run_regression(ds, {"good": str(ok), "bad": str(ok)}, cache)
        bad = next(r for r in report.results if r.case == "bad")
        assert bad.verdict == CaseVerdict.FAIL
        assert report.failed == 1 and report.passed == 1

    def test_golden_missing_counts_as_failed(self, tmp_path: Path) -> None:
        ds = ReplayDataset.create("shop", datasets_dir=tmp_path / "ds")
        ds.add_case_ref(TestCase(name="ghost", task="ghost", fingerprint="nope"))
        cache = GoldenRunCache(cache_dir=str(tmp_path / "g"))
        _, ok = _record(tmp_path / "ok", "ok")
        report = run_regression(ds, {"ghost": str(ok)}, cache)
        assert report.failed == 1
        assert report.risk == RiskLevel.HIGH

    def test_cli_always_writes_report(self, tmp_path: Path, monkeypatch: Any,
                                      capsys: Any) -> None:
        monkeypatch.chdir(tmp_path)
        _, golden = _record(tmp_path / "src", "src")
        cli.main(["dataset", "create", "shop"])
        cli.main(["dataset", "add", "shop", "--task", "good", "--vcr", str(golden)])
        cli.main(["dataset", "add", "shop", "--task", "bad", "--vcr", str(golden)])
        cand = tmp_path / "cand"
        cand.mkdir()
        shutil.copy(golden, cand / "good.vcr")
        (cand / "bad.vcr").write_text("{broken\n")
        with pytest.raises(SystemExit) as exc:
            cli.main(["ci", "run", "--dataset", "shop", "--candidates", str(cand),
                      "--report", "r.json", "--markdown", "r.md", "--quiet"])
        assert exc.value.code == 1
        data = _strict_loads((tmp_path / "r.json").read_text())
        assert data["passed"] == 1 and data["failed"] == 1
        assert "cannot load candidate" in (tmp_path / "r.md").read_text()


# -- 3. GitHub workflow-command escaping ----------------------------------------


class TestAnnotations:
    def test_escaping_rules(self) -> None:
        line = github_annotation("error", "A: b, c%", "x%y\r\n::error::pwned")
        assert line == "::error title=A%3A b%2C c%25::x%25y%0D%0A::error::pwned"
        assert "\n" not in line and "\r" not in line

    def test_tool_name_cannot_inject_commands(self, tmp_path: Path, monkeypatch: Any,
                                              capsys: Any) -> None:
        ds = ReplayDataset.create("x", datasets_dir=tmp_path / "ds")
        cache = GoldenRunCache(cache_dir=str(tmp_path / "g"))
        rec, _ = _record(tmp_path / "g", "g")
        ds.add_case("t", rec, golden_cache=cache)
        _, cand = _record(tmp_path / "c", "c", tool="x\n::add-mask::secret")
        report = run_regression(ds, {"t": str(cand)}, cache)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        emit_github_annotations(report)
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 1
        assert out[0].startswith("::warning title=AgentCI t::")
        assert "%0A::add-mask::secret" in out[0]


# -- 4. atomic writes ------------------------------------------------------------


class TestAtomicWrites:
    def test_bundle_export_failure_leaves_nothing(self, tmp_path: Path,
                                                 monkeypatch: Any) -> None:
        import tarfile

        from agent_vcr.bundle import export_bundle

        _, src = _record(tmp_path, "s")
        dest = tmp_path / "out.vcrb"

        def boom(*_a: Any, **_k: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(tarfile.TarFile, "add", boom)
        with pytest.raises(OSError, match="disk full"):
            export_bundle(src, output=dest)
        assert not dest.exists()
        assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))

    def test_bundle_export_into_directory(self, tmp_path: Path) -> None:
        from agent_vcr.bundle import export_bundle, verify_bundle

        _, src = _record(tmp_path / "src", "sess")
        out_dir = tmp_path / "bundles"
        out_dir.mkdir()
        dest = export_bundle(src, output=out_dir)
        assert dest == out_dir / "sess.vcrb"
        assert verify_bundle(dest)["ok"] is True

    def test_dataset_save_leaves_no_tmp(self, tmp_path: Path) -> None:
        ds = ReplayDataset.create("d", datasets_dir=tmp_path)
        assert ds.version == 1
        assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


# -- 5. dataset names -----------------------------------------------------------


class TestDatasetNames:
    @pytest.mark.parametrize("name", ["../x", "a/b", "..", ".", "", "/abs", "a\\b",
                                      ".hidden", "nul\0byte"])
    def test_unsafe_names_rejected(self, tmp_path: Path, name: str) -> None:
        with pytest.raises(ValueError):
            ReplayDataset.create(name, datasets_dir=tmp_path / "ds")
        assert not (tmp_path / "x.json").exists()

    def test_load_does_not_create_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "typo"
        with pytest.raises(FileNotFoundError):
            ReplayDataset.load("refunds", datasets_dir=missing)
        assert not missing.exists()

    def test_cli_rejects_traversal(self, tmp_path: Path, monkeypatch: Any,
                                   capsys: Any) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            cli.main(["dataset", "create", "../escape"])
        assert exc.value.code == 1
        assert "Invalid dataset name" in capsys.readouterr().err
        assert not (tmp_path / ".vcr" / "escape.json").exists()


# -- 6. candidates dir + case-name matching -------------------------------------


class TestCandidates:
    def _setup(self, tmp_path: Path, monkeypatch: Any) -> Path:
        monkeypatch.chdir(tmp_path)
        _, golden = _record(tmp_path / "src", "src")
        cli.main(["dataset", "create", "shop"])
        cli.main(["dataset", "add", "shop", "--task", "Refund $10", "--vcr", str(golden)])
        return golden

    def test_missing_candidates_dir_errors(self, tmp_path: Path, monkeypatch: Any,
                                           capsys: Any) -> None:
        self._setup(tmp_path, monkeypatch)
        with pytest.raises(SystemExit) as exc:
            cli.main(["ci", "run", "--dataset", "shop", "--candidates", "nope"])
        assert exc.value.code == 1
        assert "--candidates directory not found: nope" in capsys.readouterr().err

    def test_slug_matches_task_with_spaces(self, tmp_path: Path, monkeypatch: Any,
                                           capsys: Any) -> None:
        golden = self._setup(tmp_path, monkeypatch)
        assert "refund-10.vcr" in capsys.readouterr().out
        assert case_slug("Refund $10") == "refund-10"
        cand = tmp_path / "cand"
        cand.mkdir()
        shutil.copy(golden, cand / "refund-10.vcr")
        shutil.copy(golden, cand / "stray.vcr")
        cli.main(["ci", "run", "--dataset", "shop", "--candidates", str(cand), "--quiet",
                  "--report", "r.json"])
        assert json.loads((tmp_path / "r.json").read_text())["passed"] == 1
        assert "stray.vcr matches no case" in capsys.readouterr().err

    def test_help_documents_matching(self, capsys: Any) -> None:
        with pytest.raises(SystemExit):
            cli.main(["ci", "run", "--help"])
        out = " ".join(capsys.readouterr().out.split())
        assert "defaults to the --task string" in out
        assert "refund-10.vcr" in out


# -- 7/8. TUI ---------------------------------------------------------------------


class TestTUI:
    @pytest.mark.asyncio
    async def test_resume_on_zero_frames(self, tmp_path: Path) -> None:
        pytest.importorskip("textual")
        from agent_vcr.tui import VCRApp

        _, path = _record(tmp_path, "empty", frames=False)
        app = VCRApp(str(path))
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            assert app.player is not None and app.player.frames == []
            await pilot.press("r")
            await pilot.press("e")
            await pilot.pause(0.05)
        assert not (tmp_path / "resume_state.json").exists()

    @pytest.mark.asyncio
    async def test_empty_edited_state_is_used(self, tmp_path: Path) -> None:
        pytest.importorskip("textual")
        from agent_vcr.tui import VCRApp

        _, path = _record(tmp_path, "s")
        app = VCRApp(str(path))
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            app._edited_state = {}
            app.action_resume()
            await pilot.pause(0.05)
        payload = json.loads((tmp_path / "resume_state.json").read_text())
        assert payload["edited"] is True
        assert payload["state"] == {}
        assert not list(tmp_path.glob(".*.tmp"))

    @pytest.mark.asyncio
    async def test_load_failure_sets_error_and_code(self, tmp_path: Path) -> None:
        pytest.importorskip("textual")
        from agent_vcr.tui import VCRApp

        bad = tmp_path / "bad.vcr"
        bad.write_text("garbage\n")
        app = VCRApp(str(bad))
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
        assert app.load_error and "Error loading VCR file" in app.load_error
        assert app.return_code == 1

    def test_cli_load_failure_exits_nonzero(self, tmp_path: Path, monkeypatch: Any,
                                            capsys: Any) -> None:
        pytest.importorskip("textual")
        from agent_vcr.tui import VCRApp

        original_run = VCRApp.run
        monkeypatch.setattr(VCRApp, "run",
                            lambda self, **kw: original_run(self, headless=True))
        bad = tmp_path / "bad.vcr"
        bad.write_text("garbage\n")
        with pytest.raises(SystemExit) as exc:
            cli.main([str(bad)])
        assert exc.value.code == 1
        assert "Error loading VCR file" in capsys.readouterr().err


# -- 9. CLI DX ---------------------------------------------------------------------


class TestCLIDX:
    def test_version(self, capsys: Any) -> None:
        import agent_vcr

        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"vcr {agent_vcr.__version__}"

    def test_unknown_command(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            cli.main(["frobnicate"])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "usage: vcr" in err and "unknown command 'frobnicate'" in err

    def test_existing_file_without_suffix_opens_tui(self, tmp_path: Path,
                                                    monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "recording").write_text("")
        launched: list[str] = []
        monkeypatch.setattr(cli, "_run_tui", launched.append)
        cli.main(["recording"])
        cli.main(["sub/dir/run.vcr"])
        assert launched == ["recording", "sub/dir/run.vcr"]

    def test_missing_tui_extra_hint(self, tmp_path: Path, monkeypatch: Any,
                                    capsys: Any) -> None:
        _, path = _record(tmp_path, "s")
        monkeypatch.delitem(sys.modules, "agent_vcr.tui", raising=False)
        # Hide textual and every already-imported textual.* submodule.
        for mod in [m for m in sys.modules if m == "textual" or m.startswith("textual.")]:
            monkeypatch.setitem(sys.modules, mod, None)
        monkeypatch.setitem(sys.modules, "textual", None)
        monkeypatch.setitem(sys.modules, "textual.app", None)
        with pytest.raises(SystemExit) as exc:
            cli.main([str(path)])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert 'pip install "ai-agent-vcr[tui]"' in err
        assert "Traceback" not in err and len(err.strip().splitlines()) == 1

    def test_init_corrupt_settings(self, tmp_path: Path, capsys: Any) -> None:
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / "settings.json").write_text("{not json")
        with pytest.raises(SystemExit) as exc:
            cli.main(["init", "--claude-code", "--project-dir", str(tmp_path)])
        assert exc.value.code == 1
        assert "is not valid JSON" in capsys.readouterr().err
        assert (tmp_path / ".claude" / "settings.json").read_text() == "{not json"

    def test_bundle_export_directory(self, tmp_path: Path, capsys: Any) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["bundle", "export", str(tmp_path)])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("Error:") and "is a directory" in err

    def test_bundle_export_unwritable_output(self, tmp_path: Path, capsys: Any) -> None:
        _, src = _record(tmp_path, "s")
        with pytest.raises(SystemExit) as exc:
            cli.main(["bundle", "export", str(src), "-o", str(tmp_path / "no" / "x.vcrb")])
        assert exc.value.code == 1
        assert "Output directory does not exist" in capsys.readouterr().err

    def test_token_without_base_url(self, capsys: Any) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["cloud", "status", "--token", "abc"])
        assert exc.value.code == 2
        assert "--token requires --base-url" in capsys.readouterr().err

    def test_serve_passes_args_through(self, monkeypatch: Any) -> None:
        import agent_vcr.server as server

        seen: list[list[str]] = []
        monkeypatch.setattr(server, "main", lambda argv=None: seen.append(list(argv or [])))
        cli.main(["serve", "--port", "9999", "--help"])
        assert seen == [["--port", "9999", "--help"]]
