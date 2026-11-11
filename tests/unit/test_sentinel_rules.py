"""Regression tests for Sentinel bug fixes (false PASS / false positives / wiring)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import VCRRecorder
from openhands_sentinel.analyzer import CodeAnalyzer, Severity
from openhands_sentinel.cli import (
    EXIT_BAD_TARGET,
    EXIT_BLOCKED,
    EXIT_NO_FILES,
    EXIT_PASSED,
    scan_directory,
    watch_directory,
)
from openhands_sentinel.sentinel import Sentinel, SentinelConfig

SYNTAX_ERROR = "def broken(:\n    pass\n"


def _sentinel(tmp_path: Path, **config: Any) -> Sentinel:
    s = Sentinel(
        config=SentinelConfig(**config),
        recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr-rec")),
    )
    s.start_session("fixes")
    return s


def _rules(result: Any, rule: str) -> list[Any]:
    return [v for v in result.violations if v.rule == rule]


# ── Bug 1: dot-dir components of the scan root itself ──────────────────


class TestScanRootUnderDotDir:
    def _make_project(self, tmp_path: Path) -> Path:
        proj = tmp_path / ".hidden" / "proj"
        (proj / "pkg").mkdir(parents=True)
        (proj / "pkg" / "bad.py").write_text(SYNTAX_ERROR)
        return proj

    def test_absolute_path_under_hidden_dir(self, tmp_path: Path) -> None:
        proj = self._make_project(tmp_path)
        assert scan_directory(str(proj), SentinelConfig()) == EXIT_BLOCKED

    def test_relative_parent_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._make_project(tmp_path)
        other = tmp_path / ".hidden" / "other"
        other.mkdir()
        monkeypatch.chdir(other)
        assert scan_directory("../proj", SentinelConfig()) == EXIT_BLOCKED

    def test_dot_dirs_inside_root_still_skipped(self, tmp_path: Path) -> None:
        proj = tmp_path / ".hidden" / "proj"
        (proj / ".cache").mkdir(parents=True)
        (proj / ".cache" / "bad.py").write_text(SYNTAX_ERROR)
        (proj / "venv").mkdir()
        (proj / "venv" / "bad.py").write_text(SYNTAX_ERROR)
        (proj / "ok.py").write_text("x = 1\n")
        assert scan_directory(str(proj), SentinelConfig()) == EXIT_PASSED

    def test_cli_process_exit_code(self, tmp_path: Path) -> None:
        proj = self._make_project(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "openhands_sentinel.cli", "scan", str(proj)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode != 0
        assert "PASSED" not in proc.stdout


# ── Bug 2: missing target / no Python files ────────────────────────────


class TestMissingOrEmptyTarget:
    def test_missing_directory_is_error_and_not_created(self, tmp_path: Path) -> None:
        missing = tmp_path / "typo_dir"
        assert scan_directory(str(missing), SentinelConfig()) == EXIT_BAD_TARGET
        assert not missing.exists()

    def test_file_target_is_error(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        assert scan_directory(str(f), SentinelConfig()) == EXIT_BAD_TARGET

    def test_watch_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        code = watch_directory(str(missing), SentinelConfig(), interval=0.01, once=True)
        assert code == EXIT_BAD_TARGET
        assert not missing.exists()

    def test_no_python_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "README.md").write_text("hi\n")
        assert scan_directory(str(tmp_path), SentinelConfig()) == EXIT_NO_FILES
        out = capsys.readouterr().out
        assert "No Python files found" in out
        assert "PASSED" not in out
        assert EXIT_NO_FILES != 0


# ── Bug 3: qualified-name duplicate detection ──────────────────────────


class TestQualifiedDuplicates:
    def test_same_method_name_in_different_classes_not_flagged(self) -> None:
        a = CodeAnalyzer()
        a.analyze("a.py", "class A:\n    def run(self):\n        return 1\n")
        r = a.analyze("b.py", "class B:\n    def run(self):\n        return 2\n")
        assert _rules(r, "duplicate_function") == []
        assert r.passed

    def test_same_method_name_same_body_different_classes_not_flagged(self) -> None:
        a = CodeAnalyzer()
        a.analyze("a.py", "class A:\n    def run(self):\n        return 1\n")
        r = a.analyze("b.py", "class B:\n    def run(self):\n        return 1\n")
        assert _rules(r, "duplicate_function") == []

    def test_dunders_not_flagged(self) -> None:
        a = CodeAnalyzer()
        code = "class A:\n    def __init__(self):\n        self.x = 1\n"
        a.analyze("a.py", code)
        r = a.analyze("b.py", code)
        assert _rules(r, "duplicate_function") == []

    def test_copy_pasted_method_same_class_flagged(self) -> None:
        a = CodeAnalyzer()
        code = "class Repo:\n    def save(self, x):\n        return x * 2\n"
        a.analyze("a.py", code)
        r = a.analyze("b.py", code)
        dupes = _rules(r, "duplicate_function")
        assert len(dupes) == 1
        assert dupes[0].function_name == "Repo.save"
        assert dupes[0].severity == Severity.CRITICAL

    def test_same_toplevel_name_different_body_is_warning(self) -> None:
        a = CodeAnalyzer()
        a.analyze("a.py", "def main():\n    return 1\n")
        r = a.analyze("b.py", "def main():\n    print('x')\n")
        dupes = _rules(r, "duplicate_function")
        assert len(dupes) == 1
        assert dupes[0].severity == Severity.WARNING
        assert dupes[0].details["same_body"] is False
        assert r.passed

    def test_identical_toplevel_ignores_formatting_and_comments(self) -> None:
        a = CodeAnalyzer()
        a.analyze("a.py", "def f(x):\n    return x+1\n")
        r = a.analyze("b.py", "# copy\ndef f(x):\n    return (x + 1)  # same\n")
        dupes = _rules(r, "duplicate_function")
        assert len(dupes) == 1
        assert dupes[0].severity == Severity.CRITICAL

    def test_nested_functions_not_tracked(self) -> None:
        a = CodeAnalyzer()
        code = "def deco(fn):\n    def wrapper():\n        return fn()\n    return wrapper\n"
        a.analyze("a.py", code.replace("deco", "deco_a"))
        r = a.analyze("b.py", code.replace("deco", "deco_b"))
        assert _rules(r, "duplicate_function") == []

    def test_cli_scan_methods_in_two_files_passes(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("class A:\n    def run(self):\n        return 1\n")
        (tmp_path / "b.py").write_text("class B:\n    def run(self):\n        return 2\n")
        assert scan_directory(str(tmp_path), SentinelConfig()) == EXIT_PASSED


# ── Bug 4: growth measured per edit with a minimum size ────────────────


class TestGrowthRate:
    def test_stub_filled_in_not_flagged(self) -> None:
        a = CodeAnalyzer()
        a.analyze("app.py", "def f():\n    pass\n")
        body = "def f():\n" + "".join(f"    x{i} = {i}\n" for i in range(7))
        r = a.analyze("app.py", body)
        assert _rules(r, "rapid_growth") == []

    def test_not_critical_forever(self) -> None:
        a = CodeAnalyzer(max_file_lines=9999)
        a.analyze("app.py", "x = 1\n")
        big = "".join(f"v{i} = {i}\n" for i in range(100))
        assert len(_rules(a.analyze("app.py", big), "rapid_growth")) == 1
        # A small follow-up edit must not re-report growth vs the first version
        r = a.analyze("app.py", big + "w = 1\n")
        assert _rules(r, "rapid_growth") == []

    def test_large_single_edit_flagged_against_previous(self) -> None:
        a = CodeAnalyzer(max_file_lines=9999)
        a.analyze("app.py", "".join(f"v{i} = {i}\n" for i in range(30)))
        r = a.analyze("app.py", "".join(f"v{i} = {i}\n" for i in range(200)))
        growth = _rules(r, "rapid_growth")
        assert len(growth) == 1
        assert growth[0].details["previous_lines"] == 31


# ── Bug 5: stale index entries ─────────────────────────────────────────


class TestStaleIndex:
    def test_moved_function_not_flagged(self) -> None:
        a = CodeAnalyzer()
        a.analyze("old.py", "def helper():\n    return 1\n")
        a.analyze("old.py", "x = 1\n")  # helper removed from old.py
        r = a.analyze("new.py", "def helper():\n    return 1\n")
        assert _rules(r, "duplicate_function") == []

    def test_moved_class_not_flagged(self) -> None:
        a = CodeAnalyzer()
        a.analyze("old.py", "class C:\n    pass\n")
        a.analyze("old.py", "x = 1\n")
        r = a.analyze("new.py", "class C:\n    pass\n")
        assert _rules(r, "duplicate_class") == []

    def test_forget_file_on_delete(self) -> None:
        a = CodeAnalyzer()
        a.analyze("old.py", "def helper():\n    return 1\n")
        a.forget_file("old.py")
        r = a.analyze("new.py", "def helper():\n    return 1\n")
        assert _rules(r, "duplicate_function") == []
        assert a.get_trajectory_summary()["tracked_files"] == 1

    def test_sentinel_forget_file(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        s.check_file("old.py", "def helper():\n    return 1\n")
        s.forget_file("old.py")
        r = s.check_file("new.py", "def helper():\n    return 1\n")
        assert _rules(r, "duplicate_function") == []


# ── Bug 6: parameter counting ──────────────────────────────────────────


class TestParamCounting:
    def _count(self, code: str) -> int | None:
        a = CodeAnalyzer(max_function_params=0)
        found = _rules(a.analyze("app.py", code), "too_many_params")
        return found[0].details["param_count"] if found else None

    def test_all_kinds_counted(self) -> None:
        code = "def f(a, /, b, *args, c, d=1, **kw):\n    pass\n"
        assert self._count(code) == 6

    def test_self_and_cls_excluded(self) -> None:
        code = (
            "class K:\n"
            "    def m(self, a):\n        pass\n"
            "    @classmethod\n    def c(cls, a):\n        pass\n"
        )
        a = CodeAnalyzer(max_function_params=0)
        counts = [
            v.details["param_count"]
            for v in _rules(a.analyze("app.py", code), "too_many_params")
        ]
        assert counts == [1, 1]

    def test_staticmethod_keeps_first_param(self) -> None:
        code = "class K:\n    @staticmethod\n    def s(self, a):\n        pass\n"
        assert self._count(code) == 2

    def test_method_with_max_params_not_flagged_due_to_self(self) -> None:
        a = CodeAnalyzer(max_function_params=2)
        r = a.analyze("app.py", "class K:\n    def m(self, a, b):\n        pass\n")
        assert _rules(r, "too_many_params") == []

    def test_kwonly_bloat_flagged(self) -> None:
        a = CodeAnalyzer(max_function_params=3)
        r = a.analyze("app.py", "def f(*, a, b, c, d, e):\n    pass\n")
        assert len(_rules(r, "too_many_params")) == 1


# ── Bug 7: sentinel.py wiring ──────────────────────────────────────────


class FileWriteAction:
    def __init__(self, path: str, content: str) -> None:
        self.path = path
        self.content = content


class FileEditAction:
    def __init__(self, path: str, **kw: Any) -> None:
        self.path = path
        self.content = ""
        self.command = kw.get("command", "str_replace")
        self.file_text = kw.get("file_text")
        self.old_str = kw.get("old_str")
        self.new_str = kw.get("new_str")


class FileEditObservation:
    def __init__(self, path: str, new_content: str | None = None) -> None:
        self.path = path
        self.new_content = new_content


class FakeStream:
    def __init__(self) -> None:
        self.events: list[tuple[Any, Any]] = []
        self.subscribers: list[Any] = []

    def subscribe(self, _sid: Any, callback: Any, _cid: str) -> None:
        self.subscribers.append(callback)

    def add_event(self, event: Any, source: Any) -> None:
        self.events.append((event, source))


class TestBlockOnBlocker:
    def test_blocker_marks_result_blocked(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        result, warning = s.check_and_warn("bad.py", SYNTAX_ERROR)
        assert result.blocked
        assert warning is not None and "BLOCKED" in warning
        assert s.stats.blocked_actions == 1
        assert s.stats.to_dict()["blocked_actions"] == 1
        assert "Blocked actions" in s.get_report()

    def test_disabled_does_not_block(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path, block_on_blocker=False)
        result, warning = s.check_and_warn("bad.py", SYNTAX_ERROR)
        assert not result.blocked
        assert warning is not None and "BLOCKED" not in warning
        assert s.stats.blocked_actions == 0

    def test_non_blocker_not_blocked(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        assert not s.check_file("a.py", "from os import *\n").blocked

    def test_event_path_returns_blocked_result(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        r = s._handle_file_event(FileWriteAction("bad.py", SYNTAX_ERROR))
        assert r is not None and r.blocked


class TestAutoAnalyze:
    def test_auto_analyze_false_skips_analysis(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path, auto_analyze=False)
        s.attach(FakeStream())
        s._on_event(FileWriteAction("a.py", "x = 1\n"))
        assert s.stats.files_analyzed == 0

    def test_auto_analyze_true_analyzes(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        s.attach(FakeStream())
        s._on_event(FileWriteAction("a.py", "x = 1\n"))
        assert s.stats.files_analyzed == 1


class TestFileEditAction:
    def test_str_replace_analyzed_from_disk_after_edit(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("x = 1\n")
        s = _sentinel(tmp_path)
        s.attach(FakeStream())
        s._on_event(FileEditAction(str(target), old_str="x = 1", new_str="from os import *"))
        assert s.stats.files_analyzed == 0  # edit not applied yet
        target.write_text("from os import *\n")  # runtime applies the edit
        s._on_event(FileEditObservation(str(target)))
        assert s.stats.files_analyzed == 1
        assert s._results_history[-1].warning_count == 1

    def test_observation_new_content_preferred(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        s.attach(FakeStream())
        path = str(tmp_path / "missing.py")
        s._on_event(FileEditAction(path, old_str="a", new_str="b"))
        s._on_event(FileEditObservation(path, new_content=SYNTAX_ERROR))
        assert s.stats.blockers == 1

    def test_create_uses_file_text(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        s.attach(FakeStream())
        s._on_event(FileEditAction("new.py", command="create", file_text="x = 1\n"))
        assert s.stats.files_analyzed == 1

    def test_unrelated_observation_ignored(self, tmp_path: Path) -> None:
        s = _sentinel(tmp_path)
        s.attach(FakeStream())
        s._on_event(FileEditObservation("other.py", new_content="x = 1\n"))
        assert s.stats.files_analyzed == 0


class TestWarningInjection:
    def test_source_and_failure_logging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import types

        msg_mod = types.ModuleType("openhands.events.action")

        class MessageAction:
            def __init__(self, content: str, wait_for_response: bool) -> None:
                self.content = content

        msg_mod.MessageAction = MessageAction  # type: ignore[attr-defined]
        for name in ("openhands", "openhands.events", "openhands.events.event"):
            monkeypatch.setitem(sys.modules, name, None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "openhands.events.action", msg_mod)

        stream = FakeStream()
        s = _sentinel(tmp_path)
        s.attach(stream)
        s._on_event(FileWriteAction("a.py", "from os import *\n"))
        assert len(stream.events) == 1
        assert stream.events[0][1] == "user"

        def boom(event: Any, source: Any) -> None:
            raise RuntimeError("nope")

        stream.add_event = boom  # type: ignore[method-assign]
        with caplog.at_level(logging.WARNING, logger="openhands_sentinel.sentinel"):
            s._on_event(FileWriteAction("b.py", "from os import *\n"))
        assert any("Failed to inject warning" in r.message for r in caplog.records)
