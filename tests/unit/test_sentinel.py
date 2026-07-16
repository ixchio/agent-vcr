"""
Tests for OpenHands Sentinel — the real-time code quality guardian.

Covers:
  - CodeAnalyzer: duplicate detection, complexity, function length,
    parameter bloat, file size, class bloat, wildcard imports, growth rate,
    syntax errors, trajectory memory, non-Python files
  - Sentinel: session lifecycle, check_file, check_and_warn, VCR recording,
    report generation, stats tracking, violation callbacks, self-correction
  - SentinelConfig: custom thresholds
  - CLI: scan_directory (captures exit codes)
"""

from __future__ import annotations

import os

import pytest

from agent_vcr import VCRRecorder
from openhands_sentinel.analyzer import (
    AnalysisResult,
    CodeAnalyzer,
    Severity,
    Violation,
)
from openhands_sentinel.sentinel import Sentinel, SentinelConfig, SentinelStats

# ═══════════════════════════════════════════════════
#  CodeAnalyzer Tests
# ═══════════════════════════════════════════════════


class TestCodeAnalyzerDuplicates:
    """Cross-file duplicate function detection."""

    def test_duplicate_function_across_files(self):
        analyzer = CodeAnalyzer()
        # First file defines hash_password
        code_a = "def hash_password(pw):\n    return pw\n"
        result_a = analyzer.analyze("auth/utils.py", code_a)
        assert len(result_a.violations) == 0

        # Second file duplicates it
        code_b = "def hash_password(pw):\n    return pw\n"
        result_b = analyzer.analyze("api/handlers.py", code_b)

        dupes = [v for v in result_b.violations if v.rule == "duplicate_function"]
        assert len(dupes) == 1
        assert dupes[0].severity == Severity.CRITICAL
        assert "auth/utils.py" in dupes[0].message
        assert dupes[0].function_name == "hash_password"

    def test_same_name_same_file_no_violation(self):
        analyzer = CodeAnalyzer()
        # Rewriting the same file should NOT flag duplicate
        code = "def process():\n    pass\n"
        analyzer.analyze("app.py", code)
        result = analyzer.analyze("app.py", code)  # same file again
        dupes = [v for v in result.violations if v.rule == "duplicate_function"]
        assert len(dupes) == 0

    def test_different_function_names_no_violation(self):
        analyzer = CodeAnalyzer()
        analyzer.analyze("a.py", "def foo():\n    pass\n")
        result = analyzer.analyze("b.py", "def bar():\n    pass\n")
        assert len(result.violations) == 0

    def test_duplicate_class_across_files(self):
        analyzer = CodeAnalyzer()
        analyzer.analyze("models/user.py", "class User:\n    pass\n")
        result = analyzer.analyze("api/models.py", "class User:\n    pass\n")
        dupes = [v for v in result.violations if v.rule == "duplicate_class"]
        assert len(dupes) == 1
        assert "models/user.py" in dupes[0].message


class TestCodeAnalyzerComplexity:
    """Cyclomatic complexity detection."""

    def test_simple_function_passes(self):
        analyzer = CodeAnalyzer(max_complexity=10)
        code = "def simple():\n    return 42\n"
        result = analyzer.analyze("app.py", code)
        complexity_violations = [v for v in result.violations if v.rule == "high_complexity"]
        assert len(complexity_violations) == 0

    def test_high_complexity_caught(self):
        analyzer = CodeAnalyzer(max_complexity=3)
        code = (
            "def complex_func(x):\n"
            "    if x > 0:\n"
            "        if x > 10:\n"
            "            if x > 100:\n"
            "                for i in range(x):\n"
            "                    if i % 2 == 0:\n"
            "                        pass\n"
            "    return x\n"
        )
        result = analyzer.analyze("app.py", code)
        cx = [v for v in result.violations if v.rule == "high_complexity"]
        assert len(cx) == 1
        assert cx[0].function_name == "complex_func"
        assert cx[0].details["complexity"] > 3

    def test_boolean_operators_add_complexity(self):
        analyzer = CodeAnalyzer(max_complexity=2)
        code = (
            "def check(a, b, c, d):\n"
            "    if a and b and c and d:\n"
            "        return True\n"
            "    return False\n"
        )
        result = analyzer.analyze("app.py", code)
        cx = [v for v in result.violations if v.rule == "high_complexity"]
        assert len(cx) == 1
        # if (1) + 3 boolean ops (and, and, and) = complexity 5
        assert cx[0].details["complexity"] >= 4


class TestCodeAnalyzerFunctionLength:
    """Function length detection."""

    def test_short_function_passes(self):
        analyzer = CodeAnalyzer(max_function_lines=50)
        code = "def short():\n    return 1\n"
        result = analyzer.analyze("app.py", code)
        long = [v for v in result.violations if v.rule == "function_too_long"]
        assert len(long) == 0

    def test_long_function_caught(self):
        analyzer = CodeAnalyzer(max_function_lines=5)
        lines = ["def monster():"]
        for i in range(20):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        code = "\n".join(lines) + "\n"

        result = analyzer.analyze("app.py", code)
        long = [v for v in result.violations if v.rule == "function_too_long"]
        assert len(long) == 1
        assert long[0].function_name == "monster"
        assert long[0].details["length"] > 5

    def test_very_long_function_is_critical(self):
        analyzer = CodeAnalyzer(max_function_lines=10)
        lines = ["def huge():"]
        for i in range(25):  # > 2x threshold
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        code = "\n".join(lines) + "\n"

        result = analyzer.analyze("app.py", code)
        long = [v for v in result.violations if v.rule == "function_too_long"]
        assert len(long) == 1
        assert long[0].severity == Severity.CRITICAL

    def test_dunder_methods_skipped(self):
        analyzer = CodeAnalyzer(max_function_lines=3)
        lines = ["class Foo:"]
        lines.append("    def __init__(self):")
        for i in range(10):
            lines.append(f"        self.x_{i} = {i}")
        code = "\n".join(lines) + "\n"

        result = analyzer.analyze("app.py", code)
        long = [v for v in result.violations if v.rule == "function_too_long"]
        assert len(long) == 0


class TestCodeAnalyzerParams:
    """Parameter bloat detection."""

    def test_normal_params_pass(self):
        analyzer = CodeAnalyzer(max_function_params=7)
        code = "def ok(a, b, c):\n    pass\n"
        result = analyzer.analyze("app.py", code)
        params = [v for v in result.violations if v.rule == "too_many_params"]
        assert len(params) == 0

    def test_too_many_params_caught(self):
        analyzer = CodeAnalyzer(max_function_params=3)
        code = "def bloated(a, b, c, d, e, f):\n    pass\n"
        result = analyzer.analyze("app.py", code)
        params = [v for v in result.violations if v.rule == "too_many_params"]
        assert len(params) == 1
        assert params[0].details["param_count"] == 6
        assert "config object" in params[0].message


class TestCodeAnalyzerFileSize:
    """File size and growth detection."""

    def test_small_file_passes(self):
        analyzer = CodeAnalyzer(max_file_lines=500)
        code = "x = 1\ny = 2\n"
        result = analyzer.analyze("app.py", code)
        size = [v for v in result.violations if v.rule == "file_too_large"]
        assert len(size) == 0

    def test_large_file_caught(self):
        analyzer = CodeAnalyzer(max_file_lines=10)
        code = "\n".join([f"line_{i} = {i}" for i in range(50)]) + "\n"
        result = analyzer.analyze("app.py", code)
        size = [v for v in result.violations if v.rule == "file_too_large"]
        assert len(size) == 1

    def test_rapid_growth_detected(self):
        analyzer = CodeAnalyzer(max_file_lines=9999)
        # First version: small
        analyzer.analyze("app.py", "x = 1\n")
        # Second version: 10x bigger (> 200% growth)
        big = "\n".join([f"line_{i} = {i}" for i in range(100)]) + "\n"
        result = analyzer.analyze("app.py", big)
        growth = [v for v in result.violations if v.rule == "rapid_growth"]
        assert len(growth) == 1
        assert growth[0].details["growth_percent"] > 200


class TestCodeAnalyzerClassBloat:
    """Class method count detection."""

    def test_small_class_passes(self):
        analyzer = CodeAnalyzer(max_class_methods=15)
        code = "class Foo:\n    def a(self):\n        pass\n"
        result = analyzer.analyze("app.py", code)
        bloat = [v for v in result.violations if v.rule == "class_too_large"]
        assert len(bloat) == 0

    def test_bloated_class_caught(self):
        analyzer = CodeAnalyzer(max_class_methods=3)
        methods = "\n".join(
            [f"    def method_{i}(self):\n        pass" for i in range(5)]
        )
        code = f"class God:\n{methods}\n"
        result = analyzer.analyze("app.py", code)
        bloat = [v for v in result.violations if v.rule == "class_too_large"]
        assert len(bloat) == 1
        assert bloat[0].details["method_count"] == 5


class TestCodeAnalyzerImports:
    """Wildcard import detection."""

    def test_normal_import_passes(self):
        analyzer = CodeAnalyzer()
        code = "from os import path\n"
        result = analyzer.analyze("app.py", code)
        wildcards = [v for v in result.violations if v.rule == "wildcard_import"]
        assert len(wildcards) == 0

    def test_wildcard_import_caught(self):
        analyzer = CodeAnalyzer()
        code = "from os import *\n"
        result = analyzer.analyze("app.py", code)
        wildcards = [v for v in result.violations if v.rule == "wildcard_import"]
        assert len(wildcards) == 1
        assert "import specific names" in wildcards[0].message


class TestCodeAnalyzerEdgeCases:
    """Edge cases and special behavior."""

    def test_syntax_error_returns_blocker(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze("bad.py", "def broken(:\n    pass\n")
        assert len(result.violations) == 1
        assert result.violations[0].rule == "syntax_error"
        assert result.violations[0].severity == Severity.BLOCKER

    def test_non_python_file_returns_empty(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze("readme.md", "# Hello world")
        assert len(result.violations) == 0
        assert result.metrics == {}

    def test_empty_file_passes(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze("empty.py", "")
        assert len(result.violations) == 0

    def test_trajectory_memory_persists(self):
        analyzer = CodeAnalyzer()
        analyzer.analyze("a.py", "def shared():\n    pass\n")
        summary = analyzer.get_trajectory_summary()
        assert summary["tracked_functions"] == 1
        assert summary["tracked_files"] == 1

    def test_reset_clears_memory(self):
        analyzer = CodeAnalyzer()
        analyzer.analyze("a.py", "def foo():\n    pass\n")
        analyzer.reset()
        summary = analyzer.get_trajectory_summary()
        assert summary["tracked_functions"] == 0
        assert summary["tracked_files"] == 0

    def test_async_functions_analyzed(self):
        analyzer = CodeAnalyzer(max_function_lines=3)
        lines = ["async def long_async():"]
        for i in range(10):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        code = "\n".join(lines) + "\n"
        result = analyzer.analyze("app.py", code)
        long = [v for v in result.violations if v.rule == "function_too_long"]
        assert len(long) == 1
        assert long[0].function_name == "long_async"


class TestCodeAnalyzerMetrics:
    """Metrics computation."""

    def test_metrics_computed(self):
        analyzer = CodeAnalyzer()
        code = (
            "import os\nfrom sys import argv\n\n"
            "class Foo:\n    def bar(self):\n        pass\n\n"
            "def standalone():\n    pass\n"
        )
        result = analyzer.analyze("app.py", code)
        assert result.metrics["import_count"] == 2
        assert result.metrics["class_count"] == 1
        assert result.metrics["function_count"] == 2  # bar + standalone


# ═══════════════════════════════════════════════════
#  Violation / AnalysisResult Model Tests
# ═══════════════════════════════════════════════════


class TestViolation:
    def test_to_dict(self):
        v = Violation(
            rule="duplicate_function",
            message="foo already exists",
            severity=Severity.CRITICAL,
            file_path="app.py",
            line=10,
            function_name="foo",
        )
        d = v.to_dict()
        assert d["rule"] == "duplicate_function"
        assert d["severity"] == "critical"
        assert d["line"] == 10

    def test_to_agent_warning(self):
        v = Violation(
            rule="high_complexity",
            message="Too complex",
            severity=Severity.WARNING,
            file_path="utils.py",
            line=5,
        )
        warn = v.to_agent_warning()
        assert "SENTINEL" in warn
        assert "WARNING" in warn
        assert "utils.py:5" in warn


class TestAnalysisResult:
    def test_passed_with_no_violations(self):
        r = AnalysisResult(file_path="app.py")
        assert r.passed is True

    def test_passed_with_only_warnings(self):
        r = AnalysisResult(file_path="app.py", violations=[
            Violation(rule="x", message="x", severity=Severity.WARNING, file_path="app.py"),
        ])
        assert r.passed is True

    def test_not_passed_with_critical(self):
        r = AnalysisResult(file_path="app.py", violations=[
            Violation(rule="x", message="x", severity=Severity.CRITICAL, file_path="app.py"),
        ])
        assert r.passed is False

    def test_counts(self):
        r = AnalysisResult(file_path="app.py", violations=[
            Violation(rule="a", message="a", severity=Severity.BLOCKER, file_path="app.py"),
            Violation(rule="b", message="b", severity=Severity.CRITICAL, file_path="app.py"),
            Violation(rule="c", message="c", severity=Severity.WARNING, file_path="app.py"),
            Violation(rule="d", message="d", severity=Severity.WARNING, file_path="app.py"),
        ])
        assert r.blocker_count == 1
        assert r.critical_count == 1
        assert r.warning_count == 2


# ═══════════════════════════════════════════════════
#  Sentinel Orchestrator Tests
# ═══════════════════════════════════════════════════


class TestSentinel:
    """Sentinel orchestrator integration tests."""

    @pytest.fixture()
    def workspace(self, tmp_path):
        return str(tmp_path / ".vcr")

    @pytest.fixture()
    def sentinel(self, workspace):
        recorder = VCRRecorder(output_dir=workspace)
        s = Sentinel(recorder=recorder)
        s.start_session("test-session")
        return s

    def test_check_clean_file(self, sentinel):
        result = sentinel.check_file("app.py", "def hello():\n    return 'hi'\n")
        assert result.passed
        assert len(result.violations) == 0
        assert sentinel.stats.files_analyzed == 1

    def test_check_file_with_violations(self, sentinel):
        code = "def bloated(a, b, c, d, e, f, g, h, i):\n    pass\n"
        sentinel.check_file("app.py", code)
        assert sentinel.stats.total_violations > 0

    def test_stats_accumulate(self, sentinel):
        sentinel.check_file("a.py", "def ok():\n    pass\n")
        sentinel.check_file("b.py", "def ok2():\n    pass\n")
        assert sentinel.stats.files_analyzed == 2

    def test_check_and_warn_clean(self, sentinel):
        result, warning = sentinel.check_and_warn("app.py", "x = 1\n")
        assert warning is None

    def test_check_and_warn_with_violations(self, sentinel):
        code = "def bloated(a, b, c, d, e, f, g, h, i):\n    pass\n"
        result, warning = sentinel.check_and_warn("app.py", code)
        assert warning is not None
        assert "SENTINEL" in warning
        assert "please fix" in warning.lower()
        assert sentinel.stats.agent_warnings_sent == 1

    def test_vcr_frames_recorded(self, sentinel, workspace):
        sentinel.check_file("app.py", "x = 1\n")
        path = sentinel.save()
        assert os.path.exists(path)

        from agent_vcr import VCRPlayer
        player = VCRPlayer.load(path)
        assert len(player.frames) >= 1
        # Sentinel records analysis as a frame
        sentinel_frames = [f for f in player.frames if f.node_name == "sentinel:analysis"]
        assert len(sentinel_frames) == 1

    def test_violation_callback_fires(self, workspace):
        fired = []
        recorder = VCRRecorder(output_dir=workspace)
        s = Sentinel(recorder=recorder, on_violation=lambda v: fired.append(v))
        s.start_session("cb-test")

        code = "def bloated(a, b, c, d, e, f, g, h, i, j):\n    pass\n"
        s.check_file("app.py", code)
        assert len(fired) > 0
        assert all(isinstance(v, Violation) for v in fired)

    def test_get_report(self, sentinel):
        sentinel.check_file("a.py", "def foo():\n    pass\n")
        sentinel.check_file("b.py", "from os import *\n")
        report = sentinel.get_report()
        assert "SENTINEL SESSION REPORT" in report
        assert "Files analyzed" in report
        assert "2" in report

    def test_trajectory_detection_through_sentinel(self, sentinel):
        """End-to-end: Sentinel catches cross-file duplicate via trajectory."""
        sentinel.check_file("auth/utils.py", "def verify_token(t):\n    return True\n")
        result = sentinel.check_file("api/auth.py", "def verify_token(t):\n    return True\n")

        dupes = [v for v in result.violations if v.rule == "duplicate_function"]
        assert len(dupes) == 1
        assert "auth/utils.py" in dupes[0].message

    def test_self_correction_flow(self, sentinel):
        """Simulate: agent writes bad code → sentinel flags → agent rewrites clean."""
        # Bad version
        bad = "def handle(a, b, c, d, e, f, g, h, i, j):\n    pass\n"
        r1 = sentinel.check_file("handler.py", bad)
        assert len(r1.violations) > 0

        # Agent rewrites
        good = "def handle(request, ctx):\n    pass\n"
        r2 = sentinel.check_file("handler.py", good)
        assert len(r2.violations) == 0
        assert sentinel.stats.self_corrections == 1


class TestSentinelConfig:
    """Custom config thresholds."""

    def test_strict_config(self, tmp_path):
        config = SentinelConfig(
            max_function_lines=10,
            max_complexity=3,
            max_file_lines=50,
            max_function_params=3,
        )
        recorder = VCRRecorder(output_dir=str(tmp_path / ".vcr"))
        s = Sentinel(config=config, recorder=recorder)
        s.start_session("strict")

        code = "def f(a, b, c, d):\n    pass\n"  # 4 params > 3
        result = s.check_file("app.py", code)
        params = [v for v in result.violations if v.rule == "too_many_params"]
        assert len(params) == 1

    def test_relaxed_config(self, tmp_path):
        config = SentinelConfig(
            max_function_lines=999,
            max_complexity=999,
            max_function_params=999,
        )
        recorder = VCRRecorder(output_dir=str(tmp_path / ".vcr"))
        s = Sentinel(config=config, recorder=recorder)
        s.start_session("relaxed")

        code = "def f(a, b, c, d, e, f, g, h, i, j):\n    pass\n"
        result = s.check_file("app.py", code)
        assert len(result.violations) == 0

    def test_auto_start_session(self, tmp_path):
        """Sentinel auto-starts a session if check_file is called first."""
        recorder = VCRRecorder(output_dir=str(tmp_path / ".vcr"))
        s = Sentinel(recorder=recorder)
        # Don't call start_session — it should auto-start
        result = s.check_file("app.py", "x = 1\n")
        assert result is not None
        assert s._session_started


class TestSentinelStats:
    def test_to_dict(self):
        stats = SentinelStats(files_analyzed=5, total_violations=3, blockers=1)
        d = stats.to_dict()
        assert d["files_analyzed"] == 5
        assert d["blockers"] == 1


# ═══════════════════════════════════════════════════
#  CLI Tests
# ═══════════════════════════════════════════════════


class TestSentinelCLI:
    """Test the scan_directory function directly."""

    def test_scan_clean_directory(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello():\n    return 1\n")
        (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")

        from openhands_sentinel.cli import scan_directory

        config = SentinelConfig()
        exit_code = scan_directory(str(tmp_path), config)
        assert exit_code == 0

    def test_scan_directory_with_violations(self, tmp_path):
        # Create a file with critical violations
        lines = ["def monster():"]
        for i in range(120):
            lines.append(f"    x_{i} = {i}")
        lines.append("    return x_0")
        (tmp_path / "bad.py").write_text("\n".join(lines) + "\n")

        from openhands_sentinel.cli import scan_directory

        config = SentinelConfig(max_function_lines=10)
        exit_code = scan_directory(str(tmp_path), config)
        assert exit_code >= 1  # critical or blocker found

    def test_scan_empty_directory_passes(self, tmp_path):
        """Scanning an empty directory (no .py files) should pass cleanly."""
        from openhands_sentinel.cli import scan_directory

        empty = tmp_path / "empty_project"
        empty.mkdir()
        exit_code = scan_directory(str(empty), SentinelConfig())
        assert exit_code == 0

    def test_scan_skips_hidden_and_venv(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.py").write_text("from os import *\n")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.py").write_text("from os import *\n")
        (tmp_path / "app.py").write_text("x = 1\n")

        from openhands_sentinel.cli import scan_directory

        exit_code = scan_directory(str(tmp_path), SentinelConfig())
        assert exit_code == 0  # only app.py scanned, which is clean

    def test_watch_directory_once(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello():\n    return 1\n")

        from openhands_sentinel.cli import watch_directory

        exit_code = watch_directory(str(tmp_path), SentinelConfig(), interval=0.01, once=True)
        assert exit_code == 0
        assert (tmp_path / ".vcr").exists()


# ═══════════════════════════════════════════════════
#  Real-World Scenario Tests
# ═══════════════════════════════════════════════════


class TestRealWorldScenarios:
    """End-to-end scenarios mimicking actual agent behavior."""

    @pytest.fixture()
    def sentinel(self, tmp_path):
        recorder = VCRRecorder(output_dir=str(tmp_path / ".vcr"))
        s = Sentinel(recorder=recorder)
        s.start_session("scenario")
        return s

    def test_agent_builds_project_incrementally(self, sentinel):
        """Simulate an agent writing multiple files in sequence."""
        # Step 1: Agent writes clean auth module
        r1 = sentinel.check_file("auth/utils.py", (
            "def hash_password(password: str) -> str:\n"
            "    import bcrypt\n"
            "    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()\n\n"
            "def verify_password(password: str, hashed: str) -> bool:\n"
            "    import bcrypt\n"
            "    return bcrypt.checkpw(password.encode(), hashed.encode())\n"
        ))
        assert r1.passed

        # Step 2: Agent writes clean API routes
        r2 = sentinel.check_file("api/routes.py", (
            "from auth.utils import hash_password\n\n"
            "def create_user(request, ctx):\n"
            "    hashed = hash_password(request.password)\n"
            "    return ctx.db.insert(hashed)\n"
        ))
        assert r2.passed

        # Step 3: Agent writes handlers that DUPLICATE hash_password
        r3 = sentinel.check_file("api/handlers.py", (
            "def hash_password(pw):\n"
            "    import hashlib\n"
            "    return hashlib.sha256(pw.encode()).hexdigest()\n\n"
            "def handle_auth(request):\n"
            "    return hash_password(request.password)\n"
        ))
        dupes = [v for v in r3.violations if v.rule == "duplicate_function"]
        assert len(dupes) >= 1
        assert "auth/utils.py" in dupes[0].message

        # Step 4: Agent self-corrects
        r4 = sentinel.check_file("api/handlers.py", (
            "from auth.utils import hash_password\n\n"
            "def handle_auth(request):\n"
            "    return hash_password(request.password)\n"
        ))
        dupes4 = [v for v in r4.violations if v.rule == "duplicate_function"]
        assert len(dupes4) == 0

        # Verify stats
        assert sentinel.stats.files_analyzed == 4
        assert sentinel.stats.total_violations >= 1

    def test_full_audit_trail_saved(self, sentinel, tmp_path):
        """Verify the entire Sentinel session is saved as VCR frames."""
        sentinel.check_file("a.py", "def foo():\n    pass\n")
        sentinel.check_file("b.py", "from os import *\n")  # wildcard warning
        sentinel.check_file("c.py", "def bar():\n    pass\n")

        path = sentinel.save()
        from agent_vcr import VCRPlayer
        player = VCRPlayer.load(path)

        # 3 files → 3 sentinel:analysis frames
        sentinel_frames = [f for f in player.frames if f.node_name == "sentinel:analysis"]
        assert len(sentinel_frames) == 3

        # The second file should have violation recorded in frame output
        frame_b = sentinel_frames[1]
        output = frame_b.output_state
        assert output["violation_count"] >= 1
