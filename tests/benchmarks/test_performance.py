"""Performance benchmarks for Agent VCR.

Every benchmark has an assertion threshold. If it regresses, CI fails.
Run locally:  python -m pytest tests/benchmarks/ -v --benchmark-only --no-cov
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from agent_vcr.golden_cache import GoldenRunCache
from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REALISTIC_STATE = {
    "query": "Build a REST API with JWT auth, rate limiting, and Postgres",
    "plan": {
        "steps": [f"Step {i}: do something important" for i in range(20)],
        "reasoning": "We need to scaffold the project first." * 10,
    },
    "context": {
        "files_written": [f"src/module_{i}.py" for i in range(15)],
        "dependencies": ["fastapi", "sqlalchemy", "pyjwt", "redis", "pydantic"],
    },
    "messages": [
        {"role": "user", "content": "Build a REST API" + " with auth" * 50},
        {"role": "assistant", "content": "Here is the plan:\n" + "step detail\n" * 100},
    ],
    "metadata": {"tokens": 4200, "model": "gpt-4o", "cost": 0.0126},
}


def _build_session(tmp_path: Path, n_frames: int, state_size: str = "small") -> Path:
    """Helper to build a session with n_frames and specified payload size."""
    recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    recorder.start_session(f"bench_{n_frames}")
    for i in range(n_frames):
        if state_size == "realistic":
            inp = {**REALISTIC_STATE, "iteration": i}
            out = {**REALISTIC_STATE, "result": f"output_{i}", "iteration": i}
        elif state_size == "large":
            inp = {"iteration": i, "data": "x" * 2000, "nested": {"a": list(range(100))}}
            out = {"result": i * 2, "data": "y" * 2000, "nested": {"b": list(range(100))}}
        else:
            inp = {"iteration": i, "data": "x" * 100}
            out = {"result": i * 2}
        meta = FrameMetadata(
            latency_ms=float(50 + i % 200),
            tokens_used=100 + i % 500,
            cost_usd=0.003 + (i % 50) * 0.0001,
            model="gpt-4o",
        )
        recorder.record_step(f"step_{i % 10}", inp, out, metadata=meta)
    return recorder.save()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_session(tmp_path: Path) -> Path:
    return _build_session(tmp_path, 1000, "small")


@pytest.fixture
def large_session(tmp_path: Path) -> Path:
    return _build_session(tmp_path, 10000, "small")


@pytest.fixture
def realistic_session(tmp_path: Path) -> Path:
    return _build_session(tmp_path, 200, "realistic")


@pytest.fixture
def large_payload_session(tmp_path: Path) -> Path:
    return _build_session(tmp_path, 1000, "large")


# ---------------------------------------------------------------------------
# Core Recording & Playback
# ---------------------------------------------------------------------------


class TestCorePerformance:
    """Core recorder/player benchmarks with hard limits."""

    def test_record_frame_overhead(self, benchmark, tmp_path: Path) -> None:
        """Single frame record must average <5ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session()
        i = 0

        def record_one() -> None:
            nonlocal i
            recorder.record_step(
                f"step_{i % 10}",
                {"iteration": i, "data": "x" * 100},
                {"result": i * 2},
            )
            i += 1

        benchmark.pedantic(record_one, rounds=1000, warmup_rounds=10)
        assert benchmark.stats["mean"] * 1000 < 5.0

    def test_record_realistic_payload(self, benchmark, tmp_path: Path) -> None:
        """Recording a realistic agent state (~2KB) must average <10ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session()
        i = 0

        def record_realistic() -> None:
            nonlocal i
            recorder.record_step(
                f"step_{i % 10}",
                {**REALISTIC_STATE, "iteration": i},
                {**REALISTIC_STATE, "result": f"out_{i}"},
            )
            i += 1

        benchmark.pedantic(record_realistic, rounds=500, warmup_rounds=5)
        assert benchmark.stats["mean"] * 1000 < 10.0

    def test_record_large_payload(self, benchmark, tmp_path: Path) -> None:
        """Recording a large state (~5KB) must average <15ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session()
        big_state = {"data": "x" * 5000, "list": list(range(500))}
        i = 0

        def record_large() -> None:
            nonlocal i
            recorder.record_step(
                f"step_{i % 10}",
                {**big_state, "i": i},
                {**big_state, "result": i},
            )
            i += 1

        benchmark.pedantic(record_large, rounds=200, warmup_rounds=5)
        assert benchmark.stats["mean"] * 1000 < 15.0

    def test_write_throughput(self, benchmark, tmp_path: Path) -> None:
        """10K frames must sustain >1000 frames/sec to disk."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=True, buffer_size=1000)
        recorder.start_session()

        def write_10k() -> None:
            for i in range(10000):
                recorder.record_step(
                    f"step_{i % 10}",
                    {"iteration": i},
                    {"result": i * 2},
                )
            recorder.save()

        benchmark.pedantic(write_10k, rounds=1, warmup_rounds=0)
        fps = 10000 / benchmark.stats["mean"]
        assert fps > 1000, f"Write speed {fps:.0f} fps below 1000"

    def test_load_10k_session(self, benchmark, large_session: Path) -> None:
        """Loading 10K frames from JSONL must complete in <500ms."""
        player = benchmark(lambda: VCRPlayer.load(large_session))
        assert benchmark.stats["mean"] * 1000 < 500
        assert len(player.frames) == 10000

    def test_load_realistic_session(self, benchmark, realistic_session: Path) -> None:
        """Loading 200 realistic frames must complete in <100ms."""
        player = benchmark(lambda: VCRPlayer.load(realistic_session))
        assert benchmark.stats["mean"] * 1000 < 100
        assert len(player.frames) == 200

    def test_goto_frame(self, benchmark, large_session: Path) -> None:
        """Random-access goto_frame must average <1ms."""
        player = VCRPlayer.load(large_session)
        indices = [0, 100, 1000, 5000, 9999]
        idx_iter = itertools.cycle(indices)

        benchmark.pedantic(
            lambda: player.goto_frame(next(idx_iter) % 10000),
            rounds=1000,
            warmup_rounds=10,
        )
        assert benchmark.stats["mean"] * 1000 < 1.0


# ---------------------------------------------------------------------------
# Time-Travel Operations
# ---------------------------------------------------------------------------


class TestTimeTravelPerformance:
    """Benchmarks for time-travel features: compare, fork, resume."""

    def test_compare_frames(self, benchmark, large_session: Path) -> None:
        """Comparing two frames must average <5ms."""
        player = VCRPlayer.load(large_session)

        pairs = [(0, 100), (500, 5000), (0, 9999), (4999, 5000)]
        pair_iter = itertools.cycle(pairs)

        def compare_one() -> dict:
            a, b = next(pair_iter)
            return player.compare_frames(a, b)

        benchmark.pedantic(compare_one, rounds=200, warmup_rounds=5)
        assert benchmark.stats["mean"] * 1000 < 5.0

    def test_fork_session(self, benchmark, tmp_path: Path) -> None:
        """Forking from a frame must complete in <10ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session("parent")
        for i in range(100):
            recorder.record_step(
                f"step_{i}",
                {"i": i},
                {"result": i * 2},
            )
        recorder.save()

        idx_iter = itertools.cycle([0, 25, 50, 75, 99])

        def fork_one() -> VCRRecorder:
            return recorder.fork(
                from_frame=next(idx_iter),
                state_overrides={"fix": "patched"},
            )

        benchmark.pedantic(fork_one, rounds=100, warmup_rounds=5)
        assert benchmark.stats["mean"] * 1000 < 10.0

    def test_get_errors(self, benchmark, tmp_path: Path) -> None:
        """Scanning 1K frames for errors must complete in <5ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session("error_scan")
        for i in range(1000):
            if i % 100 == 0:
                recorder.record_error(
                    f"step_{i}", {"i": i}, ValueError("test"), latency_ms=1.0,
                )
            else:
                recorder.record_step(f"step_{i}", {"i": i}, {"r": i})
        path = recorder.save()
        player = VCRPlayer.load(path)

        benchmark.pedantic(player.get_errors, rounds=500, warmup_rounds=10)
        assert benchmark.stats["mean"] * 1000 < 5.0


# ---------------------------------------------------------------------------
# Ghost Replay (Golden Cache)
# ---------------------------------------------------------------------------


class TestGhostReplayPerformance:
    """Benchmarks for Ghost Replay — the zero-cost replay system."""

    def test_ghost_save(self, benchmark, tmp_path: Path) -> None:
        """Saving a 50-frame golden run must complete in <50ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path / "vcr"), auto_save=False)
        recorder.start_session("golden_src")
        for i in range(50):
            recorder.record_step(
                f"step_{i}",
                {**REALISTIC_STATE, "i": i},
                {**REALISTIC_STATE, "result": i},
                metadata=FrameMetadata(tokens_used=200, cost_usd=0.006, latency_ms=80.0),
            )
        recorder.save()

        task_i = itertools.count()

        def save_golden() -> str:
            cache = GoldenRunCache(cache_dir=str(tmp_path / f"golden_{next(task_i)}"))
            return cache.save_golden_run(f"Build REST API variant {next(task_i)}", recorder)

        benchmark.pedantic(save_golden, rounds=20, warmup_rounds=2)
        assert benchmark.stats["mean"] * 1000 < 50.0

    def test_ghost_replay(self, benchmark, tmp_path: Path) -> None:
        """Replaying a 50-frame golden run must complete in <20ms (zero LLM)."""
        recorder = VCRRecorder(output_dir=str(tmp_path / "vcr"), auto_save=False)
        recorder.start_session("golden_src")
        for i in range(50):
            recorder.record_step(
                f"step_{i}",
                {"i": i, "data": "x" * 200},
                {"result": i, "data": "y" * 200},
                metadata=FrameMetadata(tokens_used=200, cost_usd=0.006, latency_ms=80.0),
            )
        recorder.save()

        task = "Build REST API with JWT"
        cache = GoldenRunCache(cache_dir=str(tmp_path / "golden"))
        cache.save_golden_run(task, recorder)

        replay_recorder = VCRRecorder(output_dir=str(tmp_path / "replay"), auto_save=False)

        def replay() -> tuple:
            return cache.replay(task, recorder=replay_recorder)

        result = benchmark.pedantic(replay, rounds=20, warmup_rounds=2)
        assert benchmark.stats["mean"] * 1000 < 20.0

        outputs, ledger = result
        assert ledger.steps_replayed == 50
        assert ledger.replay_tokens == 0
        assert ledger.savings_percent == 100.0

    def test_ghost_fingerprint_lookup(self, benchmark, tmp_path: Path) -> None:
        """Cache lookup for an existing task must be <1ms."""
        recorder = VCRRecorder(output_dir=str(tmp_path / "vcr"), auto_save=False)
        recorder.start_session("fp_test")
        recorder.record_step("s", {"a": 1}, {"b": 2})
        recorder.save()

        cache = GoldenRunCache(cache_dir=str(tmp_path / "golden"))
        for i in range(100):
            cache.save_golden_run(f"Task variant {i}", recorder)

        def lookup() -> bool:
            return cache.has_golden_run("Task variant 42")

        benchmark.pedantic(lookup, rounds=1000, warmup_rounds=10)
        assert benchmark.stats["mean"] * 1000 < 1.0


# ---------------------------------------------------------------------------
# Storage Efficiency
# ---------------------------------------------------------------------------


class TestStorageEfficiency:
    """Non-timing benchmarks for storage format quality."""

    def test_diff_mode_savings(self, tmp_path: Path) -> None:
        """Diff mode must save ≥30% storage vs full mode."""
        base_padding = "x" * 1024
        state = {"base": base_padding, "counter": 0}

        r_diff = VCRRecorder(output_dir=str(tmp_path), auto_save=False, diff_mode=True)
        r_diff.start_session("diff")
        for i in range(1000):
            new_state = {**state, "counter": i}
            r_diff.record_step(f"step_{i % 10}", state, new_state)
            state = new_state
        size_diff = r_diff.save().stat().st_size

        state = {"base": base_padding, "counter": 0}
        r_full = VCRRecorder(output_dir=str(tmp_path), auto_save=False, diff_mode=False)
        r_full.start_session("full")
        for i in range(1000):
            new_state = {**state, "counter": i}
            r_full.record_step(f"step_{i % 10}", state, new_state)
            state = new_state
        size_full = r_full.save().stat().st_size

        savings = (1 - size_diff / size_full) * 100
        assert savings > 30, f"Diff mode savings {savings:.1f}% below 30%"

    def test_realistic_session_file_size(self, tmp_path: Path) -> None:
        """A 100-frame realistic session should be <1MB on disk."""
        path = _build_session(tmp_path, 100, "realistic")
        size_kb = path.stat().st_size / 1024
        assert size_kb < 1024, f"Session file {size_kb:.0f}KB exceeds 1MB"

    def test_context_manager_auto_save(self, tmp_path: Path) -> None:
        """Context manager must auto-save even on exception."""
        try:
            with VCRRecorder(output_dir=str(tmp_path)) as r:
                r.start_session("ctx_test")
                r.record_step("s", {"a": 1}, {"b": 2})
                raise RuntimeError("simulated crash")
        except RuntimeError:
            pass

        saved = list(tmp_path.glob("*.vcr"))
        assert len(saved) == 1, "Context manager failed to auto-save"
