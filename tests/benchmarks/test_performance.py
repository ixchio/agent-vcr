"""Performance benchmarks for Agent VCR using pytest-benchmark."""

from pathlib import Path

import pytest

from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_session(tmp_path: Path):
    """A session with 1 000 frames."""
    recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    recorder.start_session("bench_small")
    for i in range(1000):
        recorder.record_step(
            node_name=f"step_{i % 10}",
            input_state={"iteration": i, "data": "x" * 100},
            output_state={"result": i * 2},
        )
    vcr_path = recorder.save()
    return vcr_path


@pytest.fixture
def large_session(tmp_path: Path):
    """A session with 10 000 frames (load & goto benchmarks)."""
    recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    recorder.start_session("bench_large")
    for i in range(10000):
        recorder.record_step(
            node_name=f"step_{i % 10}",
            input_state={"iteration": i, "data": "x" * 500},
            output_state={"result": i * 2, "data": "y" * 500},
        )
    return recorder.save()


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestPerformanceBenchmarks:
    """Benchmarks to ensure Agent VCR meets performance requirements."""

    def test_benchmark_recorder_overhead(self, benchmark, tmp_path: Path) -> None:
        """Record overhead per frame must be <5ms on average."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        recorder.start_session()
        i = 0

        def record_one_frame() -> None:
            nonlocal i
            recorder.record_step(
                node_name=f"step_{i % 10}",
                input_state={"iteration": i, "data": "x" * 100},
                output_state={"result": i * 2},
            )
            i += 1

        benchmark.pedantic(record_one_frame, rounds=1000, warmup_rounds=10)

        # Assert the mean is under 5ms
        mean_ms = benchmark.stats["mean"] * 1000
        assert mean_ms < 5.0, f"Mean recording overhead {mean_ms:.3f}ms exceeds 5ms limit"

    def test_benchmark_file_write_speed(self, benchmark, tmp_path: Path) -> None:
        """Writing 10 000 frames must sustain >1 000 frames/sec."""
        recorder = VCRRecorder(output_dir=str(tmp_path), auto_save=True, buffer_size=1000)
        recorder.start_session()

        def write_and_save() -> None:
            for i in range(10000):
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state={"iteration": i},
                    output_state={"result": i * 2},
                )
            recorder.save()

        benchmark.pedantic(write_and_save, rounds=1, warmup_rounds=0)

        elapsed_s = benchmark.stats["mean"]
        fps = 10000 / elapsed_s
        assert fps > 1000, f"Write speed {fps:.0f} frames/sec below 1 000 limit"

    def test_benchmark_load_speed(self, benchmark, large_session: Path) -> None:
        """Loading a 10 000-frame session must complete in <500ms."""

        def load() -> VCRPlayer:
            return VCRPlayer.load(large_session)

        player = benchmark(load)

        load_ms = benchmark.stats["mean"] * 1000
        assert load_ms < 500, f"Load time {load_ms:.2f}ms exceeds 500ms limit"
        assert len(player.frames) == 10000

    def test_benchmark_goto_performance(self, benchmark, large_session: Path) -> None:
        """Random-access goto_frame must average <1ms across the session."""
        player = VCRPlayer.load(large_session)
        access_indices = [0, 100, 1000, 5000, 9999]
        idx_iter = iter(access_indices * 200)  # 1 000 iterations across all indices

        def goto_one() -> dict:
            return player.goto_frame(next(idx_iter) % 10000)

        benchmark.pedantic(goto_one, rounds=1000, warmup_rounds=10)

        mean_ms = benchmark.stats["mean"] * 1000
        assert mean_ms < 1.0, f"Goto time {mean_ms:.3f}ms exceeds 1ms limit"

    def test_benchmark_file_size_diff_mode(self, tmp_path: Path) -> None:
        """Diff mode must save ≥30% storage vs full mode (not a time benchmark)."""
        state = {"base": "value", "counter": 0}

        # Diff mode
        r_diff = VCRRecorder(output_dir=str(tmp_path), auto_save=False, diff_mode=True)
        r_diff.start_session("diff")
        for i in range(1000):
            new_state = {**state, "counter": i}
            r_diff.record_step(f"step_{i % 10}", state, new_state)
            state = new_state
        size_diff = r_diff.save().stat().st_size

        # Full mode
        state = {"base": "value", "counter": 0}
        r_full = VCRRecorder(output_dir=str(tmp_path), auto_save=False, diff_mode=False)
        r_full.start_session("full")
        for i in range(1000):
            new_state = {**state, "counter": i}
            r_full.record_step(f"step_{i % 10}", state, new_state)
            state = new_state
        size_full = r_full.save().stat().st_size

        savings = (1 - size_diff / size_full) * 100
        assert savings > 30, f"Diff mode savings {savings:.1f}% below 30% target"
