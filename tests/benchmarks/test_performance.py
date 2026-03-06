"""Performance benchmarks for Agent VCR."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


class TestPerformanceBenchmarks:
    """Benchmarks to ensure VCR meets performance requirements."""

    def test_benchmark_recorder_overhead(self):
        """Measure recording overhead per frame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            times = []
            for i in range(1000):
                state = {"iteration": i, "data": "x" * 100}

                start = time.perf_counter()
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state=state,
                    output_state=state,
                )
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            p99_time = sorted(times)[int(len(times) * 0.99)]

            print(f"\nRecording Overhead:")
            print(f"  Average: {avg_time:.3f}ms")
            print(f"  P99: {p99_time:.3f}ms")

            # Assert performance requirements
            assert avg_time < 5.0, f"Average overhead {avg_time:.3f}ms exceeds 5ms limit"
            assert p99_time < 10.0, f"P99 overhead {p99_time:.3f}ms exceeds 10ms limit"

    def test_benchmark_file_write_speed(self):
        """Measure file write throughput."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=True, buffer_size=1000)
            recorder.start_session()

            # Generate 10k frames
            start = time.perf_counter()
            for i in range(10000):
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state={"iteration": i},
                    output_state={"result": i * 2},
                )

            save_start = time.perf_counter()
            vcr_path = recorder.save()
            save_time = (time.perf_counter() - save_start) * 1000
            total_time = (time.perf_counter() - start) * 1000

            frames_per_sec = 10000 / (total_time / 1000)

            print(f"\nFile Write Speed:")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Save time: {save_time:.2f}ms")
            print(f"  Throughput: {frames_per_sec:.0f} frames/sec")

            assert frames_per_sec > 1000, f"Write speed {frames_per_sec:.0f} frames/sec below 1000 limit"

    def test_benchmark_load_speed(self):
        """Measure session load time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large session
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("large_session")

            for i in range(10000):
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state={"iteration": i, "data": "x" * 500},
                    output_state={"result": i * 2, "data": "y" * 500},
                )

            vcr_path = recorder.save()

            # Measure load time
            start = time.perf_counter()
            player = VCRPlayer.load(vcr_path)
            load_time = (time.perf_counter() - start) * 1000

            print(f"\nLoad Speed:")
            print(f"  Frames: {len(player.frames)}")
            print(f"  Load time: {load_time:.2f}ms")
            print(f"  Per frame: {load_time / len(player.frames):.3f}ms")

            assert load_time < 500, f"Load time {load_time:.2f}ms exceeds 500ms limit"

    @pytest.mark.skipif(
        not pytest.importorskip("psutil", reason="psutil not installed"),
        reason="psutil not installed",
    )
    def test_benchmark_memory_footprint(self):
        """Measure memory usage during recording."""
        import psutil

        process = psutil.Process(os.getpid())

        with tempfile.TemporaryDirectory() as tmpdir:
            initial_mem = process.memory_info().rss / 1024 / 1024  # MB

            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            # Record many frames
            for i in range(5000):
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state={"iteration": i, "data": "x" * 1000},
                    output_state={"result": i * 2, "data": "y" * 1000},
                )

            final_mem = process.memory_info().rss / 1024 / 1024  # MB
            mem_increase = final_mem - initial_mem

            print(f"\nMemory Footprint:")
            print(f"  Initial: {initial_mem:.2f} MB")
            print(f"  Final: {final_mem:.2f} MB")
            print(f"  Increase: {mem_increase:.2f} MB")
            print(f"  Per frame: {mem_increase / 5000 * 1024:.3f} KB")

            assert mem_increase < 100, f"Memory increase {mem_increase:.2f}MB exceeds 100MB limit"

    def test_benchmark_goto_performance(self):
        """Measure time-travel (goto) performance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            for i in range(10000):
                recorder.record_step(
                    node_name=f"step_{i % 10}",
                    input_state={"iteration": i},
                    output_state={"result": i * 2},
                )

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            # Benchmark random access
            times = []
            for idx in [0, 100, 1000, 5000, 9999]:
                start = time.perf_counter()
                state = player.goto_frame(idx)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_goto = sum(times) / len(times)

            print(f"\nGoto Performance:")
            print(f"  Average: {avg_goto:.3f}ms")
            print(f"  Min: {min(times):.3f}ms")
            print(f"  Max: {max(times):.3f}ms")

            assert avg_goto < 1.0, f"Goto time {avg_goto:.3f}ms exceeds 1ms limit"

    def test_benchmark_file_size(self):
        """Measure storage efficiency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with diff mode
            recorder_diff = VCRRecorder(output_dir=tmpdir, auto_save=False, diff_mode=True)
            recorder_diff.start_session("diff_mode")

            state = {"base": "value", "counter": 0}
            for i in range(1000):
                new_state = {**state, "counter": i}
                recorder_diff.record_step(
                    node_name=f"step_{i % 10}",
                    input_state=state,
                    output_state=new_state,
                )
                state = new_state

            vcr_path_diff = recorder_diff.save()
            size_diff = vcr_path_diff.stat().st_size

            # Test without diff mode
            recorder_full = VCRRecorder(output_dir=tmpdir, auto_save=False, diff_mode=False)
            recorder_full.start_session("full_mode")

            state = {"base": "value", "counter": 0}
            for i in range(1000):
                new_state = {**state, "counter": i}
                recorder_full.record_step(
                    node_name=f"step_{i % 10}",
                    input_state=state,
                    output_state=new_state,
                )
                state = new_state

            vcr_path_full = recorder_full.save()
            size_full = vcr_path_full.stat().st_size

            savings = (1 - size_diff / size_full) * 100

            print(f"\nStorage Efficiency:")
            print(f"  Full mode: {size_full / 1024:.2f} KB")
            print(f"  Diff mode: {size_diff / 1024:.2f} KB")
            print(f"  Savings: {savings:.1f}%")

            assert savings > 30, f"Diff mode savings {savings:.1f}% below 30% target"


if __name__ == "__main__":
    bench = TestPerformanceBenchmarks()
    bench.test_benchmark_recorder_overhead()
    bench.test_benchmark_file_write_speed()
    bench.test_benchmark_load_speed()
    bench.test_benchmark_memory_footprint()
    bench.test_benchmark_goto_performance()
    bench.test_benchmark_file_size()
