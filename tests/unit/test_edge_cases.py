"""Edge case tests for Agent VCR."""

import json
import tempfile
import threading
from pathlib import Path

import pytest

from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


class TestEmptySession:
    def test_save_empty_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session(session_id="empty")

            path = recorder.save()
            assert path.exists()

            player = VCRPlayer.load(path)
            assert len(player.frames) == 0
            assert player.session.session_id == "empty"

    def test_empty_session_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()
            path = recorder.save()

            player = VCRPlayer.load(path)
            assert player.list_nodes() == []
            assert player.get_errors() == []
            assert player.get_total_latency() == 0.0
            assert player.get_total_tokens() == 0
            assert player.get_total_cost() == 0.0

    def test_empty_session_goto_frame_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()
            path = recorder.save()

            player = VCRPlayer.load(path)
            with pytest.raises(IndexError):
                player.goto_frame(0)


class TestCorruptFile:
    def test_corrupt_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.vcr"
            with open(path, "w") as f:
                # Valid session header
                f.write(json.dumps({
                    "type": "session",
                    "data": {
                        "session_id": "corrupt-test",
                        "created_at": "2024-01-01T00:00:00+00:00",
                        "updated_at": "2024-01-01T00:00:00+00:00",
                    },
                }) + "\n")
                # Corrupt line
                f.write("not valid json\n")
                # Another corrupt line
                f.write("{incomplete\n")

            player = VCRPlayer.load(path)
            assert player.session.session_id == "corrupt-test"
            assert len(player.frames) == 0

    def test_missing_session_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_header.vcr"
            with open(path, "w") as f:
                # A valid frame line but no session header
                f.write(json.dumps({
                    "type": "frame",
                    "data": {
                        "session_id": "x",
                        "node_name": "n",
                        "input_state": {},
                        "output_state": {},
                    },
                }) + "\n")

            with pytest.raises(ValueError, match="No session header"):
                VCRPlayer.load(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            VCRPlayer.load("/nonexistent/path.vcr")

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.vcr"
            path.touch()

            with pytest.raises(ValueError, match="No session header"):
                VCRPlayer.load(path)


class TestConcurrentRecording:
    def test_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            errors = []

            def record_batch(start_idx):
                try:
                    for i in range(50):
                        recorder.record_step(
                            node_name=f"thread_{start_idx}_step_{i}",
                            input_state={"idx": start_idx + i},
                            output_state={"result": (start_idx + i) * 2},
                        )
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=record_batch, args=(i * 50,))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(recorder.get_frames()) == 200


class TestLargeState:
    def test_large_state_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            large_state = {
                "data": "x" * 100_000,
                "nested": {"deep": {"values": list(range(10000))}},
            }

            recorder.record_step("large_node", large_state, large_state)
            path = recorder.save()

            player = VCRPlayer.load(path)
            state = player.goto_frame(0)
            assert len(state["data"]) == 100_000


class TestUnicodeHandling:
    def test_unicode_in_node_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            recorder.record_step(
                node_name="处理步骤_🚀",
                input_state={"キー": "値"},
                output_state={"résultat": "succès", "emoji": "🎉"},
            )

            path = recorder.save()
            player = VCRPlayer.load(path)

            assert player.frames[0].node_name == "处理步骤_🚀"
            state = player.goto_frame(0)
            assert state["résultat"] == "succès"
            assert state["emoji"] == "🎉"


class TestGotoTime:
    def test_goto_time_with_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            recorder.record_step("n1", {}, {"v": 1})
            recorder.record_step("n2", {}, {"v": 2})
            recorder.record_step("n3", {}, {"v": 3})

            path = recorder.save()
            player = VCRPlayer.load(path)

            # Use the timestamp of the second frame
            ts = player.frames[1].timestamp.isoformat()
            state = player.goto_time(ts)
            assert state["v"] == 2

    def test_goto_time_with_datetime(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            recorder.record_step("n1", {}, {"v": 1})
            recorder.record_step("n2", {}, {"v": 2})

            path = recorder.save()
            player = VCRPlayer.load(path)

            # Use actual datetime object
            target = player.frames[0].timestamp
            state = player.goto_time(target)
            assert state["v"] == 1


class TestRecorderNoSession:
    def test_record_step_without_session_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            with pytest.raises(RuntimeError, match="No active session"):
                recorder.record_step("node", {}, {})

    def test_save_without_session_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            with pytest.raises(RuntimeError, match="No active session"):
                recorder.save()


class TestDiffMode:
    def test_diff_mode_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False, diff_mode=True)
            recorder.start_session()

            recorder.record_step("n1", {}, {"a": 1, "b": 2})
            recorder.record_step("n2", {}, {"a": 1, "b": 3, "c": 4})

            frames = recorder.get_frames()
            # First frame has no diff (no previous state)
            assert frames[0].state_diff is None
            # Second frame should have a diff
            assert frames[1].state_diff is not None

            diff = frames[1].state_diff
            ops = {d["op"] for d in diff}
            assert "replace" in ops or "add" in ops
