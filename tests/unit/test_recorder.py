"""Unit tests for VCR Recorder."""

import tempfile

from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.recorder import VCRRecorder


class TestVCRRecorder:
    def test_start_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            session = recorder.start_session(session_id="test-session")

            assert session.session_id == "test-session"
            assert recorder.get_session() == session

    def test_record_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            frame = recorder.record_step(
                node_name="test_node",
                input_state={"input": "data"},
                output_state={"output": "result"},
            )

            assert frame.node_name == "test_node"
            assert frame.input_state == {"input": "data"}
            assert frame.output_state == {"output": "result"}
            assert len(recorder.get_frames()) == 1

    def test_record_step_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            metadata = FrameMetadata(
                latency_ms=100,
                tokens_used=50,
                cost_usd=0.001,
            )

            frame = recorder.record_step(
                node_name="test_node",
                input_state={},
                output_state={},
                metadata=metadata,
            )

            assert frame.metadata.latency_ms > 0
            assert frame.metadata.tokens_used == 50

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session(session_id="test-save")

            recorder.record_step("node1", {"a": 1}, {"b": 2})
            recorder.record_step("node2", {"b": 2}, {"c": 3})

            path = recorder.save()

            assert path.exists()
            assert path.name == "test-save.vcr"

            with open(path) as f:
                lines = f.readlines()
                assert len(lines) == 3  # header + 2 frames

    def test_fork_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session(session_id="parent")

            recorder.record_step("node1", {"a": 1}, {"b": 2})
            recorder.record_step("node2", {"b": 2}, {"c": 3})

            forked = recorder.fork(from_frame=0, new_session_id="child")

            assert forked.get_session().parent_session_id == "parent"
            assert forked.get_session().forked_from_frame == 0

    def test_record_llm_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            frame = recorder.record_llm_call(
                model="gpt-4",
                messages=[{"role": "user", "content": "hello"}],
                response={"content": "hi"},
                tokens_input=10,
                tokens_output=5,
                latency_ms=500,
                cost_usd=0.0001,
            )

            assert frame.frame_type == FrameType.LLM_CALL
            assert frame.node_name == "llm:gpt-4"
            assert frame.metadata.tokens_used == 15

    def test_record_tool_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            frame = recorder.record_tool_call(
                tool_name="calculator",
                tool_input={"expression": "1+1"},
                tool_output=2,
                latency_ms=100,
            )

            assert frame.frame_type == FrameType.TOOL_CALL
            assert frame.node_name == "tool:calculator"

    def test_record_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            error = ValueError("test error")
            frame = recorder.record_error(
                node_name="failing_node",
                input_state={},
                error=error,
                latency_ms=50,
            )

            assert frame.frame_type == FrameType.ERROR
            assert frame.metadata.error_type == "ValueError"
            assert frame.metadata.error_message == "test error"

    def test_session_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            session = recorder.start_session()

            recorder.record_step(
                "node1",
                {},
                {},
                metadata=FrameMetadata(tokens_used=100, cost_usd=0.001),
            )
            recorder.record_step(
                "node2",
                {},
                {},
                metadata=FrameMetadata(tokens_used=200, cost_usd=0.002),
            )

            assert session.total_tokens == 300
            assert session.total_cost_usd == 0.003
