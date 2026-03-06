"""Unit tests for VCR Player."""

import tempfile
from pathlib import Path

from agent_vcr.models import FrameMetadata, ResumeConfig
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


class TestVCRPlayer:
    def create_test_vcr(self, tmpdir: str, session_id: str = "test") -> Path:
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session(session_id=session_id)

        recorder.record_step(
            node_name="node1",
            input_state={"step": 0, "value": 0},
            output_state={"step": 1, "value": 10},
            metadata=FrameMetadata(latency_ms=100),
        )
        recorder.record_step(
            node_name="node2",
            input_state={"step": 1, "value": 10},
            output_state={"step": 2, "value": 20},
            metadata=FrameMetadata(latency_ms=200),
        )
        recorder.record_step(
            node_name="node3",
            input_state={"step": 2, "value": 20},
            output_state={"step": 3, "value": 30},
            metadata=FrameMetadata(latency_ms=300),
        )

        return recorder.save()

    def test_load_vcr_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            assert player.session.session_id == "test"
            assert len(player.frames) == 3

    def test_goto_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            state = player.goto_frame(1)
            assert state["step"] == 2
            assert state["value"] == 20

    def test_get_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            frame = player.get_frame(0)
            assert frame.node_name == "node1"

            frame = player.get_frame(2)
            assert frame.node_name == "node3"

    def test_list_nodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            nodes = player.list_nodes()
            assert nodes == ["node1", "node2", "node3"]

    def test_get_total_latency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            assert player.get_total_latency() > 0

    def test_compare_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            diff = player.compare_frames(0, 1)

            assert "modified" in diff
            assert diff["modified"]["step"]["before"] == 1
            assert diff["modified"]["step"]["after"] == 2

    def test_export_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            state = player.export_state(1)
            assert state["step"] == 2
            assert state["value"] == 20

    def test_to_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            data = player.to_dict()

            assert "session" in data
            assert "frames" in data
            assert "statistics" in data
            assert data["statistics"]["total_frames"] == 3

    def test_load_by_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.create_test_vcr(tmpdir, "my-session")
            player = VCRPlayer.load_by_id("my-session", vcr_dir=tmpdir)

            assert player.session.session_id == "my-session"

    def test_get_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            recorder.record_step("ok_node", {}, {}, metadata=FrameMetadata())
            recorder.record_error(
                "error_node",
                {},
                ValueError("test error"),
                latency_ms=50,
            )

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            errors = player.get_errors()
            assert len(errors) == 1
            assert errors[0].node_name == "error_node"


class TestVCRPlayerResume:
    def create_test_vcr(self, tmpdir: str) -> Path:
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session(session_id="original")

        recorder.record_step(
            node_name="add",
            input_state={"a": 2, "b": 3, "result": None},
            output_state={"a": 2, "b": 3, "result": 5},
        )
        recorder.record_step(
            node_name="multiply",
            input_state={"a": 2, "b": 3, "result": 5},
            output_state={"a": 2, "b": 3, "result": 30},
        )

        return recorder.save()

    def test_resume_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            def mock_agent(state):
                return {**state, "result": state.get("a", 0) * state.get("b", 0) * 10}

            new_recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            new_session_id = player.resume(
                agent_callable=mock_agent,
                config=ResumeConfig(from_frame=0),
                recorder=new_recorder,
            )

            assert new_session_id is not None

            forked_player = VCRPlayer.load_by_id(new_session_id, vcr_dir=tmpdir)
            assert forked_player.session.parent_session_id == "original"
            assert forked_player.session.forked_from_frame == 0

    def test_resume_with_state_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vcr_path = self.create_test_vcr(tmpdir)
            player = VCRPlayer.load(vcr_path)

            state = player.goto_frame(0)
            assert state["a"] == 2

            def mock_agent(state):
                return state

            new_recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            new_session_id = player.resume(
                agent_callable=mock_agent,
                config=ResumeConfig(
                    from_frame=0,
                    state_overrides={"a": 10, "b": 20},
                ),
                recorder=new_recorder,
            )

            forked_player = VCRPlayer.load_by_id(new_session_id, vcr_dir=tmpdir)
            final_state = forked_player.goto_frame(0)

            assert final_state["a"] == 10
            assert final_state["b"] == 20
