"""Async unit tests for AsyncVCRRecorder."""

import tempfile

import pytest

from agent_vcr.async_recorder import AsyncVCRRecorder
from agent_vcr.async_player import AsyncVCRPlayer
from agent_vcr.models import FrameMetadata, FrameType


@pytest.mark.asyncio
class TestAsyncVCRRecorder:
    async def test_start_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            session = await recorder.start_session(session_id="async-test")

            assert session.session_id == "async-test"
            assert recorder.get_session() is not None

    async def test_record_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            frame = await recorder.record_step(
                node_name="test_node",
                input_state={"x": 1},
                output_state={"x": 2},
            )

            assert frame.node_name == "test_node"
            assert len(recorder.get_frames()) == 1

    async def test_record_llm_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            frame = await recorder.record_llm_call(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                response="hello",
                tokens_input=10,
                tokens_output=5,
                latency_ms=100,
            )

            assert frame.frame_type == FrameType.LLM_CALL
            assert frame.metadata.model == "gpt-4"

    async def test_record_tool_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            frame = await recorder.record_tool_call(
                tool_name="search",
                tool_input={"query": "test"},
                tool_output=["result1"],
                latency_ms=50,
            )

            assert frame.frame_type == FrameType.TOOL_CALL

    async def test_record_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            frame = await recorder.record_error(
                node_name="err_node",
                input_state={"x": 1},
                error=ValueError("test error"),
                latency_ms=10,
            )

            assert frame.frame_type == FrameType.ERROR
            assert frame.metadata.error_message == "test error"

    async def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session(session_id="save-test")

            await recorder.record_step("n1", {"a": 1}, {"b": 2})
            await recorder.record_step("n2", {"b": 2}, {"c": 3})

            path = await recorder.save()
            assert path.exists()

            # Load with AsyncVCRPlayer
            player = await AsyncVCRPlayer.load(path)
            assert player.session.session_id == "save-test"
            assert len(player.frames) == 2

    async def test_fork(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session(session_id="parent")

            await recorder.record_step("n1", {}, {"v": 1})

            forked = await recorder.fork(from_frame=0)
            session = forked.get_session()
            assert session.parent_session_id == "parent"
            assert session.forked_from_frame == 0

    async def test_no_session_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)

            with pytest.raises(RuntimeError, match="No active session"):
                await recorder.record_step("node", {}, {})

            with pytest.raises(RuntimeError, match="No active session"):
                await recorder.save()

    async def test_diff_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False, diff_mode=True)
            await recorder.start_session()

            await recorder.record_step("n1", {}, {"a": 1, "b": 2})
            await recorder.record_step("n2", {}, {"a": 1, "b": 3, "c": 4})

            frames = recorder.get_frames()
            assert frames[0].state_diff is None
            assert frames[1].state_diff is not None


@pytest.mark.asyncio
class TestAsyncVCRPlayer:
    async def test_load_and_navigate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            await recorder.record_step("n1", {}, {"v": 1})
            await recorder.record_step("n2", {}, {"v": 2})
            await recorder.record_step("n3", {}, {"v": 3})

            path = await recorder.save()

            player = await AsyncVCRPlayer.load(path)
            assert len(player.frames) == 3

            state = player.goto_frame(1)
            assert state["v"] == 2

    async def test_list_nodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            await recorder.record_step("a", {}, {})
            await recorder.record_step("b", {}, {})
            await recorder.record_step("a", {}, {})

            path = await recorder.save()
            player = await AsyncVCRPlayer.load(path)

            assert player.list_nodes() == ["a", "b"]

    async def test_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            await recorder.record_step(
                "n1", {}, {},
                metadata=FrameMetadata(latency_ms=100, tokens_used=50, cost_usd=0.01),
            )
            await recorder.record_step(
                "n2", {}, {},
                metadata=FrameMetadata(latency_ms=200, tokens_used=100, cost_usd=0.02),
            )

            path = await recorder.save()
            player = await AsyncVCRPlayer.load(path)

            assert player.get_total_latency() == 300.0
            assert player.get_total_tokens() == 150
            assert player.get_total_cost() == pytest.approx(0.03)

    async def test_goto_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AsyncVCRRecorder(output_dir=tmpdir, auto_save=False)
            await recorder.start_session()

            await recorder.record_step("n1", {}, {"v": 1})
            await recorder.record_step("n2", {}, {"v": 2})

            path = await recorder.save()
            player = await AsyncVCRPlayer.load(path)

            ts = player.frames[1].timestamp
            state = player.goto_time(ts)
            assert state["v"] == 2
