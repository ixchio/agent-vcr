"""Unit tests for CrewAI integration — all crewai objects are fully mocked."""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import (
    VCRCrewAI,
    VCRCrewCallback,
    vcr_task,
    vcr_task_async,
)
from agent_vcr.models import FrameType


def make_recorder(tmpdir: str) -> VCRRecorder:
    recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
    recorder.start_session("test_crewai")
    return recorder


def make_mock_task(description: str = "test_task") -> MagicMock:
    task = MagicMock()
    task.description = description
    task.execute = MagicMock(return_value={"result": f"{description}_output"})
    return task


def make_mock_crew(tasks: list[MagicMock]) -> MagicMock:
    crew = MagicMock()
    crew.tasks = tasks

    def mock_kickoff(*args: Any, **kwargs: Any) -> str:
        for task in tasks:
            task.execute(context={})
        return "crew_output"

    crew.kickoff = MagicMock(side_effect=mock_kickoff)
    return crew

class TestVCRCrewAI:
    def test_kickoff_records_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            task1 = make_mock_task("research")
            task2 = make_mock_task("write")
            crew = make_mock_crew([task1, task2])

            vcr_crew = VCRCrewAI(recorder)
            result = vcr_crew.kickoff(crew)

            assert result == "crew_output"
            # 2 tasks → 2 frames recorded (task wrappers call original which records)
            assert len(recorder.frames) == 2
            node_names = [f.node_name for f in recorder.frames]
            assert "research" in node_names
            assert "write" in node_names

    def test_kickoff_with_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            task = make_mock_task("plan")
            crew = make_mock_crew([task])

            vcr_crew = VCRCrewAI(recorder)
            inputs = {"topic": "AI agents"}
            vcr_crew.kickoff(crew, inputs=inputs)

            # Crew.kickoff called with inputs
            crew.kickoff.assert_called_once_with(inputs=inputs)

    def test_kickoff_records_error_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            failing_task = make_mock_task("bad_task")
            failing_task.execute.side_effect = RuntimeError("task blew up")
            crew = make_mock_crew([failing_task])
            crew.kickoff.side_effect = RuntimeError("crew failed")

            vcr_crew = VCRCrewAI(recorder)
            with pytest.raises(RuntimeError, match="crew failed"):
                vcr_crew.kickoff(crew)

            error_frames = [f for f in recorder.frames if f.frame_type == FrameType.ERROR]
            assert len(error_frames) == 1
            assert "kickoff" in error_frames[0].node_name

    def test_wrap_task_records_individual_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            task = make_mock_task("individual")

            vcr_crew = VCRCrewAI(recorder)
            vcr_crew.wrap_task(task)

            # call the patched execute
            task.execute(context={"key": "value"})

            assert len(recorder.frames) == 1
            assert recorder.frames[0].node_name == "individual"

    def test_tasks_restored_after_kickoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            task = make_mock_task("restore_test")
            original_execute = task.execute
            crew = make_mock_crew([task])

            vcr_crew = VCRCrewAI(recorder)
            vcr_crew.kickoff(crew)

            assert task.execute is original_execute

    def test_invalid_crew_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            vcr_crew = VCRCrewAI(recorder)
            with pytest.raises(ValueError, match="tasks"):
                vcr_crew.kickoff(object())

    def test_extract_state_variants(self) -> None:
        vcr_crew = VCRCrewAI(MagicMock())

        assert vcr_crew._extract_state(None) == {}
        assert vcr_crew._extract_state({"a": 1}) == {"a": 1}

        pydantic_obj = MagicMock()
        pydantic_obj.model_dump.return_value = {"x": 2}
        assert vcr_crew._extract_state(pydantic_obj) == {"x": 2}

        plain_obj = MagicMock(spec=[])
        plain_obj.__dict__ = {"y": 3}
        result = vcr_crew._extract_state(plain_obj)
        assert "y" in result
class TestVCRCrewCallback:
    def test_on_task_end_records_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            cb = VCRCrewCallback(recorder)

            cb.on_task_start("research AI", "Researcher")
            cb.on_task_end("research AI", "Researcher", output={"summary": "done"})

            assert len(recorder.frames) == 1
            frame = recorder.frames[0]
            assert frame.node_name == "research AI"
            assert frame.input_state["agent_role"] == "Researcher"

    def test_on_agent_action_records_tool_call_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            cb = VCRCrewCallback(recorder)

            cb.on_agent_action("Researcher", "search_web", {"query": "AI trends"})

            assert len(recorder.frames) == 1
            assert recorder.frames[0].frame_type == FrameType.TOOL_CALL

    def test_on_tool_end_records_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            cb = VCRCrewCallback(recorder)

            cb.on_tool_end(
                tool_name="search_web",
                tool_input={"query": "AI"},
                tool_output={"results": ["r1", "r2"]},
                latency_ms=42.0,
            )

            frames = [f for f in recorder.frames if f.frame_type == FrameType.TOOL_CALL]
            assert len(frames) == 1

    def test_on_task_error_records_error_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            cb = VCRCrewCallback(recorder)

            err = ValueError("task failed")
            cb.on_task_error("research AI", "Researcher", error=err)

            error_frames = [f for f in recorder.frames if f.frame_type == FrameType.ERROR]
            assert len(error_frames) == 1
            assert error_frames[0].node_name == "research AI"

    def test_on_task_end_without_start_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)
            cb = VCRCrewCallback(recorder)
            # Should not raise even without a matching on_task_start
            cb.on_task_end("orphan_task", "Writer", output="some output")
            assert len(recorder.frames) == 1


# ---------------------------------------------------------------------------
# vcr_task decorator tests
# ---------------------------------------------------------------------------


class TestVcrTaskDecorator:
    def test_records_successful_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task(recorder, task_name="my_task")
            def my_task(context: dict) -> str:
                return "success"

            result = my_task({"input": "data"})

            assert result == "success"
            assert len(recorder.frames) == 1
            assert recorder.frames[0].node_name == "my_task"
            assert recorder.frames[0].output_state["result"] == "success"

    def test_uses_function_name_when_no_task_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task(recorder)
            def auto_named_task() -> str:
                return "done"

            auto_named_task()
            assert recorder.frames[0].node_name == "auto_named_task"

    def test_records_error_and_reraises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task(recorder, task_name="failing_task")
            def failing_task() -> None:
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                failing_task()

            error_frames = [f for f in recorder.frames if f.frame_type == FrameType.ERROR]
            assert len(error_frames) == 1

    def test_preserves_function_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task(recorder)
            def described_task() -> None:
                """This is a docstring."""

            assert described_task.__name__ == "described_task"
            assert described_task.__doc__ == "This is a docstring."


# ---------------------------------------------------------------------------
# vcr_task_async decorator tests
# ---------------------------------------------------------------------------


class TestVcrTaskAsyncDecorator:
    @pytest.mark.asyncio
    async def test_records_successful_async_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task_async(recorder, task_name="async_task")
            async def async_task(context: dict) -> str:
                return "async_success"

            result = await async_task({"input": "async_data"})

            assert result == "async_success"
            assert len(recorder.frames) == 1
            assert recorder.frames[0].node_name == "async_task"

    @pytest.mark.asyncio
    async def test_records_error_in_async_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = make_recorder(tmpdir)

            @vcr_task_async(recorder, task_name="async_fail")
            async def async_fail() -> None:
                raise ValueError("async boom")

            with pytest.raises(ValueError, match="async boom"):
                await async_fail()

            error_frames = [f for f in recorder.frames if f.frame_type == FrameType.ERROR]
            assert len(error_frames) == 1
