"""Unit tests for LangGraph integration."""

import tempfile
from unittest.mock import MagicMock

import pytest

from agent_vcr.integrations.langgraph import (
    LangGraphCallback,
    VCRLangGraph,
    vcr_record,
)
from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.recorder import VCRRecorder


class TestVCRLangGraph:
    def test_wrap_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            wrapper = VCRLangGraph(recorder)

            def my_node(state):
                return {"result": state["value"] * 2}

            wrapped = wrapper.wrap_node("my_node", my_node)
            result = wrapped({"value": 5})

            assert result == {"result": 10}
            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].node_name == "my_node"

    def test_wrap_node_records_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            wrapper = VCRLangGraph(recorder)

            def failing_node(state):
                raise ValueError("node failed")

            wrapped = wrapper.wrap_node("failing_node", failing_node)

            with pytest.raises(ValueError, match="node failed"):
                wrapped({"v": 1})

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].frame_type == FrameType.ERROR
            assert frames[0].metadata.error_type == "ValueError"

    def test_wrap_graph_with_mock(self):
        pytest.importorskip("langgraph")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            wrapper = VCRLangGraph(recorder)

            # Mock a LangGraph-like StateGraph
            mock_graph = MagicMock()
            mock_graph.nodes = {
                "planner": lambda state: {"plan": "do stuff"},
                "executor": lambda state: {"result": "done"},
            }

            wrapped_graph = wrapper.wrap_graph(mock_graph)

            # Call wrapped nodes
            wrapped_graph.nodes["planner"]({"query": "test"})
            wrapped_graph.nodes["executor"]({"plan": "do stuff"})

            frames = recorder.get_frames()
            assert len(frames) == 2
            assert frames[0].node_name == "planner"
            assert frames[1].node_name == "executor"

    def test_wrap_graph_requires_nodes_attribute(self):
        pytest.importorskip("langgraph")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()
            wrapper = VCRLangGraph(recorder)

            with pytest.raises(ValueError, match="'nodes' attribute"):
                wrapper.wrap_graph(object())


class TestLangGraphCallback:
    def test_on_node_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            callback = LangGraphCallback(recorder)
            callback.on_node_end(
                node_name="test_node",
                state={"input": "data"},
                output={"output": "result"},
                latency_ms=100.5,
            )

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].node_name == "test_node"

    def test_on_llm_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            callback = LangGraphCallback(recorder)
            callback.on_llm_end(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                response={"content": "hello"},
                tokens_input=10,
                tokens_output=5,
                latency_ms=500.0,
            )

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].frame_type == FrameType.LLM_CALL

    def test_on_tool_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            callback = LangGraphCallback(recorder)
            callback.on_tool_end(
                tool_name="calculator",
                tool_input={"expr": "1+1"},
                tool_output=2,
                latency_ms=50.0,
            )

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].frame_type == FrameType.TOOL_CALL

    def test_extract_state_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            callback = LangGraphCallback(recorder)

            # None
            assert callback._extract_state(None) == {}

            # Dict
            assert callback._extract_state({"key": "value"}) == {"key": "value"}

            # Object with __dict__
            class MyState:
                def __init__(self):
                    self.x = 1

            state = MyState()
            assert callback._extract_state(state)["x"] == 1

            # Fallback
            result = callback._extract_state(42)
            assert result == {"_raw": "42"}


class TestVCRRecordDecorator:
    def test_basic_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            @vcr_record(recorder, node_name="my_func")
            def my_func(x, y):
                return x + y

            result = my_func(3, 4)
            assert result == 7

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].node_name == "my_func"

    def test_preserves_function_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            @vcr_record(recorder)
            def my_custom_function():
                """My docstring."""
                return 42

            assert my_custom_function.__name__ == "my_custom_function"
            assert my_custom_function.__doc__ == "My docstring."

    def test_auto_node_name_from_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            @vcr_record(recorder)
            def compute_stuff():
                return 42

            compute_stuff()
            frames = recorder.get_frames()
            assert frames[0].node_name == "compute_stuff"

    def test_error_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session()

            @vcr_record(recorder, node_name="failing")
            def failing_func():
                raise RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                failing_func()

            frames = recorder.get_frames()
            assert len(frames) == 1
            assert frames[0].frame_type == FrameType.ERROR
            assert frames[0].metadata.error_message == "boom"
