"""LangGraph integration tests against the real ``langgraph`` package.

These exercise actual ``StateGraph`` builders and compiled graphs (not mocks),
so regressions like wrapping ``StateNodeSpec`` with a plain function are caught.
Skipped when langgraph is not installed.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from agent_vcr.integrations.langgraph import VCRLangGraph  # noqa: E402
from agent_vcr.models import FrameType  # noqa: E402
from agent_vcr.recorder import VCRRecorder  # noqa: E402


class State(TypedDict, total=False):
    x: int
    log: Annotated[list, operator.add]


def _recorder(tmp_path: Any) -> VCRRecorder:
    rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    rec.start_session("lg")
    return rec


def _two_node_builder() -> StateGraph:
    def inc(state: State) -> dict:
        return {"x": state["x"] + 1, "log": ["inc"]}

    def double(state: State, config: RunnableConfig) -> dict:  # config injection survives
        assert config["configurable"].get("thread_id") in (None, "t1")
        return {"x": state["x"] * 2, "log": ["double"]}

    g = StateGraph(State)
    g.add_node("inc", inc)
    g.add_node("double", double)
    g.add_edge(START, "inc")
    g.add_edge("inc", "double")
    g.add_edge("double", END)
    return g


def test_wrap_builder_before_compile(tmp_path: Any) -> None:
    rec = _recorder(tmp_path)
    g = VCRLangGraph(rec).wrap_graph(_two_node_builder())
    result = g.compile().invoke({"x": 1, "log": []})
    assert result["x"] == 4
    frames = rec.get_frames()
    assert [f.node_name for f in frames] == ["inc", "double"]
    assert frames[0].output_state["x"] == 2
    assert frames[1].input_state["x"] == 2


def test_wrap_compiled_graph(tmp_path: Any) -> None:
    rec = _recorder(tmp_path)
    app = _two_node_builder().compile()
    VCRLangGraph(rec).wrap_graph(app)
    assert app.invoke({"x": 3, "log": []})["x"] == 8
    assert [f.node_name for f in rec.get_frames()] == ["inc", "double"]


def test_wrap_is_idempotent(tmp_path: Any) -> None:
    rec = _recorder(tmp_path)
    vcr = VCRLangGraph(rec)
    g = vcr.wrap_graph(vcr.wrap_graph(_two_node_builder()))
    g.compile().invoke({"x": 1, "log": []})
    assert len(rec.get_frames()) == 2


def test_async_nodes(tmp_path: Any) -> None:
    async def anode(state: State) -> dict:
        await asyncio.sleep(0)
        return {"x": state["x"] + 10}

    g = StateGraph(State)
    g.add_node("anode", anode)
    g.add_edge(START, "anode")
    g.add_edge("anode", END)
    rec = _recorder(tmp_path)
    app = VCRLangGraph(rec).wrap_graph(g).compile()
    assert asyncio.run(app.ainvoke({"x": 1}))["x"] == 11
    frames = rec.get_frames()
    assert len(frames) == 1 and frames[0].output_state == {"x": 11}


def test_command_return_recorded(tmp_path: Any) -> None:
    def router(state: State) -> Command:
        return Command(update={"x": 99}, goto="sink")

    def sink(state: State) -> dict:
        return {"log": ["sink"]}

    g = StateGraph(State)
    g.add_node("router", router)
    g.add_node("sink", sink)
    g.add_edge(START, "router")
    g.add_edge("sink", END)
    rec = _recorder(tmp_path)
    out = VCRLangGraph(rec).wrap_graph(g).compile().invoke({"x": 0, "log": []})
    assert out["x"] == 99
    router_frame = rec.get_frames()[0]
    assert router_frame.output_state["update"] == {"x": 99}
    assert router_frame.output_state["__command__"]["goto"] == "sink"


def test_node_error_recorded_and_reraised(tmp_path: Any) -> None:
    def boom(state: State) -> dict:
        raise ValueError("kaboom")

    g = StateGraph(State)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    rec = _recorder(tmp_path)
    app = VCRLangGraph(rec).wrap_graph(g).compile()
    with pytest.raises(ValueError, match="kaboom"):
        app.invoke({"x": 1})
    frames = rec.get_frames()
    assert frames[-1].frame_type == FrameType.ERROR
    assert frames[-1].metadata.error_type == "ValueError"


def test_wrap_node_async_and_config(tmp_path: Any) -> None:
    rec = _recorder(tmp_path)
    vcr = VCRLangGraph(rec)

    async def anode(state: State, config: RunnableConfig) -> dict:
        assert "configurable" in config
        return {"x": state["x"] + 1}

    g = StateGraph(State)
    g.add_node("anode", vcr.wrap_node("anode", anode))
    g.add_edge(START, "anode")
    g.add_edge("anode", END)
    assert asyncio.run(g.compile().ainvoke({"x": 1}))["x"] == 2
    assert [f.node_name for f in rec.get_frames()] == ["anode"]


# ── LangGraphCallback as a real LangChain callback handler ───────────────


def test_callback_handler_records_nodes_llm_and_tools(tmp_path: Any) -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.tools import tool

    from agent_vcr.integrations.langgraph import LangGraphCallback

    llm = FakeListChatModel(responses=["refund approved"])

    @tool
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return f"order {order_id}: duplicate"

    def think(state: State, config: RunnableConfig) -> dict:
        reply = llm.invoke("should we refund?", config=config)
        return {"log": [str(reply.content)]}

    def act(state: State, config: RunnableConfig) -> dict:
        return {"log": [lookup.invoke({"order_id": "42"}, config=config)]}

    g = StateGraph(State)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_edge(START, "think")
    g.add_edge("think", "act")
    g.add_edge("act", END)

    rec = _recorder(tmp_path)
    out = g.compile().invoke(
        {"x": 0, "log": []}, config={"callbacks": [LangGraphCallback(rec)]}
    )
    assert out["log"] == ["refund approved", "order 42: duplicate"]

    frames = rec.get_frames()
    by_type = {f.frame_type: f for f in frames}
    node_names = [f.node_name for f in frames if f.frame_type == FrameType.NODE_EXECUTION]
    assert node_names == ["think", "act"]  # one frame per node, nested runs excluded
    assert by_type[FrameType.LLM_CALL].output_state["response"]["content"] == "refund approved"
    tool_frame = by_type[FrameType.TOOL_CALL]
    assert tool_frame.node_name == "tool:lookup"
    assert tool_frame.output_state["result"] == "order 42: duplicate"


def test_callback_handler_records_node_error(tmp_path: Any) -> None:
    from agent_vcr.integrations.langgraph import LangGraphCallback

    def boom(state: State) -> dict:
        raise RuntimeError("bad node")

    g = StateGraph(State)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    rec = _recorder(tmp_path)
    with pytest.raises(RuntimeError):
        g.compile().invoke({"x": 1}, config={"callbacks": [LangGraphCallback(rec)]})
    errors = [f for f in rec.get_frames() if f.frame_type == FrameType.ERROR]
    assert len(errors) == 1 and errors[0].node_name == "boom"
