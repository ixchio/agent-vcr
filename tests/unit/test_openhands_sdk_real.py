"""VCRRuntime and Sentinel on the real OpenHands SDK 1.x, without model calls.

Drives the code paths ``Agent.step`` uses (emit ActionEvent → execute the tool
→ ObservationEvent) with hand-built events and the real ``file_editor`` and
``terminal`` tools. Skipped unless ``openhands-sdk`` and ``openhands-tools``
are installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openhands.sdk")
pytest.importorskip("openhands.tools.file_editor")
pytest.importorskip("openhands.tools.terminal")

from openhands.sdk import LLM, Agent, Conversation, Tool  # noqa: E402
from openhands.sdk.event import ActionEvent  # noqa: E402
from openhands.sdk.llm import MessageToolCall, TextContent  # noqa: E402
from openhands.tools.file_editor import FileEditorTool  # noqa: E402
from openhands.tools.file_editor.definition import FileEditorAction  # noqa: E402
from openhands.tools.terminal import TerminalTool  # noqa: E402
from openhands.tools.terminal.definition import TerminalAction  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from agent_vcr import VCRPlayer, VCRRecorder  # noqa: E402
from agent_vcr.integrations.openhands_hook import VCRRuntime  # noqa: E402
from openhands_sentinel.sentinel import Sentinel  # noqa: E402

SYNTAX_ERROR = "def broken(:\n    pass\n"


def _event(tool: str, action: Any, i: int, response: str = "r1") -> ActionEvent:
    call = MessageToolCall(id=f"call_{i}", name=tool, arguments=json.dumps(action.model_dump()),
                           origin="completion")
    return ActionEvent(action=action, thought=[TextContent(text="")], tool_name=tool,
                       tool_call_id=call.id, tool_call=call, llm_response_id=response)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _conversation(ws: Path, callbacks: list[Any]) -> tuple[Agent, Any]:
    agent = Agent(llm=LLM(model="gpt-4o-mini", api_key=SecretStr("unused"), usage_id="t"),
                  tools=[Tool(name=FileEditorTool.name), Tool(name=TerminalTool.name)])
    conv = Conversation(agent=agent, workspace=str(ws), callbacks=callbacks, visualizer=None)
    return agent, conv


def _step(agent: Agent, conv: Any, *events: ActionEvent) -> list[Any]:
    """Emit all tool calls of one LLM response, then execute them in order."""
    for event in events:
        conv._on_event(event)
    return [agent._execute_action_event(conv, e, on_event=conv._on_event) for e in events]


def test_vcr_runtime_records_real_tool_calls(workspace: Path, tmp_path: Path) -> None:
    vcr = VCRRuntime(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")), session_id="sdk")
    agent, conv = _conversation(workspace, [vcr])
    target = str(workspace / "app.py")
    try:
        with conv.state:
            _step(agent, conv,
                  _event("file_editor", FileEditorAction(command="create", path=target,
                                                         file_text="x = 1\n"), 1),
                  _event("terminal", TerminalAction(command="sh -c 'echo out; exit 3'"), 2))
            _step(agent, conv, _event("file_editor", FileEditorAction(
                command="str_replace", path=target, old_str="nope", new_str="y"), 3))
    finally:
        conv.close()
    player = VCRPlayer.load(vcr.save())

    assert [f.node_name for f in player.frames] == [
        "openhands:file_editor", "openhands:terminal", "openhands:file_editor"]
    create_in, create_out = player.get_input_state(0), player.get_output_state(0)
    assert create_in["args"]["command"] == "create" and create_in["args"]["path"] == target
    assert create_out["status"] == "ok" and "created" in create_out["output"]
    assert create_out["observation"]["new_content"] == "x = 1\n"
    terminal_out = player.get_output_state(1)
    assert terminal_out["exit_code"] == 3 and "out" in terminal_out["output"]
    assert player.frames[2].metadata.error_type == "ToolError"
    assert all(f.metadata.latency_ms > 0 for f in player.frames)


def test_sentinel_rejects_next_action_after_blocker(workspace: Path, tmp_path: Path) -> None:
    sentinel = Sentinel(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")))
    agent, conv = _conversation(workspace, [sentinel])
    sentinel.bind(conv)
    target = str(workspace / "bad.py")
    marker = workspace / "ran.txt"
    try:
        with conv.state:
            create, run = _step(
                agent, conv,
                _event("file_editor", FileEditorAction(command="create", path=target,
                                                       file_text=SYNTAX_ERROR), 1),
                _event("terminal", TerminalAction(command=f"touch {marker}"), 2))
            assert type(create).__name__ == "ObservationEvent"
            assert type(run).__name__ == "UserRejectObservation"
            assert "SENTINEL BLOCKED" in run.rejection_reason
            assert not marker.exists()  # the terminal command never ran

            # The agent fixes the file with file_editor: allowed, and the gate lifts.
            (fix,) = _step(agent, conv, _event("file_editor", FileEditorAction(
                command="str_replace", path=target, old_str="def broken(:",
                new_str="def broken():"), 3))
            assert type(fix).__name__ == "ObservationEvent"
            assert sentinel.blocked_files == {}
            (again,) = _step(agent, conv, _event("terminal",
                                                 TerminalAction(command=f"touch {marker}"), 4))
            assert type(again).__name__ == "ObservationEvent"
    finally:
        conv.close()
    assert marker.exists()
    assert sentinel.stats.rejected_actions == 1 and sentinel.stats.self_corrections == 1


def test_sentinel_warning_delivered_as_user_message(workspace: Path, tmp_path: Path) -> None:
    sentinel = Sentinel(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")))
    agent, conv = _conversation(workspace, [sentinel])
    sentinel.bind(conv)
    try:
        with conv.state:
            _step(agent, conv, _event("file_editor", FileEditorAction(
                command="create", path=str(workspace / "w.py"), file_text="from os import *\n"),
                1))
        assert sentinel.deliver_warnings() == 1
        last = conv.state.events[-1]
        assert type(last).__name__ == "MessageEvent" and last.source == "user"
        assert "Wildcard import" in last.llm_message.content[0].text
    finally:
        conv.close()
