"""OpenHands SDK airbag integration, exercised on the real SDK without model calls.

Drives the same code paths ``Agent.step`` uses (emit ActionEvent → execute)
with hand-built events. Skipped unless ``openhands-sdk`` is installed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("openhands.sdk")
pytest.importorskip("openhands.tools.terminal.definition")

from openhands.sdk import LLM, Agent, Conversation  # noqa: E402
from openhands.sdk.event import ActionEvent  # noqa: E402
from openhands.sdk.llm import MessageToolCall, TextContent  # noqa: E402
from openhands.tools.file_editor.definition import FileEditorAction  # noqa: E402
from openhands.tools.terminal.definition import TerminalAction  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from agent_vcr.airbag.store import ShadowStore  # noqa: E402
from agent_vcr.integrations.openhands_airbag import AirbagCallback, protect  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    return ws


def _event(tool: str, action: Any, i: int) -> ActionEvent:
    call = MessageToolCall(id=f"call_{i}", name=tool, arguments=json.dumps(action.model_dump()),
                           origin="completion")
    return ActionEvent(action=action, thought=[TextContent(text="")], tool_name=tool,
                       tool_call_id=call.id, tool_call=call, llm_response_id="r1")


def _conversation(ws: Path, callback: AirbagCallback) -> tuple[Agent, Any]:
    agent = Agent(llm=LLM(model="gpt-4o-mini", api_key=SecretStr("unused"), usage_id="t"),
                  tools=[])
    conv = Conversation(agent=agent, workspace=str(ws), callbacks=[callback], visualizer=None)
    return agent, conv


def test_snapshot_before_terminal_action_and_view_ignored(workspace: Path) -> None:
    callback = AirbagCallback()
    _, conv = _conversation(workspace, callback)
    protect(conv, callback)
    with conv.state:
        conv._on_event(_event("terminal", TerminalAction(command="rm -rf src"), 1))
        conv._on_event(_event("file_editor",
                              FileEditorAction(command="view", path=str(workspace / "src")), 2))
    assert len(ShadowStore(workspace).snapshots()) == 1
    assert [d.verdict.level for d in callback.decisions] == ["destructive"]


def test_irreversible_action_is_rejected(workspace: Path) -> None:
    callback = AirbagCallback()
    agent, conv = _conversation(workspace, callback)
    protect(conv, callback)
    with conv.state:
        event = _event("terminal", TerminalAction(command="psql -h db.prod -c 'DROP TABLE x'"), 3)
        conv._on_event(event)
        result = agent._execute_action_event(conv, event, on_event=conv._on_event)
    assert type(result).__name__ == "UserRejectObservation"
    assert "airbag" in result.rejection_reason


def test_confirm_mode_pauses_only_irreversible(workspace: Path) -> None:
    callback = AirbagCallback()
    agent, conv = _conversation(workspace, callback)
    protect(conv, callback, confirm=True)
    with conv.state:
        risky = _event("terminal", TerminalAction(command="terraform destroy -auto-approve"), 4)
        conv._on_event(risky)
        assert agent._requires_user_confirmation(conv.state, [risky])
    with conv.state:
        harmless = _event("terminal", TerminalAction(command="pytest -q"), 5)
        conv._on_event(harmless)
        assert not agent._requires_user_confirmation(conv.state, [harmless])
