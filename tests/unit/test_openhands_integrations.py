"""VCRRuntime and Sentinel on OpenHands — fake events, always run.

The fakes mirror the OpenHands SDK 1.x event shapes (``ActionEvent``,
``ObservationEvent``, ``UserRejectObservation``, ``AgentErrorEvent``,
``state.block_action``) and the legacy V0 EventStream (``subscribe``).
The same flows run on the real SDK in ``test_openhands_sdk_real.py``.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import VCRPlayer, VCRRecorder
from agent_vcr.integrations.openhands_hook import VCRRuntime
from openhands_sentinel.sentinel import Sentinel, SentinelConfig

SYNTAX_ERROR = "def broken(:\n    pass\n"
_ids = itertools.count(1)


# ── SDK 1.x fakes (class names are what the integrations dispatch on) ──────


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


class _Action:
    def __init__(self, **fields: Any) -> None:
        self.fields = fields
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self.fields)


class _Observation(_Action):
    def __init__(self, text: str, **fields: Any) -> None:
        super().__init__(is_error=fields.pop("is_error", False), **fields)
        self.text = text


class ActionEvent:
    def __init__(self, tool_name: str, action: _Action | None, thought: str = "") -> None:
        n = next(_ids)
        self.id = f"action-{n}"
        self.tool_call_id = f"call-{n}"
        self.tool_name = tool_name
        self.action = action
        self.thought = [_Text(thought)] if thought else []
        self.tool_call = _Action(arguments='{"command": "x"}')


class ObservationEvent:
    def __init__(self, action: ActionEvent, observation: _Observation) -> None:
        self.action_id = action.id
        self.tool_call_id = action.tool_call_id
        self.tool_name = action.tool_name
        self.observation = observation


class UserRejectObservation:
    def __init__(self, action: ActionEvent, reason: str) -> None:
        self.action_id = action.id
        self.tool_call_id = action.tool_call_id
        self.tool_name = action.tool_name
        self.rejection_reason = reason


class AgentErrorEvent:
    def __init__(self, action: ActionEvent, error: str) -> None:
        self.tool_call_id = action.tool_call_id
        self.tool_name = action.tool_name
        self.error = error


class _State:
    def __init__(self) -> None:
        self.blocked_actions: dict[str, str] = {}

    def block_action(self, action_id: str, reason: str) -> None:
        self.blocked_actions[action_id] = reason


class FakeConversation:
    def __init__(self) -> None:
        self.state = _State()
        self.messages: list[str] = []

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def run_tool(self, callback: Any, action: ActionEvent, observation: _Observation) -> Any:
        """What ``Agent._execute_action_event`` does: reject if blocked, else observe."""
        reason = self.state.blocked_actions.pop(action.id, None)
        result: Any = (UserRejectObservation(action, reason) if reason is not None
                       else ObservationEvent(action, observation))
        callback(result)
        return result


def _create(path: Path, text: str) -> tuple[ActionEvent, _Observation]:
    action = ActionEvent("file_editor", _Action(command="create", path=str(path), file_text=text))
    return action, _Observation(f"File created successfully at: {path}", command="create",
                                path=str(path), new_content=text)


def _terminal(command: str, output: str = "", exit_code: int = 0) -> tuple[ActionEvent, Any]:
    action = ActionEvent("terminal", _Action(command=command, is_input=False))
    return action, _Observation(output, command=command, exit_code=exit_code)


# ── VCRRuntime on SDK 1.x events ────────────────────────────────────────────


def _runtime(tmp_path: Path, **kw: Any) -> VCRRuntime:
    return VCRRuntime(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")),
                      session_id="oh-sdk", **kw)


def test_runtime_records_one_frame_per_action_observation_pair(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path)
    conv = FakeConversation()
    action, obs = _terminal("pytest -q", "1 failed", exit_code=1)
    vcr(action)
    assert vcr.stats["frame_count"] == 0 and vcr.stats["pending_actions"] == 1
    conv.run_tool(vcr, action, obs)
    path = vcr.save()

    player = VCRPlayer.load(path)
    assert [f.node_name for f in player.frames] == ["openhands:terminal"]
    frame = player.frames[0]
    assert player.get_input_state(0)["args"] == {"command": "pytest -q", "is_input": False}
    out = player.get_output_state(0)
    assert out["output"] == "1 failed" and out["exit_code"] == 1 and out["status"] == "ok"
    assert frame.metadata.latency_ms >= 0 and frame.frame_type.value == "tool"
    assert vcr.stats["pending_actions"] == 0


def test_runtime_records_rejections_errors_and_tool_errors(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path)
    rejected, _ = _terminal("rm -rf /")
    vcr(rejected)
    vcr(UserRejectObservation(rejected, "blocked by airbag"))
    invalid = ActionEvent("terminal", None)
    vcr(invalid)
    vcr(AgentErrorEvent(invalid, "Error validating args"))
    failed = ActionEvent("file_editor", _Action(command="str_replace", path="/x.py"))
    vcr(failed)
    vcr(ObservationEvent(failed, _Observation("old_str not found", is_error=True,
                                              command="str_replace", path="/x.py")))
    frames = VCRPlayer.load(vcr.save()).frames

    assert [f.output_state["status"] for f in frames] == ["rejected", "error", "error"]
    assert frames[0].metadata.error_type == "Rejected"
    assert frames[0].metadata.error_message == "blocked by airbag"
    assert frames[1].metadata.error_type == "AgentError"
    assert frames[1].input_state["invalid_tool_call"] is True
    assert frames[1].input_state["args"] == {"command": "x"}  # raw tool-call arguments
    assert frames[2].metadata.error_type == "ToolError"
    assert frames[2].frame_type.value == "error"


def test_runtime_guardrail_truncates_oversized_pair(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path, max_frame_bytes=2_000)
    action, obs = _terminal("cat big.log", "x" * 10_000)
    vcr(action)
    vcr(ObservationEvent(action, obs))
    frame = VCRPlayer.load(vcr.save()).frames[0]
    assert frame.input_state["_vcr_truncated"] is True
    assert frame.output_state["_vcr_truncated"] is True
    assert len(frame.output_state["_preview"]) == 1000
    assert frame.output_state["exit_code"] == 0
    assert vcr.stats["oversized_frames"] == 1


def test_runtime_flushes_unanswered_actions_on_save(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path)
    action, _ = _terminal("sleep 999")
    vcr(action)
    frames = VCRPlayer.load(vcr.save()).frames
    assert frames[0].output_state["status"] == "no_observation"
    assert frames[0].metadata.error_type == "NoObservation"


def test_runtime_ignores_sdk_non_tool_events(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path)
    event = type("MessageEvent", (), {"__module__": "openhands.sdk.event"})()
    vcr(event)
    assert vcr.stats["frame_count"] == 0


def test_runtime_attach_rejects_sdk_conversation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="callbacks=\\[vcr_runtime\\]"):
        _runtime(tmp_path).attach(FakeConversation())
    with pytest.raises(TypeError, match="subscribe"):
        _runtime(tmp_path).attach(object())


def test_runtime_never_raises_into_the_agent(tmp_path: Path,
                                             caplog: pytest.LogCaptureFixture) -> None:
    vcr = _runtime(tmp_path)
    action, obs = _terminal("ls")
    vcr(action)

    def boom(*_: Any, **__: Any) -> None:
        raise OSError("disk full")

    vcr.recorder.record_step = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        vcr(ObservationEvent(action, obs))
    assert "Failed to record" in caplog.text


# ── VCRRuntime on the legacy V0 EventStream ─────────────────────────────────


class FakeStream:
    def __init__(self) -> None:
        self.subscribers: list[Any] = []

    def subscribe(self, _sid: Any, callback: Any, _cid: str) -> None:
        self.subscribers.append(callback)


class CmdRunAction:
    def __init__(self, command: str) -> None:
        self.command = command


class CmdOutputObservation:
    def __init__(self, content: str, exit_code: int) -> None:
        self.content = content
        self.exit_code = exit_code


def test_runtime_v0_stream_records_each_event(tmp_path: Path) -> None:
    vcr = _runtime(tmp_path)
    stream = FakeStream()
    vcr.attach(stream)
    stream.subscribers[0](CmdRunAction("ls"))
    stream.subscribers[0](CmdOutputObservation("a.py", 0))
    frames = VCRPlayer.load(vcr.save()).frames
    assert [f.node_name for f in frames] == [
        "openhands:CmdRunAction", "openhands:CmdOutputObservation"]
    assert frames[0].frame_type.value == "tool"
    assert vcr.stats["attached"] is True


# ── Sentinel on SDK 1.x events ──────────────────────────────────────────────


def _sentinel(tmp_path: Path, **config: Any) -> tuple[Sentinel, FakeConversation]:
    sentinel = Sentinel(config=SentinelConfig(**config),
                        recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")))
    conv = FakeConversation()
    sentinel.bind(conv)
    return sentinel, conv


def test_sentinel_analyzes_file_editor_create(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path)
    action, obs = _create(tmp_path / "ok.py", "x = 1\n")
    sentinel(action)
    conv.run_tool(sentinel, action, obs)
    assert sentinel.stats.files_analyzed == 1
    assert sentinel.blocked_files == {}


def test_sentinel_view_and_failed_edits_not_analyzed(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path)
    view = ActionEvent("file_editor", _Action(command="view", path=str(tmp_path)))
    sentinel(view)
    conv.run_tool(sentinel, view, _Observation("listing", command="view", path=str(tmp_path)))
    bad = ActionEvent("file_editor", _Action(command="str_replace", path=str(tmp_path / "a.py")))
    sentinel(bad)
    conv.run_tool(sentinel, bad, _Observation("no match", is_error=True, command="str_replace",
                                              path=str(tmp_path / "a.py"), new_content=None))
    assert sentinel.stats.files_analyzed == 0


def test_sentinel_str_replace_reads_disk_when_no_new_content(tmp_path: Path) -> None:
    target = tmp_path / "mod.py"
    target.write_text("from os import *\n")
    sentinel, conv = _sentinel(tmp_path)
    action = ActionEvent("file_editor", _Action(command="str_replace", path=str(target)))
    sentinel(action)
    conv.run_tool(sentinel, action, _Observation("edited", command="str_replace",
                                                 path=str(target), new_content=None))
    assert sentinel.stats.files_analyzed == 1
    assert sentinel._results_history[-1].warning_count == 1


def test_sentinel_blocker_rejects_next_action_until_fixed(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path)
    bad_path = tmp_path / "bad.py"
    create, obs = _create(bad_path, SYNTAX_ERROR)
    sentinel(create)
    conv.run_tool(sentinel, create, obs)
    assert str(bad_path) in sentinel.blocked_files

    # Next action that is not a fix: rejected through state.block_action.
    run, run_obs = _terminal("python bad.py")
    sentinel(run)
    result = conv.run_tool(sentinel, run, run_obs)
    assert isinstance(result, UserRejectObservation)
    assert "SENTINEL BLOCKED" in result.rejection_reason
    assert sentinel.stats.rejected_actions == 1

    # `think` and edits of the broken file are allowed; a clean rewrite lifts the gate.
    think = ActionEvent("think", _Action(thought="fix it"))
    sentinel(think)
    assert think.id not in conv.state.blocked_actions
    fix, fix_obs = _create(bad_path, "def fixed():\n    pass\n")
    sentinel(fix)
    assert isinstance(conv.run_tool(sentinel, fix, fix_obs), ObservationEvent)
    assert sentinel.blocked_files == {}
    run2, run2_obs = _terminal("python bad.py")
    sentinel(run2)
    assert isinstance(conv.run_tool(sentinel, run2, run2_obs), ObservationEvent)
    assert sentinel.stats.self_corrections == 1


def test_sentinel_blocker_rejects_rest_of_the_same_batch(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path)
    create, obs = _create(tmp_path / "bad.py", SYNTAX_ERROR)
    run, run_obs = _terminal("pytest")
    sentinel(create)  # one LLM response, two tool calls: both emitted first
    sentinel(run)
    conv.run_tool(sentinel, create, obs)
    assert run.id in conv.state.blocked_actions
    assert isinstance(conv.run_tool(sentinel, run, run_obs), UserRejectObservation)


def test_sentinel_block_on_blocker_false_never_rejects(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path, block_on_blocker=False)
    create, obs = _create(tmp_path / "bad.py", SYNTAX_ERROR)
    sentinel(create)
    conv.run_tool(sentinel, create, obs)
    run, _ = _terminal("ls")
    sentinel(run)
    assert conv.state.blocked_actions == {}
    assert sentinel.pending_warnings and "SENTINEL CODE REVIEW" in sentinel.pending_warnings[0]


def test_sentinel_unbound_logs_instead_of_rejecting(tmp_path: Path,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    sentinel = Sentinel(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")))
    conv = FakeConversation()
    create, obs = _create(tmp_path / "bad.py", SYNTAX_ERROR)
    sentinel(create)
    with caplog.at_level(logging.WARNING):
        conv.run_tool(sentinel, create, obs)
        run, _ = _terminal("ls")
        sentinel(run)
    assert "not bound" in caplog.text
    assert conv.state.blocked_actions == {} and sentinel.stats.rejected_actions == 0


def test_sentinel_warnings_are_queued_then_delivered(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path)
    create, obs = _create(tmp_path / "w.py", "from os import *\n")
    sentinel(create)
    conv.run_tool(sentinel, create, obs)
    assert conv.messages == []  # never injected mid-run
    assert sentinel.deliver_warnings() == 1
    assert "Wildcard import" in conv.messages[0]
    assert sentinel.pending_warnings == [] and sentinel.deliver_warnings() == 0


def test_sentinel_auto_analyze_false_ignores_sdk_events(tmp_path: Path) -> None:
    sentinel, conv = _sentinel(tmp_path, auto_analyze=False)
    create, obs = _create(tmp_path / "bad.py", SYNTAX_ERROR)
    sentinel(create)
    conv.run_tool(sentinel, create, obs)
    assert sentinel.stats.files_analyzed == 0


def test_sentinel_attach_with_sdk_conversation_binds(tmp_path: Path) -> None:
    sentinel = Sentinel(recorder=VCRRecorder(output_dir=str(tmp_path / ".vcr")))
    conv = FakeConversation()
    assert sentinel.attach(conv) is sentinel
    assert sentinel._conversation is conv


def test_sentinel_and_runtime_share_one_recording(tmp_path: Path) -> None:
    recorder = VCRRecorder(output_dir=str(tmp_path / ".vcr"))
    sentinel = Sentinel(recorder=recorder)
    vcr = VCRRuntime(recorder=recorder)
    conv = FakeConversation()
    sentinel.bind(conv)

    def both(event: Any) -> None:
        vcr(event)
        sentinel(event)

    create, obs = _create(tmp_path / "bad.py", SYNTAX_ERROR)
    both(create)
    conv.run_tool(both, create, obs)
    run, run_obs = _terminal("ls")
    both(run)
    conv.run_tool(both, run, run_obs)
    frames = VCRPlayer.load(vcr.save()).frames
    assert Path(vcr.save()).name == "sentinel-openhands.vcr"
    assert [f.node_name for f in frames] == [
        "openhands:file_editor", "sentinel:analysis", "sentinel:reject", "openhands:terminal"]
    assert frames[-1].output_state["status"] == "rejected"
