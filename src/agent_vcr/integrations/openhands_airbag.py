"""Agent VCR airbag for OpenHands SDK agents (``openhands-sdk`` 1.x).

Two pieces, verified against openhands-sdk 1.7.3:

* :class:`AirbagCallback` — a conversation callback. OpenHands calls it
  synchronously *before* each action executes, so it snapshots the workspace
  first; irreversible commands are blocked with ``state.block_action``
  (the agent sees a rejection and can explain to the user).
* :func:`make_security_analyzer` — marks irreversible actions HIGH risk so
  ``ConfirmRisky`` pauses the run for a human instead of blocking.

Usage::

    from agent_vcr.integrations.openhands_airbag import protect

    airbag = AirbagCallback()                       # or protect() below
    conversation = Conversation(agent=agent, workspace=ws, callbacks=[airbag])
    protect(conversation, airbag)                   # bind (+ confirm=True for prompts)

Undo anything afterwards with ``vcr undo`` in the workspace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_vcr.airbag.core import Airbag, Policy, ToolCall

logger = logging.getLogger(__name__)

#: file_editor sub-commands that change files ("view" only reads).
MUTATING_EDITOR_COMMANDS = frozenset({"create", "str_replace", "insert", "undo_edit"})
EDITOR_TOOLS = frozenset({"file_editor", "planning_file_editor"})


def describe_action(event: Any, workspace: str) -> ToolCall | None:
    """Turn an OpenHands ``ActionEvent`` into an airbag :class:`ToolCall`."""
    action = getattr(event, "action", None)
    if action is None:
        return None  # failed validation: OpenHands will not execute it
    try:
        args: dict[str, Any] = action.model_dump()
    except Exception:  # noqa: BLE001
        return None
    tool = str(getattr(event, "tool_name", "") or "")
    raw_id = getattr(event, "tool_call_id", None)
    call_id = str(raw_id) if raw_id else None

    def make(kind: str, command: str | None = None, paths: list[str] | None = None) -> ToolCall:
        return ToolCall(agent="openhands", tool_name=tool, kind=kind, cwd=workspace,
                        command=command, paths=paths or [], project_dir=workspace,
                        call_id=call_id)

    if tool == "terminal":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip() or args.get("is_input"):
            return None  # empty poll, C-c, or stdin for a running process
        return make("shell", command)
    if tool in EDITOR_TOOLS:
        if args.get("command") not in MUTATING_EDITOR_COMMANDS:
            return None
        path = args.get("path")
        return make("edit", paths=[path] if isinstance(path, str) else [])
    if tool in {"write_file", "edit"}:
        path = args.get("file_path")
        return make("edit", paths=[path] if isinstance(path, str) else [])
    if tool == "apply_patch":
        from agent_vcr.airbag.agents import _PATCH_FILE

        patch = args.get("patch") or ""
        paths = [m.strip() for m in _PATCH_FILE.findall(patch)] if isinstance(patch, str) else []
        return make("edit", paths=paths)
    return None


class AirbagCallback:
    """OpenHands conversation callback: snapshot before every mutating action.

    Args:
        workspace: Project directory (default: the conversation's workspace).
        block_irreversible: Block commands a snapshot cannot undo. Set False
            when you use ``protect(..., confirm=True)`` so a human is asked
            instead.
    """

    def __init__(self, workspace: str | Path | None = None, *, block_irreversible: bool = True,
                 policy: Policy | None = None) -> None:
        self.workspace = str(workspace) if workspace is not None else None
        self.block_irreversible = block_irreversible
        self.policy = policy
        self.conversation: Any = None
        self.decisions: list[Any] = []

    def bind(self, conversation: Any) -> AirbagCallback:
        self.conversation = conversation
        if self.workspace is None:
            self.workspace = str(conversation.state.workspace.working_dir)
        return self

    def __call__(self, event: Any) -> None:
        conversation = self.conversation
        if conversation is None:
            return  # OpenHands emits a SystemPromptEvent inside Conversation.__init__
        if type(event).__name__ != "ActionEvent":
            return
        workspace = self.workspace or str(conversation.state.workspace.working_dir)
        try:
            call = describe_action(event, workspace)
            if call is None:
                return
            decision = Airbag(workspace, policy=self.policy).before(call)
        except Exception:  # noqa: BLE001 - a callback exception would abort the agent run
            logger.warning("Agent VCR airbag failed; action allowed", exc_info=True)
            return
        self.decisions.append(decision)
        if decision.user_message:
            logger.warning("%s", decision.user_message)
        if decision.action in {"ask", "deny"} and self.block_irreversible:
            conversation.state.block_action(
                event.id, decision.agent_message or "Blocked by Agent VCR airbag")


def make_security_analyzer(workspace: str | Path) -> Any:
    """A ``SecurityAnalyzerBase`` rating irreversible actions HIGH (others LOW)."""
    from openhands.sdk.security.analyzer import SecurityAnalyzerBase
    from openhands.sdk.security.risk import SecurityRisk

    class AgentVCRAirbagAnalyzer(SecurityAnalyzerBase):  # type: ignore[misc,valid-type]
        airbag_workspace: str = ""

        def security_risk(self, action: Any) -> Any:
            call = describe_action(action, self.airbag_workspace)
            if call is None:
                return SecurityRisk.LOW
            verdict = Airbag(self.airbag_workspace).classify(call)
            if verdict.level == "irreversible":
                return SecurityRisk.HIGH
            return SecurityRisk.MEDIUM if verdict.level == "destructive" else SecurityRisk.LOW

    return AgentVCRAirbagAnalyzer(airbag_workspace=str(workspace))


def protect(conversation: Any, callback: AirbagCallback | None = None, *,
            confirm: bool = False) -> AirbagCallback:
    """Bind the airbag to a conversation.

    ``confirm=True`` pauses on irreversible actions (``WAITING_FOR_CONFIRMATION``)
    instead of blocking them. Only use it if your loop handles confirmation:
    calling ``conversation.run()`` again *approves* the paused actions.
    """
    callback = callback or AirbagCallback()
    callback.bind(conversation)
    if confirm:
        from openhands.sdk.security.confirmation_policy import ConfirmRisky
        from openhands.sdk.security.risk import SecurityRisk

        callback.block_irreversible = False
        conversation.set_security_analyzer(make_security_analyzer(callback.workspace or "."))
        conversation.set_confirmation_policy(
            ConfirmRisky(threshold=SecurityRisk.HIGH, confirm_unknown=False))
    return callback
