"""
Sentinel — The main orchestrator.

Connects the CodeAnalyzer to agent-vcr's VCRRecorder and optionally
to the OpenHands EventStream. Every analysis result is recorded as
a VCR frame — creating a full audit trail of what the agent wrote,
what Sentinel caught, and whether the agent self-corrected.

Usage:
    from openhands_sentinel import Sentinel, SentinelConfig
    from agent_vcr import VCRRecorder

    recorder = VCRRecorder()
    sentinel = Sentinel(recorder=recorder)

    # Analyze a file write (standalone mode)
    result = sentinel.check_file("app/auth.py", code_content)

    # OpenHands SDK 1.x: a conversation callback, bound for rejections
    conversation = Conversation(agent=agent, workspace=ws, callbacks=[sentinel])
    sentinel.bind(conversation)

    # Legacy OpenHands V0: attach to the EventStream
    sentinel.attach(event_stream)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_vcr import VCRRecorder
from agent_vcr.models import FrameMetadata, FrameType
from openhands_sentinel.analyzer import (
    AnalysisResult,
    CodeAnalyzer,
    Severity,
    Violation,
)

logger = logging.getLogger(__name__)

# OpenHands SDK 1.x (openhands-sdk / openhands-tools, verified against 1.7.3).
_SDK_EVENTS = frozenset({"ActionEvent", "ObservationEvent", "UserRejectObservation",
                         "AgentErrorEvent"})
_SDK_EDITOR_TOOLS = frozenset({"file_editor", "planning_file_editor"})
#: file_editor commands after which the file on disk changed.
_SDK_WRITE_COMMANDS = frozenset({"create", "str_replace", "insert", "undo_edit"})
#: Tools that cannot touch the workspace, allowed while a blocker is open.
_SDK_HARMLESS_TOOLS = frozenset({"think"})


def _blocker_keys(result: AnalysisResult) -> Counter[tuple[str, str]]:
    """Blockers by identity, ignoring line numbers (an edit above shifts them)."""
    return Counter((v.rule, v.function_name or "") for v in result.violations
                   if v.severity == Severity.BLOCKER)


@dataclass
class _Gate:
    """A file whose edit introduced blockers; later actions are rejected until it's fixed."""

    reason: str
    baseline: Counter[tuple[str, str]]  # blockers the file had before the gating edit
    on_disk: bool  # the file was readable here when gated (so "missing" means deleted)
    rejections: int = 0
    introduced: list[str] = field(default_factory=list)


def _sdk_action_path(tool: str, action: Any) -> str | None:
    if tool not in _SDK_EDITOR_TOOLS or action is None:
        return None
    path = getattr(action, "path", None)
    return os.path.normpath(str(path)) if path else None


@dataclass
class SentinelConfig:
    """Configuration for the Sentinel guardian."""

    # Analysis thresholds
    max_function_lines: int = 50
    max_complexity: int = 10
    max_file_lines: int = 500
    max_class_methods: int = 15
    max_function_params: int = 7

    # Behavior
    # Mark results with BLOCKER violations as ``blocked`` (AnalysisResult.blocked,
    # stats.blocked_actions, "BLOCKED" warning header, VCR frame output).
    block_on_blocker: bool = True
    warn_agent: bool = True  # Inject warnings into EventStream
    record_to_vcr: bool = True  # Record checks as VCR frames
    # Auto-analyze file write/edit events from an attached EventStream. When False,
    # events are still recorded but only explicit check_file() calls analyze.
    auto_analyze: bool = True

    # Frame size guardrail (addresses OpenHands issue #7402)
    max_frame_bytes: int = 50_000  # Warn if any recorded frame > 50KB
    truncate_large_frames: bool = True

    # OpenHands SDK gate: after this many rejected actions for one file, Sentinel
    # releases the gate (with a loud warning) instead of wedging the agent.
    max_rejections_per_file: int = 3


@dataclass
class SentinelStats:
    """Running statistics for a Sentinel session."""

    files_analyzed: int = 0
    total_violations: int = 0
    blockers: int = 0
    criticals: int = 0
    warnings: int = 0
    infos: int = 0
    self_corrections: int = 0
    agent_warnings_sent: int = 0
    blocked_actions: int = 0
    # Actions actually vetoed through the OpenHands SDK (``state.block_action``).
    rejected_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_analyzed": self.files_analyzed,
            "total_violations": self.total_violations,
            "blockers": self.blockers,
            "criticals": self.criticals,
            "warnings": self.warnings,
            "infos": self.infos,
            "self_corrections": self.self_corrections,
            "agent_warnings_sent": self.agent_warnings_sent,
            "blocked_actions": self.blocked_actions,
            "rejected_actions": self.rejected_actions,
        }


class Sentinel:
    """
    Real-time code quality guardian for AI agents.

    Watches every file the agent writes, runs AST analysis, detects
    quality violations, and optionally warns the agent to self-correct.
    Every check is recorded as a VCR frame for audit trails.

    Three modes:
    1. Standalone — call sentinel.check_file() manually
    2. Watch — sentinel.watch_directory() monitors a dir
    3. Native — ``Conversation(callbacks=[sentinel])`` + ``sentinel.bind(conversation)``
       (OpenHands SDK 1.x) or ``sentinel.attach(event_stream)`` (legacy V0)
    """

    def __init__(
        self,
        config: SentinelConfig | None = None,
        recorder: VCRRecorder | None = None,
        on_violation: Callable[[Violation], None] | None = None,
    ) -> None:
        self.config = config or SentinelConfig()
        self.recorder = recorder or VCRRecorder()
        self._on_violation = on_violation

        self.analyzer = CodeAnalyzer(
            max_function_lines=self.config.max_function_lines,
            max_complexity=self.config.max_complexity,
            max_file_lines=self.config.max_file_lines,
            max_class_methods=self.config.max_class_methods,
            max_function_params=self.config.max_function_params,
        )

        self.stats = SentinelStats()
        self._session_started = False
        self._event_stream: Any = None
        self._results_history: list[AnalysisResult] = []
        # Paths of FileEditActions without inline content, awaiting their observation.
        self._pending_edits: set[str] = set()
        self._pending_lock = threading.RLock()
        # OpenHands SDK 1.x state.
        self._conversation: Any = None
        # tool_call_id -> (action_id, tool_name, normalized path) for actions not yet run.
        self._pending_actions: dict[str, tuple[str, str, str | None]] = {}
        # tool_call_id -> file content read just before a file_editor write ran.
        self._pre_edit: dict[str, str | None] = {}
        # Files whose last edit introduced blockers; gates later actions.
        self._gates: dict[str, _Gate] = {}
        self._delivering = False
        #: Non-blocking warnings waiting for :meth:`deliver_warnings`.
        self.pending_warnings: list[str] = []
        self._unbound_warned = False

    def start_session(self, session_id: str | None = None) -> None:
        """Start a Sentinel monitoring session."""
        self.recorder.start_session(
            session_id=session_id or "sentinel-session",
            metadata={"mode": "sentinel", "config": self.config.__dict__},
            tags=["sentinel", "quality-guard"],
        )
        self._session_started = True
        logger.info("[sentinel] Session started")

    def check_file(
        self, file_path: str, content: str, *, previous_content: str | None = None
    ) -> AnalysisResult:
        """
        Analyze a single file and record the result.

        This is the core method. Everything else calls this.

        With ``previous_content`` (the file before the edit), only blockers the
        edit *introduced* mark the result ``blocked``; blockers that were already
        there are reported but do not block.
        """
        baseline = None if previous_content is None else self._probe(file_path, previous_content)
        return self._check(file_path, content, baseline)

    def _probe(self, file_path: str, content: str) -> Counter[tuple[str, str]]:
        """Blockers of ``content`` without touching stats, history or trajectory memory."""
        probe = CodeAnalyzer(
            max_function_lines=self.config.max_function_lines,
            max_complexity=self.config.max_complexity,
            max_file_lines=self.config.max_file_lines,
            max_class_methods=self.config.max_class_methods,
            max_function_params=self.config.max_function_params,
        )
        return _blocker_keys(probe.analyze(file_path, content))

    def _check(
        self, file_path: str, content: str, baseline: Counter[tuple[str, str]] | None
    ) -> AnalysisResult:
        if not self._session_started:
            self.start_session()

        start = time.perf_counter()
        result = self.analyzer.analyze(file_path, content)
        elapsed_ms = (time.perf_counter() - start) * 1000
        previous_result = next(
            (item for item in reversed(self._results_history) if item.file_path == file_path),
            None,
        )

        # Update stats
        self.stats.files_analyzed += 1
        self.stats.total_violations += len(result.violations)
        self.stats.blockers += result.blocker_count
        self.stats.criticals += result.critical_count
        self.stats.warnings += result.warning_count
        self.stats.infos += sum(
            1 for v in result.violations if v.severity == Severity.INFO
        )
        if (
            previous_result is not None
            and previous_result.violations
            and not result.violations
        ):
            self.stats.self_corrections += 1
        introduced = _blocker_keys(result)
        if baseline is not None:
            introduced = introduced - baseline
        if self.config.block_on_blocker and introduced:
            result.blocked = True
            self.stats.blocked_actions += 1

        self._results_history.append(result)

        # Record as VCR frame
        if self.config.record_to_vcr:
            self._record_analysis(file_path, result, elapsed_ms)

        # Fire violation callbacks
        for violation in result.violations:
            if self._on_violation:
                try:
                    self._on_violation(violation)
                except Exception:
                    logger.debug("on_violation callback error", exc_info=True)

        # Log summary
        if result.violations:
            logger.warning(
                "[sentinel] %s: %d violations (%d blocker, %d critical) in %.1fms",
                file_path,
                len(result.violations),
                result.blocker_count,
                result.critical_count,
                elapsed_ms,
            )
        else:
            logger.info("[sentinel] %s: clean ✓ (%.1fms)", file_path, elapsed_ms)
        if result.blocked:
            logger.warning(
                "[sentinel] BLOCKED: %s has %d blocker violation(s)",
                file_path, result.blocker_count,
            )

        return result

    def forget_file(self, file_path: str) -> None:
        """Drop trajectory memory for a deleted file so it no longer counts as a duplicate."""
        self.analyzer.forget_file(file_path)

    def check_and_warn(
        self, file_path: str, content: str, *, previous_content: str | None = None
    ) -> tuple[AnalysisResult, str | None]:
        """
        Analyze a file and generate an agent-facing warning string.

        ``previous_content`` works as in :meth:`check_file`.

        Returns:
            Tuple of (AnalysisResult, warning_message_or_None)
        """
        baseline = None if previous_content is None else self._probe(file_path, previous_content)
        return self._check_and_warn(file_path, content, baseline)

    def _check_and_warn(
        self, file_path: str, content: str, baseline: Counter[tuple[str, str]] | None
    ) -> tuple[AnalysisResult, str | None]:
        result = self._check(file_path, content, baseline)

        if not result.violations:
            return result, None

        # Build a concise warning message for the agent
        if result.blocked:
            lines = [
                "🛑 SENTINEL BLOCKED — This change has blocker-level issues and must "
                "not proceed. Please fix before proceeding:\n"
            ]
        elif result.blocker_count:
            lines = [
                "🛡️ SENTINEL CODE REVIEW — This file already had blocker-level issues before "
                "this edit (not blocking). Issues detected:\n"
            ]
        else:
            lines = [
                "🛡️ SENTINEL CODE REVIEW — Issues detected, please fix before proceeding:\n"
            ]

        for v in result.violations:
            lines.append(f"  {v.to_agent_warning()}")

        lines.append(
            "\nFix the issues above in your next action. "
            "Do NOT proceed to the next task until these are resolved."
        )

        warning = "\n".join(lines)
        self.stats.agent_warnings_sent += 1

        return result, warning

    def get_report(self) -> str:
        """Generate a human-readable session report."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║        SENTINEL SESSION REPORT               ║",
            "╚══════════════════════════════════════════════╝",
            "",
            f"  Files analyzed:     {self.stats.files_analyzed}",
            f"  Total violations:   {self.stats.total_violations}",
            f"    ╰─ Blockers:      {self.stats.blockers}",
            f"    ╰─ Criticals:     {self.stats.criticals}",
            f"    ╰─ Warnings:      {self.stats.warnings}",
            f"  Warnings sent:      {self.stats.agent_warnings_sent}",
            f"  Blocked actions:    {self.stats.blocked_actions}",
            f"  Self-corrections:   {self.stats.self_corrections}",
            "",
        ]

        # Trajectory summary
        trajectory = self.analyzer.get_trajectory_summary()
        lines.extend([
            "  Trajectory Memory:",
            f"    Functions tracked: {trajectory['tracked_functions']}",
            f"    Files tracked:     {trajectory['tracked_files']}",
            f"    Duplicate candidates: {trajectory['duplicate_candidates']}",
            "",
        ])

        # Recent violations
        if self._results_history:
            lines.append("  Recent Activity:")
            for result in self._results_history[-5:]:
                status = "✓ clean" if result.passed else f"✗ {len(result.violations)} issues"
                lines.append(f"    {result.file_path}: {status}")

        lines.append("")
        return "\n".join(lines)

    def save(self) -> str:
        """Save the VCR recording and return the path."""
        path = self.recorder.save()
        logger.info("[sentinel] Session saved to %s", path)
        return str(path)

    # ──────────────────────────────────────────────
    #  OpenHands SDK 1.x (Conversation callbacks)
    # ──────────────────────────────────────────────

    def bind(self, conversation: Any) -> Sentinel:
        """Bind to an OpenHands SDK 1.x ``Conversation`` so blockers can reject actions.

        Pass the Sentinel as a callback and bind it once the conversation exists::

            sentinel = Sentinel()
            conversation = Conversation(agent=agent, workspace=ws, callbacks=[sentinel])
            sentinel.bind(conversation)

        Without ``bind`` Sentinel still analyzes edits, but cannot reject anything.
        """
        self._conversation = conversation
        if not self._session_started:
            self.start_session("sentinel-openhands")
        return self

    def __call__(self, event: Any) -> None:
        """OpenHands SDK 1.x conversation callback (runs before each action executes)."""
        name = type(event).__name__
        if not self.config.auto_analyze:
            return
        try:
            if name == "ActionEvent":
                self._on_sdk_action(event)
            elif name in _SDK_EVENTS:
                self._on_sdk_result(event)
            elif name == "MessageEvent" and getattr(event, "source", None) == "user":
                self._on_user_message()
        except Exception:  # noqa: BLE001 - a callback exception would abort the agent run
            logger.warning("[sentinel] Failed to handle %s", name, exc_info=True)

    @property
    def blocked_files(self) -> dict[str, str]:
        """Files whose last edit introduced blockers (path -> agent-facing reason)."""
        with self._pending_lock:
            return {path: gate.reason for path, gate in self._gates.items()}

    def unblock(self, file_path: str | None = None) -> list[str]:
        """Release the gate for one file (or all files). Returns the paths released.

        Use it when you have decided a blocker is acceptable and the agent should
        carry on. The findings stay in the recording.
        """
        with self._pending_lock:
            if file_path is None:
                released = sorted(self._gates)
                self._gates.clear()
            else:
                path = os.path.normpath(file_path)
                released = [path] if self._gates.pop(path, None) is not None else []
        for path in released:
            self._record_release(path, "unblocked by caller")
        return released

    def deliver_warnings(self, conversation: Any = None) -> int:
        """Send queued non-blocking warnings to the agent as one user message.

        Call it between ``conversation.run()`` calls, never from a callback: a user
        message inserted while a batch of tool calls is still running would break
        the tool-call/result order the LLM expects. Returns the number delivered.
        """
        target = conversation or self._conversation
        with self._pending_lock:
            warnings = list(self.pending_warnings)
        if not warnings or target is None:
            return 0
        self._delivering = True  # our own message must not count as the user moving on
        try:
            target.send_message("\n\n".join(warnings))
        finally:
            self._delivering = False
        with self._pending_lock:
            del self.pending_warnings[: len(warnings)]
        return len(warnings)

    def _on_user_message(self) -> None:
        """A new user message means a new instruction: drop the gate."""
        if self._delivering:
            return
        with self._pending_lock:
            released = sorted(self._gates)
            self._gates.clear()
        for path in released:
            self._record_release(path, "new user message")

    def _on_sdk_action(self, event: Any) -> None:
        action = getattr(event, "action", None)
        tool = str(getattr(event, "tool_name", "") or "")
        path = _sdk_action_path(tool, action)
        action_id = str(getattr(event, "id", "") or "")
        key = str(getattr(event, "tool_call_id", "") or action_id)
        if action is None or not action_id:
            return  # failed validation: OpenHands will not execute it
        pre_edit: str | None = None
        if path is not None and getattr(action, "command", None) in _SDK_WRITE_COMMANDS:
            pre_edit = self._read_from_disk(path)  # baseline for "did this edit break it?"
        with self._pending_lock:
            self._pending_actions[key] = (action_id, tool, path)
            if path is not None:
                self._pre_edit[key] = pre_edit
            has_gate = bool(self._gates)
        if not has_gate or self._allowed_while_blocked(tool, path):
            return
        gate = self._active_gate()
        if gate and not self._allowed_while_blocked(tool, path):
            self._reject(action_id, tool, gate)

    def _on_sdk_result(self, event: Any) -> None:
        key = str(getattr(event, "tool_call_id", "") or getattr(event, "action_id", "") or "")
        with self._pending_lock:
            pending = self._pending_actions.pop(key, None)
            pre_edit = self._pre_edit.pop(key, None)
        if type(event).__name__ != "ObservationEvent":
            return  # rejected or failed: nothing was written
        tool = str(getattr(event, "tool_name", "") or "")
        observation = getattr(event, "observation", None)
        if tool not in _SDK_EDITOR_TOOLS or observation is None:
            return
        if getattr(observation, "command", None) not in _SDK_WRITE_COMMANDS:
            return
        if getattr(observation, "is_error", False):
            return  # the edit failed, the file is unchanged
        raw_path = getattr(observation, "path", None) or (pending[2] if pending else None)
        if not raw_path:
            return
        file_path = os.path.normpath(str(raw_path))
        content = getattr(observation, "new_content", None)
        on_disk = False
        if content is None:
            content = self._read_from_disk(file_path)
            on_disk = content is not None
        else:
            on_disk = os.path.isfile(file_path)
        if content is None:
            logger.warning("[sentinel] No post-edit content available for %s", file_path)
            return

        before = getattr(observation, "old_content", None)
        if before is None:
            before = pre_edit
        with self._pending_lock:
            existing = self._gates.get(file_path)
        if existing is not None:
            baseline = existing.baseline  # still compare with the file before it broke
        elif before is not None:
            baseline = self._probe(file_path, before)
        else:
            baseline = Counter()  # new file: every blocker is new

        result, warning = self._check_and_warn(file_path, content, baseline)
        if result.blocked:
            introduced = sorted({rule for rule, _ in _blocker_keys(result) - baseline})
            reason = (
                f"Sentinel rejected this action: your edit of {file_path} introduced "
                f"blocker issues ({', '.join(introduced)}). Fix {file_path} with file_editor "
                f"before doing anything else.\n\n{warning}"
            )
            with self._pending_lock:
                self._gates[file_path] = _Gate(
                    reason=reason, baseline=baseline, on_disk=on_disk,
                    rejections=existing.rejections if existing else 0, introduced=introduced)
                # Tool calls from the same LLM response are emitted before any of them
                # runs: reject the ones still waiting so nothing runs on the broken file.
                waiting = [p for p in self._pending_actions.values()
                           if not self._allowed_while_blocked(p[1], p[2])]
            for action_id, waiting_tool, _ in waiting:
                gate = self._active_gate(refresh=False)
                if gate:
                    self._reject(action_id, waiting_tool, gate)
            return
        with self._pending_lock:
            self._gates.pop(file_path, None)
            if warning and self.config.warn_agent:
                self.pending_warnings.append(warning)

    def _active_gate(self, refresh: bool = True) -> dict[str, str]:
        """The gate to enforce now: stale entries dropped, over-cap entries released."""
        with self._pending_lock:
            gates = dict(self._gates)
        released: list[tuple[str, str]] = []
        for path, gate in gates.items():
            if refresh and gate.on_disk:
                content = self._read_from_disk(path)
                if content is None:
                    released.append((path, "file no longer exists"))
                    continue
                if not (self._probe(path, content) - gate.baseline):
                    released.append((path, "fixed outside file_editor"))
                    continue
            if gate.rejections >= max(1, self.config.max_rejections_per_file):
                released.append((path, f"released after {gate.rejections} rejected actions"))
        with self._pending_lock:
            for path, _ in released:
                self._gates.pop(path, None)
            active = {path: gate.reason for path, gate in self._gates.items()}
        for path, why in released:
            self._record_release(path, why)
        return active

    def _record_release(self, path: str, why: str) -> None:
        loud = why.startswith("released after")
        message = f"[sentinel] Gate on {path} lifted: {why}"
        if loud:
            message += (". The blocker is still in the file; Sentinel stopped rejecting "
                        "actions so the agent is not wedged")
            logger.warning(message)
            if self.config.warn_agent:
                with self._pending_lock:
                    self.pending_warnings.append(
                        f"🛑 SENTINEL: {path} still has blocker issues. Sentinel stopped "
                        f"rejecting actions after {self.config.max_rejections_per_file} "
                        "attempts — fix it before you finish.")
        else:
            logger.info(message)
        if self.config.record_to_vcr:
            try:
                self.recorder.record_step(
                    node_name="sentinel:release",
                    input_state={"file_path": path},
                    output_state={"released": True, "reason": why},
                    metadata=FrameMetadata(custom={"sentinel_check": True}),
                    frame_type=FrameType.TOOL_CALL,
                )
            except Exception:  # noqa: BLE001
                logger.warning("[sentinel] Failed to record gate release", exc_info=True)

    def _allowed_while_blocked(self, tool: str, path: str | None) -> bool:
        if tool in _SDK_HARMLESS_TOOLS:
            return True
        with self._pending_lock:
            return tool in _SDK_EDITOR_TOOLS and path is not None and path in self._gates

    def _reject(self, action_id: str, tool: str, gate: dict[str, str]) -> None:
        state = getattr(self._conversation, "state", None)
        if state is None or not hasattr(state, "block_action"):
            if not self._unbound_warned:
                self._unbound_warned = True
                logger.warning(
                    "[sentinel] Blocker found but Sentinel is not bound to the conversation; "
                    "call sentinel.bind(conversation) so it can reject actions"
                )
            return
        reason = "\n\n".join(gate.values())
        state.block_action(action_id, reason)
        with self._pending_lock:
            for path in gate:
                if path in self._gates:
                    self._gates[path].rejections += 1
        self.stats.rejected_actions += 1
        logger.warning("[sentinel] Rejected %s action %s: blocker in %s",
                       tool or "unknown", action_id, ", ".join(gate))
        if self.config.record_to_vcr:
            try:
                self.recorder.record_step(
                    node_name="sentinel:reject",
                    input_state={"action_id": action_id, "tool": tool},
                    output_state={"rejected": True, "blocked_files": sorted(gate)},
                    metadata=FrameMetadata(custom={"sentinel_check": True, "rejected": True}),
                    frame_type=FrameType.TOOL_CALL,
                )
            except Exception:  # noqa: BLE001
                logger.warning("[sentinel] Failed to record rejection", exc_info=True)

    # ──────────────────────────────────────────────
    #  Legacy OpenHands V0 EventStream Integration
    # ──────────────────────────────────────────────

    def attach(self, event_stream: Any) -> Sentinel:
        """
        Attach to an OpenHands event source.

        * Legacy V0 ``EventStream`` (has ``subscribe``): Sentinel analyzes every
          FileWriteAction and FileEditAction that flows through the stream.
        * OpenHands SDK 1.x ``Conversation``: same as :meth:`bind` (Sentinel must
          also be in the conversation's ``callbacks``).

        Returns:
            self (for chaining)
        """
        if not hasattr(event_stream, "subscribe") and hasattr(event_stream, "state"):
            return self.bind(event_stream)

        try:
            from openhands.events.stream import EventStreamSubscriber
            subscriber_id = EventStreamSubscriber.RUNTIME
        except ImportError:
            subscriber_id = "sentinel"

        self._event_stream = event_stream

        # Subscribe using the OpenHands pattern
        if hasattr(event_stream, "subscribe"):
            event_stream.subscribe(subscriber_id, self._on_event, "sentinel")
            logger.info("[sentinel] Attached to OpenHands EventStream")
        else:
            logger.warning("[sentinel] EventStream does not support subscribe()")

        if not self._session_started:
            self.start_session("sentinel-openhands")

        return self

    def _on_event(self, event: Any) -> None:
        """Handle an OpenHands event from the EventStream."""
        event_type = type(event).__name__

        # Record every event as a VCR frame (the VCRRuntime hook)
        try:
            event_data = (
                event.to_dict() if hasattr(event, "to_dict")
                else event.__dict__.copy() if hasattr(event, "__dict__")
                else {"raw": str(event)}
            )

            # Frame size guardrail (addresses issue #7402)
            frame_json = json.dumps(event_data, default=str)
            frame_size = len(frame_json.encode("utf-8"))

            if frame_size > self.config.max_frame_bytes:
                logger.warning(
                    "[sentinel] Oversized frame detected: %s is %d bytes (max %d). "
                    "This is OpenHands issue #7402 in action.",
                    event_type, frame_size, self.config.max_frame_bytes,
                )
                if self.config.truncate_large_frames:
                    event_data = {
                        "_truncated": True,
                        "_original_size_bytes": frame_size,
                        "_event_type": event_type,
                        "_summary": str(event_data)[:500],
                    }

            self.recorder.record_step(
                node_name=f"openhands:{event_type}",
                input_state=event_data,
                output_state={},
                metadata=FrameMetadata(custom={"event_type": event_type}),
            )
        except Exception:
            logger.warning("[sentinel] Failed to record event", exc_info=True)

        if not self.config.auto_analyze:
            return

        # Analyze file writes
        if event_type in ("FileWriteAction", "FileEditAction"):
            self._handle_file_event(event)
        elif event_type == "FileEditObservation":
            self._handle_edit_observation(event)

    @staticmethod
    def _event_path(event: Any) -> str | None:
        path = getattr(event, "path", None) or getattr(event, "file_path", None)
        return str(path) if path else None

    @staticmethod
    def _read_from_disk(file_path: str) -> str | None:
        """Read the post-edit file contents, or None if not readable on this host."""
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except OSError:
            logger.warning("[sentinel] Could not read %s from disk", file_path, exc_info=True)
            return None

    def _handle_file_event(self, event: Any) -> AnalysisResult | None:
        """Analyze a file write/edit action and optionally warn the agent.

        FileWriteAction carries the full ``content``. OpenHands' FileEditAction
        (OH_ACI mode) carries ``command``/``file_text``/``old_str``/``new_str``
        instead: ``create`` provides the full text in ``file_text``; for other
        commands (str_replace, insert, ...) the action is only a patch, so the
        file is analyzed once the edit has happened — when its
        FileEditObservation arrives (using ``new_content`` or reading the file
        from disk if the path exists).
        """
        file_path = self._event_path(event)
        if not file_path:
            return None

        content: str | None
        if type(event).__name__ == "FileEditAction":
            # ``content`` on an LLM-based edit is a draft snippet, not the file.
            content = getattr(event, "new_content", None) or getattr(event, "file_text", None)
            if not content:
                with self._pending_lock:
                    self._pending_edits.add(file_path)
                return None
        else:
            content = getattr(event, "content", None) or getattr(event, "new_content", None)
            if not content:
                content = self._read_from_disk(file_path)
        if not content:
            return None
        return self._analyze_and_warn(file_path, content)

    def _handle_edit_observation(self, event: Any) -> AnalysisResult | None:
        """Analyze the post-edit file for a FileEditAction that had no inline content."""
        file_path = self._event_path(event)
        if not file_path:
            return None
        with self._pending_lock:
            if file_path not in self._pending_edits:
                return None
            self._pending_edits.discard(file_path)
        content = getattr(event, "new_content", None) or self._read_from_disk(file_path)
        if not content:
            logger.warning("[sentinel] No post-edit content available for %s", file_path)
            return None
        return self._analyze_and_warn(file_path, content)

    def _analyze_and_warn(self, file_path: str, content: str) -> AnalysisResult:
        result, warning = self.check_and_warn(file_path, content)

        # Inject warning back into the EventStream
        if warning and self.config.warn_agent and self._event_stream:
            try:
                from openhands.events.action import MessageAction
            except ImportError:
                logger.info("[sentinel] OpenHands not available for warning injection")
                return result
            try:
                warning_action = MessageAction(content=warning, wait_for_response=False)
                self._event_stream.add_event(warning_action, self._warning_source())
                logger.info("[sentinel] Warning injected for %s", file_path)
            except Exception:
                logger.warning("[sentinel] Failed to inject warning", exc_info=True)
        return result

    @staticmethod
    def _warning_source() -> Any:
        """EventSource for injected warnings (USER, so the agent reads it as feedback)."""
        try:
            from openhands.events.event import EventSource
        except ImportError:
            try:
                from openhands.events import EventSource
            except ImportError:
                return "user"  # EventSource is a str enum; "user" == EventSource.USER
        return EventSource.USER

    # ──────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────

    def _record_analysis(
        self, file_path: str, result: AnalysisResult, elapsed_ms: float
    ) -> None:
        """Record an analysis result as a VCR frame."""
        self.recorder.record_step(
            node_name="sentinel:analysis",
            input_state={
                "file_path": file_path,
                "trigger": "file_write",
            },
            output_state={
                "passed": result.passed,
                "blocked": result.blocked,
                "violation_count": len(result.violations),
                "violations": [v.to_dict() for v in result.violations],
                "metrics": result.metrics,
            },
            metadata=FrameMetadata(
                latency_ms=elapsed_ms,
                custom={
                    "sentinel_check": True,
                    "file_path": file_path,
                    "blocker_count": result.blocker_count,
                    "critical_count": result.critical_count,
                    "blocked": result.blocked,
                },
            ),
            frame_type=FrameType.TOOL_CALL,
        )
