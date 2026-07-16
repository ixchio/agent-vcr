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

    # Or attach to OpenHands EventStream (native mode)
    sentinel.attach(event_stream)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
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
    block_on_blocker: bool = True  # Halt agent on BLOCKER violations
    warn_agent: bool = True  # Inject warnings into EventStream
    record_to_vcr: bool = True  # Record checks as VCR frames
    auto_analyze: bool = True  # Auto-analyze on file write events

    # Frame size guardrail (addresses OpenHands issue #7402)
    max_frame_bytes: int = 50_000  # Warn if any recorded frame > 50KB
    truncate_large_frames: bool = True


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
    3. Native — sentinel.attach(event_stream) hooks into OpenHands
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
        self._event_stream = None
        self._results_history: list[AnalysisResult] = []

    def start_session(self, session_id: str | None = None) -> None:
        """Start a Sentinel monitoring session."""
        self.recorder.start_session(
            session_id=session_id or "sentinel-session",
            metadata={"mode": "sentinel", "config": self.config.__dict__},
            tags=["sentinel", "quality-guard"],
        )
        self._session_started = True
        logger.info("[sentinel] Session started")

    def check_file(self, file_path: str, content: str) -> AnalysisResult:
        """
        Analyze a single file and record the result.

        This is the core method. Everything else calls this.
        """
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

        return result

    def check_and_warn(self, file_path: str, content: str) -> tuple[AnalysisResult, str | None]:
        """
        Analyze a file and generate an agent-facing warning string.

        Returns:
            Tuple of (AnalysisResult, warning_message_or_None)
        """
        result = self.check_file(file_path, content)

        if not result.violations:
            return result, None

        # Build a concise warning message for the agent
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
    #  OpenHands EventStream Integration
    # ──────────────────────────────────────────────

    def attach(self, event_stream: Any) -> Sentinel:
        """
        Attach to an OpenHands EventStream for native integration.

        Once attached, Sentinel will automatically analyze every
        FileWriteAction and FileEditAction that flows through the stream.

        Args:
            event_stream: An OpenHands EventStream instance.

        Returns:
            self (for chaining)
        """
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
            logger.debug("[sentinel] Failed to record event", exc_info=True)

        # Analyze file writes
        if event_type in ("FileWriteAction", "FileEditAction"):
            self._handle_file_event(event)

    def _handle_file_event(self, event: Any) -> None:
        """Analyze a file write/edit event and optionally warn the agent."""
        file_path = getattr(event, "path", None) or getattr(event, "file_path", None)
        content = getattr(event, "content", None) or getattr(event, "new_content", None)

        if not file_path or not content:
            return

        result, warning = self.check_and_warn(file_path, content)

        # Inject warning back into the EventStream
        if warning and self.config.warn_agent and self._event_stream:
            try:
                from openhands.events.action import MessageAction
                warning_action = MessageAction(content=warning, wait_for_response=False)
                self._event_stream.add_event(warning_action, "sentinel")
                logger.info("[sentinel] Warning injected for %s", file_path)
            except ImportError:
                logger.info("[sentinel] OpenHands not available for warning injection")
            except Exception:
                logger.debug("[sentinel] Failed to inject warning", exc_info=True)

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
                },
            ),
            frame_type=FrameType.TOOL_CALL,
        )
