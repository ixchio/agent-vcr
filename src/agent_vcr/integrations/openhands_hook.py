"""
VCRRuntime — record OpenHands tool calls as VCR frames.

Works with both OpenHands APIs, detected at runtime:

* **OpenHands SDK 1.x** (``openhands-sdk``, verified against 1.7.3) — pass the
  runtime as a conversation callback. Each tool call becomes **one frame**: the
  ``ActionEvent`` (tool, arguments) paired with its result
  (``ObservationEvent``, ``UserRejectObservation`` or ``AgentErrorEvent``):
  output text, exit code, error and latency::

      from openhands.sdk import Conversation
      from agent_vcr.integrations.openhands_hook import VCRRuntime

      vcr = VCRRuntime(session_id="my-task")
      conversation = Conversation(agent=agent, workspace=ws, callbacks=[vcr])
      conversation.send_message("...")
      conversation.run()
      vcr.save()  # .vcr/my-task.vcr

* **Legacy OpenHands V0** (``openhands.events`` EventStream) — ``vcr.attach(
  runtime.event_stream)`` records every event on the stream as its own frame.

Both paths apply the frame-size guardrail: frames over ``max_frame_bytes`` are
logged and (by default) replaced by a short preview.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agent_vcr import VCRRecorder
from agent_vcr.models import FrameMetadata, FrameType

logger = logging.getLogger(__name__)

#: OpenHands SDK 1.x events that answer an ``ActionEvent`` (all carry ``tool_call_id``).
SDK_RESULT_EVENTS = frozenset({"ObservationEvent", "UserRejectObservation", "AgentErrorEvent"})
_PREVIEW_CHARS = 1000


def is_sdk_event(event: Any) -> bool:
    """True for OpenHands SDK 1.x tool-call events (by class name, so fakes work)."""
    name = type(event).__name__
    return name == "ActionEvent" or name in SDK_RESULT_EVENTS


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            try:
                return obj.model_dump()
            except Exception:  # noqa: BLE001
                pass
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _action_args(event: Any) -> dict[str, Any]:
    """Arguments of an ``ActionEvent``; raw tool-call arguments if validation failed."""
    action = getattr(event, "action", None)
    if action is not None:
        dumped = _dump(action)
        if isinstance(dumped, dict):
            dumped.pop("kind", None)
            return dumped
        return {"raw": dumped}
    raw = getattr(getattr(event, "tool_call", None), "arguments", None)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {}


def _thought(event: Any) -> str:
    parts = getattr(event, "thought", None) or []
    return "".join(str(getattr(p, "text", "")) for p in parts)


def _observation_text(observation: Any) -> str:
    text = getattr(observation, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(observation, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(getattr(c, "text", "")) for c in content)
    return "" if observation is None else str(observation)


@dataclass
class _PendingAction:
    tool_name: str
    action_id: str | None
    tool_call_id: str | None
    args: dict[str, Any]
    thought: str
    emitted_at: float
    extra: dict[str, Any] = field(default_factory=dict)


class VCRRuntime:
    """Record OpenHands tool calls into a :class:`VCRRecorder` session.

    Args:
        recorder: Where frames go (default: a new ``VCRRecorder()`` writing ``.vcr/``).
            If the recorder already has an active session (e.g. shared with
            Sentinel), frames are appended to it.
        session_id: Session to start when the recorder has none.
        max_frame_bytes: Frame-size guardrail (OpenHands issue #7402 pattern).
        truncate_oversized: Replace oversized frames by a preview (else only warn).

    A recording problem is logged and never raised into the agent run.
    """

    def __init__(
        self,
        recorder: VCRRecorder | None = None,
        session_id: str | None = None,
        max_frame_bytes: int = 50_000,
        truncate_oversized: bool = True,
    ) -> None:
        self.recorder = recorder or VCRRecorder()
        self.session_id = session_id or "openhands-vcr"
        self.max_frame_bytes = max_frame_bytes
        self.truncate_oversized = truncate_oversized

        self._frame_count = 0
        self._oversized_count = 0
        self._total_bytes_recorded = 0
        self._attached = False
        self._session_ready = False
        self._lock = threading.RLock()
        self._pending: dict[str, _PendingAction] = {}
        self._last_result_at = 0.0

    # ──────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────

    def __call__(self, event: Any) -> None:
        """OpenHands SDK conversation callback: ``Conversation(callbacks=[vcr])``."""
        self.record_event(event)

    def attach(self, event_stream: Any) -> VCRRuntime:
        """Attach to a legacy OpenHands V0 ``EventStream`` (one frame per event).

        OpenHands SDK 1.x conversations take callbacks only at construction, so
        pass the runtime as ``Conversation(..., callbacks=[vcr])`` instead.
        """
        if not hasattr(event_stream, "subscribe"):
            if hasattr(event_stream, "state") and hasattr(event_stream, "send_message"):
                raise TypeError(
                    "OpenHands SDK conversations cannot be attached after creation: "
                    "pass the runtime as Conversation(..., callbacks=[vcr_runtime])."
                )
            raise TypeError(
                f"Cannot attach to {type(event_stream).__name__}: missing subscribe() method"
            )

        self._start_session(mode="vcr_runtime")
        try:
            from openhands.events.stream import EventStreamSubscriber

            subscriber_id: Any = EventStreamSubscriber.RUNTIME
        except ImportError:
            subscriber_id = "vcr_runtime"
        event_stream.subscribe(subscriber_id, self._on_event, self.session_id)
        self._attached = True
        logger.info("[vcr-runtime] Attached to EventStream (session=%s)", self.session_id)
        return self

    def record_event(self, event: Any) -> None:
        """Record one event. SDK 1.x tool-call events are paired; others are V0-style."""
        if is_sdk_event(event):
            self._on_sdk_event(event)
        elif type(event).__module__.startswith("openhands.sdk"):
            return  # SDK message/state/system events: not tool calls
        else:
            self._on_event(event)

    def flush_pending(self) -> int:
        """Record actions that never got a result (e.g. the tool crashed). Returns the count."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            self._record_pair(
                item,
                output={"status": "no_observation", "output": None},
                error=("NoObservation", "no observation was received for this action"),
                finished_at=time.perf_counter(),
            )
        return len(pending)

    def save(self) -> str:
        """Record unanswered actions, save the recording and return the file path."""
        self._start_session(mode="vcr_runtime")
        self.flush_pending()
        path = self.recorder.save()
        logger.info(
            "[vcr-runtime] Session saved: %d frames, %d oversized, %.1f KB total — %s",
            self._frame_count,
            self._oversized_count,
            self._total_bytes_recorded / 1024,
            path,
        )
        return str(path)

    @property
    def stats(self) -> dict[str, Any]:
        """Get recording statistics."""
        with self._lock:
            pending = len(self._pending)
        return {
            "frame_count": self._frame_count,
            "oversized_frames": self._oversized_count,
            "total_bytes": self._total_bytes_recorded,
            "attached": self._attached,
            "pending_actions": pending,
            "session_id": self.session_id,
        }

    # ──────────────────────────────────────────────
    #  OpenHands SDK 1.x
    # ──────────────────────────────────────────────

    def _on_sdk_event(self, event: Any) -> None:
        try:
            now = time.perf_counter()
            name = type(event).__name__
            key = self._pair_key(event)
            if name == "ActionEvent":
                raw_id = getattr(event, "id", None)
                item = _PendingAction(
                    tool_name=str(getattr(event, "tool_name", "") or "unknown"),
                    action_id=str(raw_id) if raw_id else None,
                    tool_call_id=key,
                    args=_action_args(event),
                    thought=_thought(event),
                    emitted_at=now,
                )
                if getattr(event, "action", None) is None:
                    item.extra["invalid_tool_call"] = True
                with self._lock:
                    self._pending[key or f"action-{id(event)}"] = item
                return
            with self._lock:
                item_or_none = self._pending.pop(key, None) if key else None
            if item_or_none is None:
                item_or_none = _PendingAction(
                    tool_name=str(getattr(event, "tool_name", "") or "unknown"),
                    action_id=None,
                    tool_call_id=key,
                    args={},
                    thought="",
                    emitted_at=now,
                    extra={"unpaired": True},
                )
            output, error = self._describe_result(event)
            self._record_pair(item_or_none, output=output, error=error, finished_at=now)
        except Exception:  # noqa: BLE001 - a callback exception would abort the agent run
            logger.warning("[vcr-runtime] Failed to record %s", type(event).__name__,
                           exc_info=True)

    @staticmethod
    def _pair_key(event: Any) -> str | None:
        raw = getattr(event, "tool_call_id", None)
        return str(raw) if raw else None

    @staticmethod
    def _describe_result(event: Any) -> tuple[dict[str, Any], tuple[str, str] | None]:
        name = type(event).__name__
        if name == "UserRejectObservation":
            reason = str(getattr(event, "rejection_reason", "") or "")
            return ({"status": "rejected", "output": None, "rejection_reason": reason},
                    ("Rejected", reason))
        if name == "AgentErrorEvent":
            message = str(getattr(event, "error", "") or "")
            return {"status": "error", "output": None, "error": message}, ("AgentError", message)
        observation = getattr(event, "observation", None)
        text = _observation_text(observation)
        details = _dump(observation)
        if isinstance(details, dict):
            details.pop("kind", None)
            details.pop("content", None)  # same as ``output``
        is_error = bool(getattr(observation, "is_error", False))
        output: dict[str, Any] = {
            "status": "error" if is_error else "ok",
            "output": text,
            "exit_code": getattr(observation, "exit_code", None),
            "observation": details,
        }
        return output, (("ToolError", text[:500]) if is_error else None)

    def _record_pair(
        self,
        item: _PendingAction,
        *,
        output: dict[str, Any],
        error: tuple[str, str] | None,
        finished_at: float,
    ) -> None:
        self._start_session(mode="sdk_callback")
        with self._lock:
            # Actions of one LLM response run one after another: a tool starts
            # when it was emitted or when the previous tool finished, whichever is later.
            started = max(item.emitted_at, self._last_result_at)
            self._last_result_at = finished_at
            frame_number = self._frame_count
            self._frame_count += 1
        latency_ms = max(0.0, (finished_at - started) * 1000)

        input_state: dict[str, Any] = {
            "tool": item.tool_name,
            "args": item.args,
            "tool_call_id": item.tool_call_id,
            "action_id": item.action_id,
        }
        if item.thought:
            input_state["thought"] = item.thought
        input_state.update(item.extra)

        size = len(json.dumps({"input": input_state, "output": output}, default=str)
                   .encode("utf-8"))
        self._total_bytes_recorded += size
        truncated = False
        if size > self.max_frame_bytes:
            self._oversized_count += 1
            logger.warning(
                "[vcr-runtime] Oversized frame: %s is %d bytes (max %d). Frame #%d. "
                "This is the issue #7402 pattern.",
                item.tool_name, size, self.max_frame_bytes, frame_number,
            )
            if self.truncate_oversized:
                truncated = True
                input_state = {
                    "tool": item.tool_name,
                    "tool_call_id": item.tool_call_id,
                    "action_id": item.action_id,
                    "_vcr_truncated": True,
                    "_original_size_bytes": size,
                    "_preview": json.dumps(item.args, default=str)[:_PREVIEW_CHARS],
                }
                output = {
                    "status": output.get("status"),
                    "exit_code": output.get("exit_code"),
                    "_vcr_truncated": True,
                    "_preview": str(output.get("output") or output)[:_PREVIEW_CHARS],
                }

        self.recorder.record_step(
            node_name=f"openhands:{item.tool_name}",
            input_state=input_state,
            output_state=output,
            metadata=FrameMetadata(
                latency_ms=latency_ms,
                error_type=error[0] if error else None,
                error_message=error[1] if error else None,
                custom={
                    "integration": "openhands-sdk",
                    "frame_number": frame_number,
                    "frame_size_bytes": size,
                    "truncated": truncated,
                },
            ),
            frame_type=FrameType.ERROR if output.get("status") == "error" else FrameType.TOOL_CALL,
        )

    def _start_session(self, mode: str) -> None:
        with self._lock:
            if self._session_ready:
                return
            if self.recorder.get_session() is None:
                self.recorder.start_session(
                    session_id=self.session_id,
                    metadata={
                        "integration": "openhands",
                        "mode": mode,
                        "max_frame_bytes": self.max_frame_bytes,
                    },
                    tags=["openhands", "vcr-runtime"],
                )
            self._session_ready = True

    # ──────────────────────────────────────────────
    #  Legacy OpenHands V0 EventStream
    # ──────────────────────────────────────────────

    def _on_event(self, event: Any) -> None:
        """Handle an incoming event from a V0 EventStream (one frame per event)."""
        start = time.perf_counter()
        event_type = type(event).__name__

        try:
            self._start_session(mode="vcr_runtime")
            event_data = self._extract_event_data(event)

            # Frame size check (issue #7402 guardrail)
            frame_json = json.dumps(event_data, default=str)
            frame_size = len(frame_json.encode("utf-8"))
            self._total_bytes_recorded += frame_size

            if frame_size > self.max_frame_bytes:
                self._oversized_count += 1
                logger.warning(
                    "[vcr-runtime] Oversized frame: %s is %d bytes (max %d). "
                    "Frame #%d. This is the issue #7402 pattern.",
                    event_type, frame_size, self.max_frame_bytes, self._frame_count,
                )

                if self.truncate_oversized:
                    event_data = {
                        "_vcr_truncated": True,
                        "_original_size_bytes": frame_size,
                        "_event_type": event_type,
                        "_preview": str(event_data)[:_PREVIEW_CHARS],
                    }

            frame_type = self._classify_event(event_type)
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.recorder.record_step(
                node_name=f"openhands:{event_type}",
                input_state=event_data,
                output_state={},
                metadata=FrameMetadata(
                    latency_ms=elapsed_ms,
                    custom={
                        "event_type": event_type,
                        "frame_number": self._frame_count,
                        "frame_size_bytes": frame_size,
                    },
                ),
                frame_type=frame_type,
            )

            self._frame_count += 1

        except Exception as e:  # noqa: BLE001
            logger.warning("[vcr-runtime] Failed to record event %s: %s", event_type, e)

    def _extract_event_data(self, event: Any) -> dict[str, Any]:
        """Extract serializable data from an event object."""
        if hasattr(event, "to_dict"):
            data = event.to_dict()
        elif hasattr(event, "model_dump"):
            data = event.model_dump()
        elif hasattr(event, "__dict__"):
            data = {k: v for k, v in event.__dict__.items() if not k.startswith("_")}
        else:
            data = {"raw": str(event)}
        return data if isinstance(data, dict) else {"raw": data}

    def _classify_event(self, event_type: str) -> FrameType:
        """Map OpenHands event types to VCR frame types."""
        if "Error" in event_type:
            return FrameType.ERROR
        elif "Action" in event_type:
            return FrameType.TOOL_CALL
        return FrameType.NODE_EXECUTION
