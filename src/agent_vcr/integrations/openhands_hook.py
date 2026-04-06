"""
VCRRuntime — Native OpenHands EventStream hook for agent-vcr.

This is the bridge between agent-vcr's recording engine and
OpenHands' event system. Attach it to any OpenHands runtime
and every action/observation becomes a VCR frame automatically.

Usage:
    from agent_vcr.integrations.openhands_hook import VCRRuntime
    from agent_vcr import VCRRecorder

    recorder = VCRRecorder()
    vcr = VCRRuntime(recorder=recorder)
    vcr.attach(runtime.event_stream)

    # ... run your OpenHands task ...

    vcr.save()  # session saved as .vcr file
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent_vcr import VCRRecorder
from agent_vcr.models import FrameMetadata, FrameType

logger = logging.getLogger(__name__)


class VCRRuntime:
    """
    Attaches agent-vcr recording directly to the OpenHands EventStream.

    Every action (CmdRun, FileWrite, BrowseURL) and every observation
    becomes a VCR frame. No manual instrumentation needed.

    Also implements frame size guardrails — detects and warns about
    oversized frames (the OpenHands issue #7402 pattern) before they
    pollute the recording.
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

    def attach(self, event_stream: Any) -> "VCRRuntime":
        """
        Attach to an OpenHands EventStream.

        Args:
            event_stream: An OpenHands EventStream instance.

        Returns:
            self (for chaining)
        """
        self.recorder.start_session(
            session_id=self.session_id,
            metadata={
                "integration": "openhands",
                "mode": "vcr_runtime",
                "max_frame_bytes": self.max_frame_bytes,
            },
            tags=["openhands", "vcr-runtime"],
        )

        # Subscribe using the OpenHands pattern
        try:
            from openhands.events.stream import EventStreamSubscriber
            subscriber_id = EventStreamSubscriber.RUNTIME
        except ImportError:
            subscriber_id = "vcr_runtime"

        if hasattr(event_stream, "subscribe"):
            event_stream.subscribe(subscriber_id, self._on_event, self.session_id)
            self._attached = True
            logger.info(
                "[vcr-runtime] Attached to EventStream (session=%s)",
                self.session_id,
            )
        else:
            raise TypeError(
                f"Cannot attach to {type(event_stream).__name__}: "
                f"missing subscribe() method"
            )

        return self

    def record_event(self, event: Any) -> None:
        """
        Manually record an event (for use outside EventStream).

        Args:
            event: Any event-like object with to_dict() or __dict__.
        """
        self._on_event(event)

    def save(self) -> str:
        """Save the recording and return the file path."""
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
        return {
            "frame_count": self._frame_count,
            "oversized_frames": self._oversized_count,
            "total_bytes": self._total_bytes_recorded,
            "attached": self._attached,
            "session_id": self.session_id,
        }

    # ──────────────────────────────────────────────
    #  Internal
    # ──────────────────────────────────────────────

    def _on_event(self, event: Any) -> None:
        """Handle an incoming event from the EventStream."""
        start = time.perf_counter()
        event_type = type(event).__name__

        try:
            # Extract event data
            event_data = self._extract_event_data(event)

            # Frame size check (issue #7402 guardrail)
            frame_json = json.dumps(event_data, default=str)
            frame_size = len(frame_json.encode("utf-8"))
            self._total_bytes_recorded += frame_size

            if frame_size > self.max_frame_bytes:
                self._oversized_count += 1
                logger.warning(
                    "[vcr-runtime] ⚠️ Oversized frame: %s is %d bytes (max %d). "
                    "Frame #%d. This is the issue #7402 pattern.",
                    event_type, frame_size, self.max_frame_bytes, self._frame_count,
                )

                if self.truncate_oversized:
                    event_data = {
                        "_vcr_truncated": True,
                        "_original_size_bytes": frame_size,
                        "_event_type": event_type,
                        "_preview": str(event_data)[:1000],
                    }

            # Determine frame type
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

        except Exception as e:
            logger.warning("[vcr-runtime] Failed to record event %s: %s", event_type, e)

    def _extract_event_data(self, event: Any) -> dict:
        """Extract serializable data from an event object."""
        if hasattr(event, "to_dict"):
            return event.to_dict()
        elif hasattr(event, "model_dump"):
            return event.model_dump()
        elif hasattr(event, "__dict__"):
            return {
                k: v for k, v in event.__dict__.items()
                if not k.startswith("_")
            }
        else:
            return {"raw": str(event)}

    def _classify_event(self, event_type: str) -> FrameType:
        """Map OpenHands event types to VCR frame types."""
        if "Error" in event_type:
            return FrameType.ERROR
        elif "Action" in event_type:
            return FrameType.TOOL_CALL
        elif "Observation" in event_type:
            return FrameType.NODE_EXECUTION
        else:
            return FrameType.NODE_EXECUTION
