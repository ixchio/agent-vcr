"""Claude Code hook recorder for Agent VCR."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_vcr.models import Frame, FrameMetadata, FrameType, Session


def _session_id() -> str:
    return os.environ.get("AGENT_VCR_CLAUDE_SESSION_ID", "claude-code-hooks")


def _output_path() -> Path:
    output_dir = Path(os.environ.get("AGENT_VCR_DIR", ".vcr"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{_session_id()}.vcr"


def _frame_type(event: str) -> FrameType:
    if event in {"PreToolUse", "PostToolUse", "FileChanged"}:
        return FrameType.TOOL_CALL
    return FrameType.CHECKPOINT


def _summary(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": event,
        "tool_name": payload.get("tool_name") or payload.get("tool"),
        "file_path": payload.get("file_path") or payload.get("path"),
        "session_id": payload.get("session_id"),
        "cwd": payload.get("cwd"),
    }


def record_hook_event(event: str, payload: dict[str, Any]) -> Path:
    """Append one Claude Code hook event to a VCR-compatible JSONL file."""
    path = _output_path()
    session_id = _session_id()
    is_new = not path.exists() or path.stat().st_size == 0

    frame = Frame(
        session_id=session_id,
        frame_type=_frame_type(event),
        node_name=f"claude:{event}",
        input_state={"event": event, "payload": payload},
        output_state=_summary(event, payload),
        metadata=FrameMetadata(
            latency_ms=0.0,
            custom={
                "integration": "claude_code",
                "hook_event": event,
            },
        ),
    )

    with open(path, "a") as f:
        if is_new:
            session = Session(
                session_id=session_id,
                metadata={
                    "integration": "claude_code",
                    "mode": "hooks",
                    "created_by": "vcr init --claude-code",
                },
                tags=["claude-code", "hooks"],
            )
            f.write(
                json.dumps(
                    {"type": "session", "data": session.model_dump()},
                    separators=(",", ":"),
                )
                + "\n"
            )
        f.write(
            json.dumps(
                {"type": "frame", "data": frame.model_dump()},
                separators=(",", ":"),
            )
            + "\n"
        )

    return path


def main(argv: list[str] | None = None) -> None:
    """Record a Claude Code hook invocation from stdin."""
    args = argv if argv is not None else sys.argv[1:]
    event = args[0] if args else "Unknown"
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {
            "raw": sys.stdin.read(),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
    if not isinstance(payload, dict):
        payload = {"value": payload}

    record_hook_event(event, payload)


if __name__ == "__main__":
    main()
