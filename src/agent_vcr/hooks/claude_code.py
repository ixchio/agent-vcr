"""Claude Code hook recorder for Agent VCR.

Invoked by Claude Code as a ``command`` hook: the event JSON arrives on stdin
(``hook_event_name``, ``session_id``, ``cwd``, ``tool_name``, ``tool_input``,
``tool_response``, ...). Each invocation appends one frame to
``$AGENT_VCR_DIR/claude-<session_id>.vcr``.

Design constraints for a hook that runs on every tool call:

* fast startup — only stdlib + ``agent_vcr.models`` are imported;
* never blocks or breaks Claude Code — always exits 0, errors go to stderr;
* one file per Claude session, safe under concurrent hook processes
  (header created with ``O_EXCL``, each frame is a single ``O_APPEND`` write);
* bounded size — very large string fields (file bodies, tool output) are
  truncated before being stored.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_vcr.models import Frame, FrameMetadata, FrameType, Session, json_dumps_safe

#: Never text mode on Windows (no CRLF translation of the JSONL file).
_O_BINARY = getattr(os, "O_BINARY", 0)

#: Strings longer than this (characters) are truncated in recorded payloads.
MAX_FIELD_CHARS = int(os.environ.get("AGENT_VCR_HOOK_MAX_FIELD_CHARS", "8192"))

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _session_id(payload: dict[str, Any] | None = None) -> str:
    """Explicit env override, else the Claude Code session id, else a fixed fallback."""
    override = os.environ.get("AGENT_VCR_CLAUDE_SESSION_ID")
    if override:
        return _SAFE_ID.sub("_", override).strip("._") or "claude-code-hooks"
    raw = (payload or {}).get("session_id")
    if isinstance(raw, str) and raw.strip():
        return "claude-" + (_SAFE_ID.sub("_", raw).strip("._")[:120] or "session")
    return "claude-code-hooks"


def _output_path(payload: dict[str, Any] | None = None) -> Path:
    output_dir = Path(os.environ.get("AGENT_VCR_DIR", ".vcr"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{_session_id(payload)}.vcr"


def _frame_type(event: str) -> FrameType:
    if event in {"PreToolUse", "PostToolUse", "PostToolUseFailure", "FileChanged"}:
        return FrameType.TOOL_CALL
    return FrameType.CHECKPOINT


def _truncate(value: Any, limit: int = MAX_FIELD_CHARS) -> Any:
    """Recursively shorten oversized strings so recordings stay bounded."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated {len(value) - limit} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value]
    return value


def _summary(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    tool_input = payload.get("tool_input")
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    return {
        "event": event,
        "tool_name": payload.get("tool_name") or payload.get("tool"),
        "file_path": (
            tool_input.get("file_path")
            or tool_input.get("notebook_path")
            or tool_input.get("path")
            or payload.get("file_path")
            or payload.get("path")
        ),
        "command": tool_input.get("command"),
        "session_id": payload.get("session_id"),
        "cwd": payload.get("cwd"),
    }


def _write_header_once(path: Path, session_id: str, payload: dict[str, Any]) -> None:
    """Create the file with its session header exactly once, even under races."""
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, 0o600)
    except FileExistsError:
        return
    session = Session(
        session_id=session_id,
        metadata={
            "integration": "claude_code",
            "mode": "hooks",
            "created_by": "vcr init --claude-code",
            "claude_session_id": payload.get("session_id"),
            "cwd": payload.get("cwd"),
        },
        tags=["claude-code", "hooks"],
    )
    line = json_dumps_safe({"type": "session", "data": session.model_dump()}) + "\n"
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def record_hook_event(event: str, payload: dict[str, Any]) -> Path:
    """Append one Claude Code hook event to a VCR-compatible JSONL file."""
    event = str(payload.get("hook_event_name") or event or "Unknown")
    path = _output_path(payload)
    session_id = path.stem
    _write_header_once(path, session_id, payload)

    frame = Frame(
        session_id=session_id,
        frame_type=_frame_type(event),
        node_name=f"claude:{event}",
        input_state={"event": event, "payload": _truncate(payload)},
        output_state=_summary(event, payload),
        metadata=FrameMetadata(
            latency_ms=0.0,
            error_message=(
                str(payload.get("error"))[:500] if event == "PostToolUseFailure" else None
            ),
            custom={
                "integration": "claude_code",
                "hook_event": event,
                "tool_use_id": payload.get("tool_use_id"),
            },
        ),
    )
    line = json_dumps_safe({"type": "frame", "data": frame.model_dump()}) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT | _O_BINARY, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))  # single O_APPEND write: no interleaving
    finally:
        os.close(fd)
    return path


def _read_payload(stream: Any) -> dict[str, Any]:
    # Claude Code sends UTF-8 on every platform (not the Windows code page).
    binary = getattr(stream, "buffer", None)
    raw = binary.read().decode("utf-8", "replace") if binary is not None else stream.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {"raw": raw[:MAX_FIELD_CHARS], "received_at": datetime.now(timezone.utc).isoformat()}
    return payload if isinstance(payload, dict) else {"value": payload}


def main(argv: list[str] | None = None) -> int:
    """Record a Claude Code hook invocation from stdin. Never fails the hook."""
    args = argv if argv is not None else sys.argv[1:]
    event = args[0] if args else "Unknown"
    try:
        record_hook_event(event, _read_payload(sys.stdin))
    except Exception as exc:  # a recorder must never break the agent it observes
        print(f"agent-vcr hook: failed to record {event}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
