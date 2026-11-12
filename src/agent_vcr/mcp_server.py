"""``vcr mcp`` — a stdio Model Context Protocol server for Agent VCR.

Any MCP-capable agent (Claude Code, Codex CLI, Cursor, Gemini CLI, ...) can
start ``vcr mcp`` and then see and drive its own safety net: protection
status, the airbag timeline, "would this command be stopped?", manual
snapshots, undo / rewind (preview by default, restore only with explicit
confirmation), and summaries of ``.vcr`` recordings.

Standard library only. Transport: newline-delimited JSON-RPC 2.0 on
stdin/stdout (MCP stdio transport). Nothing but protocol JSON is ever written
to stdout; logs go to stderr. Requests are handled one at a time, in order.

Safety: tools never grant permissions or install anything. ``airbag_undo``
and ``airbag_rewind`` only preview unless called with ``dry_run: false`` and
``confirm: true``, and every restore is itself snapshotted first (``vcr redo``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable

from agent_vcr import __version__
from agent_vcr.airbag.cli import LEVEL_EXPLAIN, NOTHING_TO_UNDO, _ago, _summary
from agent_vcr.airbag.core import Airbag, Policy
from agent_vcr.airbag.store import AirbagError, RestoreResult, Snapshot

logger = logging.getLogger("agent_vcr.mcp")

SERVER_NAME = "agent-vcr"
#: Newest first. A client asking for one of these gets it echoed back;
#: anything else gets the newest (MCP version negotiation).
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

INSTRUCTIONS = (
    "Agent VCR airbag: every tool call you make in this project may be snapshotted so the user "
    "can undo it. Call airbag_check before a shell command that might delete or overwrite data. "
    "Use airbag_snapshot before a risky multi-step change. Only restore files with "
    "airbag_undo / airbag_rewind (dry_run=false, confirm=true) when the user explicitly asked; "
    "otherwise preview."
)

#: Directories ``recordings_list`` never descends into.
SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".nox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "site-packages",
})
SCAN_MAX_ENTRIES = 200_000
SCAN_MAX_SECONDS = 10.0
MAX_RECORDING_BYTES = 512 * 1024 * 1024
MAX_CHANGES_SHOWN = 200

AGENT_ADVICE = {
    "read": "Safe: nothing to snapshot.",
    "write": "Changes files; the airbag snapshots first and `vcr undo` can bring them back.",
    "destructive": "Destroys project data, but the airbag snapshots first so it can be undone. "
                   "Make sure this is what the user wants.",
    "irreversible": "A snapshot cannot undo this. Explain what it does and get the user's "
                    "explicit go-ahead before running it; the airbag will stop it otherwise.",
}


class ToolError(Exception):
    """A tool execution error, reported to the model as an ``isError`` result."""

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data


class ProtocolError(Exception):
    """A JSON-RPC error response."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class Tool:
    name: str
    title: str
    description: str
    properties: dict[str, dict[str, Any]]
    handler: Callable[[MCPServer, dict[str, Any]], dict[str, Any]]
    required: tuple[str, ...] = ()
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = True

    def definition(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object", "properties": self.properties, "additionalProperties": False,
        }
        if self.required:
            schema["required"] = list(self.required)
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": schema,
            "annotations": {
                "title": self.title,
                "readOnlyHint": self.read_only,
                "destructiveHint": self.destructive,
                "idempotentHint": self.idempotent,
                "openWorldHint": False,
            },
        }


# ── argument handling ────────────────────────────────────────────────────

_PROJECT_DIR = {
    "type": "string",
    "description": "Project directory (absolute, or relative to the server's project). "
                   "Default: the directory `vcr mcp` was started for.",
}


def _validate(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Check ``args`` against the tool's (flat) schema; fill defaults."""
    unknown = sorted(set(args) - set(tool.properties))
    if unknown:
        raise ToolError(f"unknown argument(s) {', '.join(unknown)}; "
                        f"expected: {', '.join(tool.properties) or 'none'}")
    for name in tool.required:
        if args.get(name) is None:
            raise ToolError(f"missing required argument {name!r}")
    out: dict[str, Any] = {}
    for name, spec in tool.properties.items():
        value = args.get(name, spec.get("default"))
        if value is None:
            out[name] = None
            continue
        kind = spec.get("type")
        if kind == "string" and not isinstance(value, str):
            raise ToolError(f"{name} must be a string")
        if kind == "boolean" and not isinstance(value, bool):
            raise ToolError(f"{name} must be true or false (a JSON boolean)")
        if kind == "integer":
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolError(f"{name} must be an integer")
            low, high = spec.get("minimum"), spec.get("maximum")
            if low is not None and value < low:
                raise ToolError(f"{name} must be at least {low}")
            if high is not None and value > high:
                raise ToolError(f"{name} must be at most {high}")
        if kind == "string" and spec.get("minLength") and not str(value).strip():
            raise ToolError(f"{name} must not be empty")
        out[name] = value
    return out


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _clip(text: Any, limit: int = 500) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _snap_info(snap: Snapshot) -> dict[str, Any]:
    return {"snapshot": snap.seq, "time": _iso(snap.time), "ago": _ago(snap.time),
            "label": snap.label, "files": snap.files,
            "agent": snap.meta.get("agent"), "command": snap.meta.get("command")}


def _action_info(action: dict[str, Any]) -> dict[str, Any]:
    ts = float(action.get("time", 0) or 0)
    return {
        "snapshot": action.get("snapshot"),
        "time": _iso(ts), "ago": _ago(ts),
        "agent": action.get("agent"), "tool": action.get("tool"),
        "level": action.get("level"),
        "command": _clip(action.get("command") or ""),
        "paths": list(action.get("paths") or [])[:20],
        "decision": action.get("decision"),
        "reasons": list(action.get("reasons") or []),
    }


def _changes(changes: list[tuple[str, str]]) -> dict[str, Any]:
    word = {"A": "restore", "D": "remove"}
    items = [{"change": word.get(status, "rewrite"), "path": rel}
             for status, rel in changes[:MAX_CHANGES_SHOWN]]
    return {"change_count": len(changes), "changes": items,
            "changes_truncated": len(changes) > MAX_CHANGES_SHOWN}


def _restore_info(result: RestoreResult) -> dict[str, Any]:
    return {"snapshot": result.target.seq, "summary": _summary(result),
            "written": result.written, "deleted": result.deleted,
            "failed": result.failed, "kept_unprotected": result.kept,
            "before_snapshot": result.before.seq,
            "redo": "The user can go forward again with `vcr redo`."}


# ── the server ───────────────────────────────────────────────────────────

class MCPServer:
    """Protocol handling, independent of the transport (see :func:`serve`)."""

    def __init__(self, project_dir: str | Path = ".") -> None:
        self.project_dir = Path(os.path.abspath(Path(project_dir).expanduser()))
        self.client_info: dict[str, Any] = {}
        self.protocol_version = LATEST_PROTOCOL_VERSION
        self.tools = {tool.name: tool for tool in TOOLS}

    # helpers used by tools

    def resolve(self, raw: str | None) -> Path:
        if not raw:
            return self.project_dir
        path = Path(raw).expanduser()
        return Path(os.path.abspath(path if path.is_absolute() else self.project_dir / path))

    def bag(self, args: dict[str, Any]) -> Airbag:
        try:
            return Airbag(self.resolve(args.get("project_dir")), policy=Policy.from_env())
        except AirbagError as err:
            raise ToolError(str(err)) from None

    # JSON-RPC

    def handle(self, message: Any) -> dict[str, Any] | None:
        """Answer one decoded message; ``None`` for notifications and responses."""
        if not isinstance(message, dict):
            return _error(None, INVALID_REQUEST,
                          "expected a JSON-RPC 2.0 object (batches are not supported)")
        msg_id = message.get("id")
        has_id = "id" in message
        method = message.get("method")
        if method is None and ("result" in message or "error" in message):
            return None  # a response to a request we never send
        if (message.get("jsonrpc") != "2.0" or not isinstance(method, str)
                or (has_id and (isinstance(msg_id, bool)
                                or not isinstance(msg_id, (str, int))))):
            return _error(msg_id if isinstance(msg_id, (str, int)) else None,
                          INVALID_REQUEST, "invalid JSON-RPC 2.0 request")
        params = message.get("params") or {}
        if not has_id:
            self._notification(method, params)
            return None
        try:
            if not isinstance(params, dict):
                raise ProtocolError(INVALID_PARAMS, "params must be an object")
            return {"jsonrpc": "2.0", "id": msg_id, "result": self._request(method, params)}
        except ProtocolError as err:
            return _error(msg_id, err.code, err.message, err.data)
        except Exception as err:  # noqa: BLE001 - the server must keep serving
            logger.exception("internal error handling %s", method)
            return _error(msg_id, INTERNAL_ERROR, f"internal error: {err}")

    def _notification(self, method: str, params: Any) -> None:
        if method == "notifications/initialized":
            logger.info("client initialized")
        else:
            logger.debug("ignoring notification %s", method)

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [tool.definition() for tool in self.tools.values()]}
        if method == "tools/call":
            return self._call_tool(params)
        raise ProtocolError(METHOD_NOT_FOUND, f"method not found: {method}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        self.protocol_version = (requested if requested in SUPPORTED_PROTOCOL_VERSIONS
                                 else LATEST_PROTOCOL_VERSION)
        info = params.get("clientInfo")
        self.client_info = info if isinstance(info, dict) else {}
        logger.info("initialize from %s (protocol %s → %s)", self.client_info.get("name"),
                    requested, self.protocol_version)
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "title": "Agent VCR", "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise ProtocolError(INVALID_PARAMS, "tools/call needs a tool name")
        tool = self.tools.get(name)
        if tool is None:
            raise ProtocolError(INVALID_PARAMS, f"Unknown tool: {name}")
        raw_args = params.get("arguments") or {}
        if not isinstance(raw_args, dict):
            return _tool_error("arguments must be an object")
        try:
            result = tool.handler(self, _validate(tool, raw_args))
        except ToolError as err:
            return _tool_error(str(err), err.data)
        except (AirbagError, KeyError, OSError, ValueError) as err:
            message = err.args[0] if isinstance(err, KeyError) and err.args else err
            return _tool_error(str(message))
        except Exception as err:  # noqa: BLE001 - a tool must never crash the server
            logger.exception("tool %s failed", name)
            return _tool_error(f"{name} failed unexpectedly: {err!r}")
        except SystemExit as err:  # defensive: reused helpers must not stop the server
            return _tool_error(f"{name} stopped: exit {err.code}")
        return {
            "content": [{"type": "text", "text": _dump(result, indent=2)}],
            "structuredContent": result,
            "isError": False,
        }


def _dump(data: Any, indent: int | None = None) -> str:
    separators = None if indent else (",", ":")
    return json.dumps(data, ensure_ascii=False, default=str, indent=indent,
                      separators=separators)


def _error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": error}


def _tool_error(message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    text = f"Error: {message}"
    if data:
        text += "\n" + _dump(data, indent=2)
    return {"content": [{"type": "text", "text": text}], "isError": True}


# ── tools ────────────────────────────────────────────────────────────────

def _tool_status(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    from agent_vcr.airbag.agents import detect_installed

    bag = server.bag(args)
    info = bag.store.info()
    hooks = detect_installed(bag.root)
    policy = bag.policy
    latest = bag.store.latest()
    undone = bag.store.read_state().get("undo_pos") is not None
    result: dict[str, Any] = {
        "project": info["root"],
        "protected": bool(policy.enabled and hooks),
        "enabled": policy.enabled,
        "mode": policy.mode,
        "on_irreversible": policy.on_irreversible,
        "store": info["store"],
        "snapshots": info["snapshots"],
        "latest": _snap_info(latest) if latest else None,
        "hooks": hooks,
        "in_undone_state": undone,
    }
    if not policy.enabled:
        result["hint"] = "The airbag is disabled by AGENT_VCR_AIRBAG=off in this environment."
    elif not hooks:
        result["hint"] = ("No agent hooks are installed, so tool calls are not snapshotted "
                          "automatically. The user can arm them with `vcr airbag install`.")
    return result


def _tool_timeline(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    bag = server.bag(args)
    actions = bag.actions()
    limit = int(args["limit"])
    shown = actions[-limit:]
    return {
        "project": str(bag.root),
        "total": len(actions),
        "actions": [_action_info(a) for a in reversed(shown)],
        "order": "newest first",
        "in_undone_state": bag.store.read_state().get("undo_pos") is not None,
    }


def _tool_check(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    from agent_vcr.airbag.classify import classify
    from agent_vcr.airbag.store import find_project_root

    project = server.resolve(args.get("project_dir"))
    cwd = server.resolve(args["cwd"]) if args.get("cwd") else project
    command = str(args["command"]).strip()
    verdict = classify(command, cwd=str(cwd), root=find_project_root(project))
    policy = Policy.from_env()
    return {
        "command": _clip(command, 2000),
        "level": verdict.level,
        "reasons": verdict.reasons,
        "categories": verdict.categories,
        "undoable": verdict.undoable,
        "meaning": LEVEL_EXPLAIN[verdict.level],
        "advice": AGENT_ADVICE[verdict.level],
        "airbag_would_stop": (verdict.level == "irreversible" and policy.enabled
                              and policy.on_irreversible != "warn"),
    }


def _tool_snapshot(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    bag = server.bag(args)
    before = bag.store.latest()
    snap = bag.manual_snapshot(_clip(args["message"], 200), agent="mcp")
    new = before is None or snap.seq != before.seq
    return {
        **_snap_info(snap),
        "project": str(bag.root),
        "new": new,
        "note": (f"Snapshot #{snap.seq} saved." if new else
                 f"Nothing changed since snapshot #{snap.seq}; it already protects this state."),
    }


def _require_confirmation(args: dict[str, Any], what: str, preview: dict[str, Any]) -> bool:
    """True to restore for real; raise when ``dry_run`` is off without ``confirm``."""
    if args["dry_run"]:
        return False
    if args["confirm"] is not True:
        raise ToolError(
            f"refusing to {what} without confirm=true. This changes the user's files: call "
            "with dry_run=false and confirm=true only when the user explicitly asked for it. "
            "Preview of what it would change:", preview)
    return True


def _tool_undo(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    bag = server.bag(args)
    steps = int(args["steps"])
    preview = bag.preview_undo(steps)
    if preview is None:
        return {"project": str(bag.root), "nothing_to_undo": True, "note": NOTHING_TO_UNDO}
    action, snap, changes = preview
    summary = {"project": str(bag.root), "action": _action_info(action),
               "snapshot": _snap_info(snap), **_changes(changes)}
    if not _require_confirmation(args, "undo", summary):
        return {"dry_run": True, **summary,
                "next": "To restore, call again with dry_run=false and confirm=true — only if "
                        "the user asked you to undo."}
    result = bag.undo(steps)
    if result is None:
        return {"project": str(bag.root), "nothing_to_undo": True, "note": NOTHING_TO_UNDO}
    undone, restore = result
    return {"dry_run": False, "project": str(bag.root), "undone": _action_info(undone),
            **_restore_info(restore)}


def _tool_rewind(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    bag = server.bag(args)
    seq = int(args["snapshot"])
    try:
        target, changes = bag.preview_rewind(seq)
    except KeyError:
        raise ToolError(f"no airbag snapshot #{seq}; list them with airbag_timeline") from None
    summary = {"project": str(bag.root), "snapshot": _snap_info(target), **_changes(changes)}
    if not _require_confirmation(args, f"rewind to snapshot #{seq}", summary):
        return {"dry_run": True, **summary,
                "next": "To restore, call again with dry_run=false and confirm=true — only if "
                        "the user asked you to rewind."}
    return {"dry_run": False, "project": str(bag.root), **_restore_info(bag.rewind(seq))}


def _read_header(path: Path) -> dict[str, Any]:
    """Session id, creation time and frame count from a recording's first line."""
    try:
        with open(path, "rb") as f:
            line = f.readline(1024 * 1024)
        record = json.loads(line)
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("type") != "session":
        return {}
    data = record.get("data")
    if not isinstance(data, dict):
        return {}
    return {"session_id": data.get("session_id"), "created_at": data.get("created_at"),
            "tags": data.get("tags") or []}


def _tool_recordings_list(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    root = server.resolve(args.get("directory"))
    if not root.is_dir():
        raise ToolError(f"not a directory: {root}")
    found: list[tuple[float, Path, int]] = []
    seen = 0
    deadline = time.monotonic() + SCAN_MAX_SECONDS
    scan_truncated = False
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        seen += len(dirs) + len(files)
        for name in files:
            if not name.endswith(".vcr"):
                continue
            path = Path(current) / name
            with contextlib.suppress(OSError):
                stat = path.stat()
                found.append((stat.st_mtime, path, stat.st_size))
        if seen > SCAN_MAX_ENTRIES or time.monotonic() > deadline:
            scan_truncated = True
            break
    found.sort(key=lambda item: item[0], reverse=True)
    limit = int(args["limit"])
    recordings = []
    for mtime, path, size in found[:limit]:
        recordings.append({"path": str(path), "relative": os.path.relpath(path, root),
                           "size_bytes": size, "modified": _iso(mtime), **_read_header(path)})
    return {"directory": str(root), "count": len(found), "recordings": recordings,
            "truncated": len(found) > limit, "scan_truncated": scan_truncated,
            "order": "newest first"}


def _tool_recording_summary(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    from agent_vcr.models import FrameType
    from agent_vcr.player import VCRPlayer

    path = server.resolve(args["path"])
    if not path.is_file():
        raise ToolError(f"recording not found: {path}")
    if path.stat().st_size > MAX_RECORDING_BYTES:
        raise ToolError(f"{path} is larger than {MAX_RECORDING_BYTES // 1048576} MB")
    player = VCRPlayer.load(path)
    frames = player.frames
    kinds: dict[str, int] = {}
    executions: dict[str, int] = {}
    models: list[str] = []
    for frame in frames:
        kinds[frame.frame_type.value] = kinds.get(frame.frame_type.value, 0) + 1
        executions[frame.node_name] = executions.get(frame.node_name, 0) + 1
        if frame.metadata.model and frame.metadata.model not in models:
            models.append(frame.metadata.model)
    errors = [
        {"index": i, "node": f.node_name, "type": f.metadata.error_type,
         "message": _clip(f.metadata.error_message or "")}
        for i, f in enumerate(frames) if f.frame_type == FrameType.ERROR
    ]
    limit = int(args["errors_limit"])
    integrity = player.verify_integrity()
    session = player.session
    return {
        "path": str(path),
        "session_id": session.session_id,
        "parent_session_id": session.parent_session_id,
        "created_at": session.created_at.isoformat(),
        "tags": session.tags,
        "metadata": (session.metadata if len(_dump(session.metadata)) <= 4000
                     else {"truncated": True, "keys": sorted(session.metadata)[:50]}),
        "frames": len(frames),
        "frame_types": kinds,
        "nodes": [{"name": n, "executions": executions[n]} for n in player.list_nodes()],
        "checkpoints": [c["label"] for c in player.list_checkpoints()],
        "error_count": len(errors),
        "errors": errors[:limit],
        "tokens": player.get_total_tokens(),
        "tokens_input": sum(f.metadata.tokens_input or 0 for f in frames),
        "tokens_output": sum(f.metadata.tokens_output or 0 for f in frames),
        "cost_usd": round(player.get_total_cost(), 6),
        "latency_ms": round(player.get_total_latency(), 3),
        "models": models,
        "integrity": {"ok": integrity["ok"], "issues": integrity["issues"][:20]},
        "skipped_lines": player.skipped_lines,
    }


_DRY_RUN = {"type": "boolean", "default": True,
            "description": "Only preview what would change (default true)."}
_CONFIRM = {"type": "boolean", "default": False,
            "description": "Must be true, together with dry_run=false, to actually restore. "
                           "Set it only when the user explicitly asked."}

TOOLS: tuple[Tool, ...] = (
    Tool(
        name="airbag_status", title="Airbag status",
        description="Is this project protected by the Agent VCR airbag? Returns whether it is "
                    "enabled, the policy, the snapshot store and count, the latest snapshot "
                    "and which agents have airbag hooks installed. Read-only.",
        properties={"project_dir": _PROJECT_DIR},
        handler=_tool_status,
    ),
    Tool(
        name="airbag_timeline", title="Airbag timeline",
        description="Recent protected agent actions (newest first): the snapshot taken before "
                    "each, when, which agent and tool, the command or edited paths and its risk "
                    "level. Use the snapshot numbers with airbag_rewind. Read-only.",
        properties={
            "project_dir": _PROJECT_DIR,
            "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 500,
                      "description": "How many actions to return (default 20)."},
        },
        handler=_tool_timeline,
    ),
    Tool(
        name="airbag_check", title="Check a shell command",
        description="Classify a shell command WITHOUT running it: read, write, destructive "
                    "(snapshot can undo it) or irreversible (a snapshot cannot undo it, e.g. a "
                    "remote DROP TABLE, force-push, publish). Call this before running a "
                    "command that might delete or overwrite data, and ask the user first when "
                    "the level is irreversible. Read-only.",
        properties={
            "command": {"type": "string", "minLength": 1,
                        "description": "The shell command, exactly as you would run it."},
            "cwd": {"type": "string",
                    "description": "Directory the command would run in (default: project_dir)."},
            "project_dir": _PROJECT_DIR,
        },
        required=("command",),
        handler=_tool_check,
    ),
    Tool(
        name="airbag_snapshot", title="Take a snapshot",
        description="Take a manual airbag snapshot of the project now, with a message, so the "
                    "user can come back to this exact state later. Safe: it only writes to the "
                    "snapshot store outside the project, never to project files.",
        properties={
            "message": {"type": "string", "default": "manual snapshot (MCP)",
                        "description": "What this save point is for."},
            "project_dir": _PROJECT_DIR,
        },
        handler=_tool_snapshot, read_only=False,
    ),
    Tool(
        name="airbag_undo", title="Undo agent actions",
        description="Undo the last file-changing agent action(s) by restoring the airbag "
                    "snapshot taken before them. By default only PREVIEWS what would change "
                    "(dry_run=true). Actually restoring CHANGES THE USER'S FILES and requires "
                    "dry_run=false AND confirm=true — do that only when the user explicitly "
                    "asked you to undo. The restore is snapshotted first, so `vcr redo` can "
                    "reverse it.",
        properties={
            "steps": {"type": "integer", "default": 1, "minimum": 1, "maximum": 100,
                      "description": "How many file-changing actions to go back (default 1)."},
            "dry_run": _DRY_RUN,
            "confirm": _CONFIRM,
            "project_dir": _PROJECT_DIR,
        },
        handler=_tool_undo, read_only=False, destructive=True, idempotent=False,
    ),
    Tool(
        name="airbag_rewind", title="Rewind to a snapshot",
        description="Restore the project to an airbag snapshot number (see airbag_timeline). "
                    "By default only PREVIEWS what would change (dry_run=true). Actually "
                    "restoring CHANGES THE USER'S FILES and requires dry_run=false AND "
                    "confirm=true — do that only when the user explicitly asked you to rewind. "
                    "The current state is snapshotted first, so `vcr redo` can reverse it.",
        properties={
            "snapshot": {"type": "integer", "minimum": 1,
                         "description": "Snapshot number, e.g. 12 for #12."},
            "dry_run": _DRY_RUN,
            "confirm": _CONFIRM,
            "project_dir": _PROJECT_DIR,
        },
        required=("snapshot",),
        handler=_tool_rewind, read_only=False, destructive=True, idempotent=False,
    ),
    Tool(
        name="recordings_list", title="List recordings",
        description="Find Agent VCR recordings (.vcr files) under a directory, newest first, "
                    "with size, time and session id. Skips .git, node_modules and virtualenvs. "
                    "Read-only.",
        properties={
            "directory": {"type": "string",
                          "description": "Where to look (default: the project directory)."},
            "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 1000,
                      "description": "Maximum recordings to return (default 50)."},
        },
        handler=_tool_recordings_list,
    ),
    Tool(
        name="recording_summary", title="Summarize a recording",
        description="Summarize one .vcr recording: frame count and types, nodes and how often "
                    "each ran, checkpoints, errors, tokens, cost, latency, models and an "
                    "integrity check. Read-only.",
        properties={
            "path": {"type": "string", "minLength": 1,
                     "description": "Path to the .vcr file (absolute or project-relative)."},
            "errors_limit": {"type": "integer", "default": 10, "minimum": 0, "maximum": 500,
                             "description": "Maximum error frames to list (default 10)."},
        },
        required=("path",),
        handler=_tool_recording_summary,
    ),
)


# ── stdio transport ──────────────────────────────────────────────────────

def serve(server: MCPServer, stdin: IO[bytes], stdout: IO[bytes]) -> int:
    """Read newline-delimited JSON-RPC from ``stdin`` until EOF; answer on ``stdout``."""
    while True:
        try:
            line = stdin.readline()
        except KeyboardInterrupt:
            return 0
        if not line:
            return 0
        if not line.strip():
            continue
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as err:
            response: dict[str, Any] | None = _error(None, PARSE_ERROR, f"parse error: {err}")
        else:
            response = server.handle(message)
        if response is None:
            continue
        try:
            stdout.write(_dump(response).encode("utf-8") + b"\n")
            stdout.flush()
        except (BrokenPipeError, ValueError):
            return 0  # the client went away


def run(project_dir: str | Path = ".") -> int:
    """Entry point for ``vcr mcp``: serve on the real stdin/stdout."""
    root = logging.getLogger("agent_vcr")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("agent-vcr mcp: %(levelname)s %(message)s"))
        root.addHandler(handler)
        level = os.environ.get("AGENT_VCR_MCP_LOG", "WARNING").upper()
        root.setLevel(getattr(logging, level, logging.WARNING))
        root.propagate = False
    server = MCPServer(project_dir)
    proto_in, proto_out = _claim_stdio()
    logger.info("serving %s on stdio", server.project_dir)
    try:
        return serve(server, proto_in, proto_out)
    finally:
        with contextlib.suppress(OSError, ValueError):
            proto_out.close()


def _claim_stdio() -> tuple[IO[bytes], IO[bytes]]:
    """Keep fds 0/1 for the protocol only.

    The protocol streams move to private file descriptors; fd 1 is pointed at
    stderr and fd 0 at /dev/null, so a stray ``print`` or a child process can
    neither corrupt the JSON stream nor swallow client messages.
    """
    try:
        sys.stdout.flush()
        in_fd, out_fd = os.dup(0), os.dup(1)
        os.dup2(2, 1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
        sys.stdout = sys.stderr
        return os.fdopen(in_fd, "rb"), os.fdopen(out_fd, "wb")
    except (OSError, ValueError, AttributeError):
        out = sys.stdout.buffer
        sys.stdout = sys.stderr
        return sys.stdin.buffer, out


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "."))
