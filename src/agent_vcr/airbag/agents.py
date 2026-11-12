"""Per-agent hook adapters: parse the agent's payload, answer in its format, install.

Contracts were taken from each agent's official docs / installed source
(see docs/airbag.md for sources and what is verified). One tolerant parser
handles every payload shape, because several agents also run each other's
hooks (Cursor and Copilot CLI read ``.claude/settings.json``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_vcr.airbag.core import Decision, ToolCall

HOOK_MODULE = "agent_vcr.airbag.hook"
MARKER = f"-m {HOOK_MODULE} "

SHELL_TOOLS = frozenset({
    "bash", "shell", "run_shell_command", "powershell", "execute_bash", "execute_command",
    "terminal", "exec_command", "local_shell", "shell_command", "container.exec", "cmd",
    "exec", "run_commands",
})
EDIT_TOOLS = frozenset({
    "write", "edit", "multiedit", "notebookedit", "write_file", "replace", "create",
    "str_replace_editor", "str_replace_based_edit_tool", "apply_patch", "write_to_file",
    "replace_in_file", "delete", "file_editor", "edit_file", "create_file", "delete_file",
    "insert_edit_into_file", "multi_edit", "editor", "planning_file_editor", "patch",
})
#: file_editor/editor sub-commands that only read.
_READ_EDITOR_COMMANDS = frozenset({"view", "read", "list"})
_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.M)


# ── parsing ──────────────────────────────────────────────────────────────

def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _first(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def parse_payload(agent: str, payload: dict[str, Any]) -> ToolCall | None:
    """Normalize any supported agent's pre-tool payload. ``None`` = not our business."""
    cline = _as_dict(payload.get("preToolUse"))  # Cline: {"preToolUse": {toolName, parameters}}
    cline_call = _as_dict(payload.get("tool_call"))  # Cline CLI: {"tool_call": {id, name, input}}
    tool_input = _as_dict(payload.get("tool_input")) or _as_dict(payload.get("toolArgs")) or \
        _as_dict(payload.get("tool_args")) or _as_dict(payload.get("arguments")) or \
        _as_dict(cline.get("parameters")) or _as_dict(cline_call.get("input"))
    tool_name = _first(payload.get("tool_name"), payload.get("toolName"), payload.get("tool"),
                       cline.get("toolName"), cline_call.get("name")) or ""
    raw_command: Any = tool_input.get("command", tool_input.get("cmd"))
    if raw_command is None and isinstance(tool_input.get("commands"), list):
        raw_command = " ; ".join(str(c) for c in tool_input["commands"])  # Cline run_commands
    if raw_command is None and not tool_name:
        raw_command = payload.get("command")  # Cursor beforeShellExecution
        tool_name = "Shell" if raw_command else ""
    command: str | None
    if isinstance(raw_command, list):
        command = shlex.join(str(part) for part in raw_command)
    else:
        command = raw_command if isinstance(raw_command, str) else None

    name = tool_name.lower()
    if name.startswith("mcp__") or name.startswith("mcp:") or name.startswith("mcp_"):
        return None
    workspace_roots = payload.get("workspace_roots") or payload.get("workspaceRoots") or []
    cwd = _first(
        tool_input.get("dir_path"), tool_input.get("working_directory"),
        tool_input.get("workdir"), tool_input.get("cwd"), payload.get("cwd"),
        payload.get("working_dir"), os.environ.get("OPENHANDS_PROJECT_DIR"),
        workspace_roots[0] if isinstance(workspace_roots, list) and workspace_roots else None,
        os.environ.get("CLAUDE_PROJECT_DIR"), os.environ.get("GEMINI_PROJECT_DIR"),
        os.environ.get("CURSOR_PROJECT_DIR"),
    ) or os.getcwd()
    project_dir = _first(
        os.environ.get("CLAUDE_PROJECT_DIR"), os.environ.get("GEMINI_PROJECT_DIR"),
        os.environ.get("CURSOR_PROJECT_DIR"), os.environ.get("OPENHANDS_PROJECT_DIR"),
        payload.get("working_dir"),
        workspace_roots[0] if isinstance(workspace_roots, list) and workspace_roots else None,
        payload.get("cwd"),
    )
    session_id = _first(payload.get("session_id"), payload.get("sessionId"),
                        payload.get("conversation_id"), payload.get("taskId"))
    call_id = _first(payload.get("tool_use_id"), payload.get("toolUseId"),
                     payload.get("call_id"), payload.get("execution_id"), cline_call.get("id"))

    def make(kind: str, command: str | None, paths: list[str]) -> ToolCall:
        return ToolCall(agent=agent, tool_name=tool_name or "Shell", kind=kind, cwd=cwd,
                        command=command, paths=paths, project_dir=project_dir,
                        session_id=session_id, call_id=call_id)

    if name in SHELL_TOOLS or (not name and command):
        if not command:
            return None
        return make("shell", command, [])
    if name in EDIT_TOOLS:
        if isinstance(raw_command, str) and raw_command.lower() in _READ_EDITOR_COMMANDS:
            return None  # e.g. OpenHands file_editor {"command": "view"}
        paths = [p for p in (
            tool_input.get("file_path"), tool_input.get("path"), tool_input.get("notebook_path"),
            tool_input.get("filePath"), tool_input.get("target_file"), tool_input.get("file"),
            payload.get("file_path"),
        ) if isinstance(p, str) and p]
        # apply_patch: Codex sends the patch text in "command", Cline in "input".
        patch = tool_input.get("input") or tool_input.get("patch") or \
            tool_input.get("patchText") or (raw_command if name == "apply_patch" else "")
        if isinstance(patch, str):
            paths.extend(m.strip() for m in _PATCH_FILE.findall(patch))
        return make("edit", None, paths)
    return None


# ── rendering ────────────────────────────────────────────────────────────

def _json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def render_claude(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Claude Code PreToolUse (also accepted by Cursor CLI and Copilot CLI)."""
    out: dict[str, Any] = {}
    if decision.user_message:
        out["systemMessage"] = decision.user_message
    if decision.action in {"ask", "deny"}:
        reason = decision.user_message if decision.action == "ask" else decision.agent_message
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision.action,
            "permissionDecisionReason": reason or "Agent VCR airbag",
        }
    return (_json(out) if out else ""), 0


def render_gemini(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Gemini CLI BeforeTool. ``ask`` exists from 0.36 (installer picks the mode)."""
    out: dict[str, Any] = {}
    if decision.user_message:
        out["systemMessage"] = decision.user_message
    if decision.action in {"ask", "deny"}:
        action = "ask" if decision.action == "ask" and options == "ask" else "deny"
        out["decision"] = action
        if action == "deny":
            out["reason"] = decision.agent_message or decision.user_message or "Agent VCR airbag"
    return (_json(out) if out else ""), 0


def render_cursor(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Cursor beforeShellExecution / preToolUse: an answer is mandatory.

    ``allow`` here means "no objection from this hook" (Cursor's documented
    examples answer allow for every command they do not care about).
    Irreversible commands are denied (see below); the user can approve one
    retry with ``vcr airbag allow``.
    """
    action = decision.action
    if action == "ask":
        # Cursor documents "ask" for beforeShellExecution, but support staff report it is
        # not reliably enforced; an unenforced ask would let the command run. Deny instead:
        # the user approves with `vcr airbag allow` and the agent retries.
        action = "deny"
    out: dict[str, Any] = {"permission": "allow" if action == "continue" else action}
    if decision.user_message:
        out["user_message"] = decision.user_message
    if action in {"ask", "deny"} and decision.agent_message:
        out["agent_message"] = decision.agent_message
    return _json(out), 0


def render_copilot(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """GitHub Copilot CLI preToolUse (camelCase). Empty output = default behavior."""
    lines: list[str] = []
    if decision.user_message and decision.action == "continue":
        lines.append(_json({"type": "progress", "message": decision.user_message}))
    if decision.action in {"ask", "deny"}:
        lines.append(_json({
            "permissionDecision": decision.action,
            "permissionDecisionReason": decision.agent_message or decision.user_message
            or "Agent VCR airbag",
        }))
    return "\n".join(lines), 0


def render_openhands(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """OpenHands SDK hooks: no "ask" — irreversible commands are denied with a reason.

    For a real confirmation prompt use the in-process integration
    (``agent_vcr.integrations.openhands_airbag``), which plugs into
    OpenHands' confirmation policy.
    """
    if decision.action in {"ask", "deny"}:
        return _json({"decision": "deny",
                      "reason": decision.agent_message or decision.user_message or "airbag"}), 0
    return "", 0


def render_cline(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Cline PreToolUse script. ``cancel`` stops the task — used only for irreversible."""
    if decision.action in {"ask", "deny"}:
        return _json({"cancel": True,
                      "errorMessage": decision.agent_message or decision.user_message or ""}), 0
    if decision.user_message:
        return _json({"cancel": False, "contextModification": decision.user_message}), 0
    return _json({"cancel": False}), 0


def render_devin(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Devin Local / Devin Desktop (formerly Windsurf): no "ask" — block irreversible."""
    if decision.action in {"ask", "deny"}:
        return _json({"decision": "block",
                      "reason": decision.agent_message or decision.user_message or ""}), 0
    return "", 0


def render_opencode(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Answer for the OpenCode plugin shim (it throws on deny; no "ask" from plugins)."""
    out: dict[str, Any] = {}
    if decision.action in {"ask", "deny"}:
        out["decision"] = "deny"
        out["reason"] = decision.agent_message or decision.user_message or "Agent VCR airbag"
    elif decision.user_message:
        out["message"] = decision.user_message
    return (_json(out) if out else ""), 0


def render_codex(decision: Decision, call: ToolCall, options: str) -> tuple[str, int]:
    """Codex CLI PreToolUse: Claude-shaped JSON, but Codex has no "ask".

    An "ask" answer is ignored by Codex and the command runs, so irreversible
    commands are denied (with the non-empty reason Codex requires).
    """
    out: dict[str, Any] = {}
    if decision.user_message:
        out["systemMessage"] = decision.user_message
    if decision.action in {"ask", "deny"}:
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": decision.agent_message or decision.user_message
            or "Blocked by Agent VCR airbag",
        }
    return (_json(out) if out else ""), 0


# ── installing ───────────────────────────────────────────────────────────

#: Interpreter paths that need no quoting in sh, bash, cmd.exe or PowerShell.
_PLAIN_PATH = re.compile(r"[A-Za-z0-9_@%+=:,./-]+")


def _interpreter(shell: str) -> str:
    """This interpreter, spelled so the agent's hook shell runs it.

    POSIX: ``shlex.quote``. Windows: agents run hooks through Git Bash
    (Claude Code), cmd.exe or PowerShell, and ``shlex.quote``'s single quotes
    mean nothing to cmd and turn the path into a string literal in PowerShell.
    Forward slashes work in all three, double quotes (only when needed) in
    bash and cmd; a PowerShell-only command uses its call operator.
    """
    python = sys.executable or "python3"
    if sys.platform != "win32":
        return shlex.quote(python)
    if shell == "powershell":
        return "& '" + python.replace("'", "''") + "'"
    path = python.replace(os.sep, "/")
    return path if _PLAIN_PATH.fullmatch(path) else f'"{path}"'


def hook_command(agent: str, option: str = "", shell: str = "default") -> str:
    """The hook command line for ``agent`` (``shell="powershell"`` for PowerShell-only hooks)."""
    return f"{_interpreter(shell)} {MARKER}{agent} pre{(' ' + option) if option else ''}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _is_ours(entry: Any) -> bool:
    text = json.dumps(entry) if not isinstance(entry, str) else entry
    return MARKER.strip() in text


def _merge_event_list(settings: dict[str, Any], event: str, entry: dict[str, Any] | None,
                      container: str = "hooks") -> None:
    hooks = settings.setdefault(container, {})
    if not isinstance(hooks, dict):
        raise ValueError(f"'{container}' must be a JSON object")
    existing = hooks.get(event, [])
    if not isinstance(existing, list):
        raise ValueError(f"'{container}.{event}' must be a JSON array")
    kept: list[Any] = []
    for item in existing:
        if isinstance(item, dict) and isinstance(item.get("hooks"), list):
            others = [h for h in item["hooks"] if not _is_ours(h)]
            if not others:
                continue
            item = {**item, "hooks": others}
        elif _is_ours(item):
            continue
        kept.append(item)
    if entry is not None:
        kept.append(entry)
    if kept:
        hooks[event] = kept
    else:
        hooks.pop(event, None)


@dataclass
class Adapter:
    name: str
    title: str
    binaries: tuple[str, ...]
    render: Callable[[Decision, ToolCall, str], tuple[str, int]]
    project_config: str | None
    user_config: str | None
    note: str = ""

    def parse(self, payload: dict[str, Any]) -> ToolCall | None:
        return parse_payload(self.name, payload)

    def config_path(self, root: str | Path, user: bool) -> Path:
        rel = self.user_config if user else self.project_config
        if rel is None:
            raise ValueError(f"{self.title} has no {'user' if user else 'project'}-level config")
        return Path(rel).expanduser() if user else Path(root) / rel


def _install_claude(path: Path) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [{"type": "command", "command": hook_command("claude"), "timeout": 30}],
    })
    _write_json(path, settings)


def _install_gemini(path: Path) -> None:
    settings = _read_json(path)
    option = "ask" if _gemini_supports_ask() else "deny"
    _merge_event_list(settings, "BeforeTool", {
        "matcher": "run_shell_command|write_file|replace",
        "hooks": [{"name": "agent-vcr-airbag", "type": "command",
                   "command": hook_command("gemini", option), "timeout": 30000}],
    })
    _write_json(path, settings)


def _install_cursor(path: Path) -> None:
    settings = _read_json(path)
    settings.setdefault("version", 1)
    _merge_event_list(settings, "beforeShellExecution",
                      {"command": hook_command("cursor", "shell"), "timeout": 30})
    _merge_event_list(settings, "preToolUse",
                      {"command": hook_command("cursor", "tool"), "matcher": "Write|Delete",
                       "timeout": 30})
    _write_json(path, settings)


def _install_copilot(path: Path) -> None:
    # A dedicated file: Copilot loads every *.json in the hooks directory.
    _write_json(path, {"version": 1, "hooks": {"preToolUse": [{
        "type": "command",
        "bash": hook_command("copilot"),
        "powershell": hook_command("copilot", shell="powershell"),
        "matcher": "bash|powershell|create|edit|str_replace_editor|apply_patch",
        "timeoutSec": 30,
    }]}})


def _install_codex(path: Path) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "Bash|apply_patch",
        "hooks": [{"type": "command", "command": hook_command("codex"), "timeout": 30}],
    })
    _write_json(path, settings)


def _install_openhands(path: Path) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "terminal|file_editor|apply_patch|write_file|edit|planning_file_editor",
        "hooks": [{"type": "command", "command": hook_command("openhands"), "timeout": 60}],
    })
    _write_json(path, settings)


def _install_devin(path: Path) -> None:
    settings = _read_json(path)  # hooks.v1.json has no top-level "hooks" wrapper
    entry = {"matcher": "exec|write|edit|apply_patch",
             "hooks": [{"type": "command", "command": hook_command("devin"), "timeout": 30}]}
    holder = {"hooks": settings}
    _merge_event_list(holder, "PreToolUse", entry)
    _write_json(path, holder["hooks"])


_OPENCODE_PLUGIN = """// Agent VCR airbag ({marker}) — snapshot before every OpenCode tool call.
// Undo with `vcr undo`. Generated by `vcr airbag install opencode`; safe to delete.
import {{ spawnSync }} from "node:child_process";

const PYTHON = {python};
const TOOLS = new Set(["bash", "shell", "edit", "write", "multiedit", "patch", "apply_patch"]);

export const AgentVcrAirbag = async ({{ directory }}) => ({{
  "tool.execute.before": async (input, output) => {{
    if (!TOOLS.has(input.tool)) return;
    const args = (output && output.args) || {{}};
    const payload = {{
      tool_name: input.tool,
      tool_input: args,
      cwd: args.workdir || directory,
      workspace_roots: [directory],
      session_id: input.sessionID,
      call_id: input.callID,
    }};
    const res = spawnSync(PYTHON, ["-m", "{module}", "opencode", "pre"], {{
      input: JSON.stringify(payload), encoding: "utf8", timeout: 30000,
    }});
    if (res.error || res.status !== 0 || !res.stdout) return; // never break OpenCode
    let answer;
    try {{ answer = JSON.parse(res.stdout); }} catch {{ return; }}
    if (answer.decision === "deny") throw new Error(answer.reason || "Blocked by Agent VCR airbag");
    if (answer.message) console.error(answer.message);
  }},
}});
"""


def _install_opencode(path: Path) -> None:
    if path.exists() and MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
        raise ValueError(f"{path} already exists and is not an Agent VCR plugin")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_OPENCODE_PLUGIN.format(marker=MARKER.strip(), module=HOOK_MODULE,
                                            python=json.dumps(sys.executable or "python3")),
                    encoding="utf-8")


_CLINE_SCRIPT = """#!/bin/sh
# Agent VCR airbag ({marker}) — snapshot before every Cline tool call.
exec {command}
"""


def _install_cline(path: Path) -> None:
    if sys.platform == "win32":
        raise ValueError("Cline hooks are not supported on Windows")
    if path.exists() and MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
        raise ValueError(f"{path} already exists and is not an Agent VCR hook; merge manually")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CLINE_SCRIPT.format(marker=MARKER.strip(), command=hook_command("cline")),
                    encoding="utf-8")
    path.chmod(0o755)


def _gemini_supports_ask() -> bool:
    exe = shutil.which("gemini")
    if exe is None:
        return False
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20,
                             stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (0, 36)


_INSTALLERS: dict[str, Callable[[Path], None]] = {
    "claude": _install_claude, "gemini": _install_gemini, "cursor": _install_cursor,
    "copilot": _install_copilot, "codex": _install_codex, "openhands": _install_openhands,
    "devin": _install_devin, "cline": _install_cline, "opencode": _install_opencode,
}

ADAPTERS: dict[str, Adapter] = {
    "claude": Adapter("claude", "Claude Code", ("claude",), render_claude,
                      ".claude/settings.json", "~/.claude/settings.json"),
    "gemini": Adapter("gemini", "Gemini CLI", ("gemini",), render_gemini,
                      ".gemini/settings.json", "~/.gemini/settings.json",
                      note="Gemini only runs hooks in trusted folders (/trust)."),
    "cursor": Adapter("cursor", "Cursor", ("cursor", "cursor-agent"), render_cursor,
                      ".cursor/hooks.json", "~/.cursor/hooks.json",
                      note="Cursor also runs .claude/settings.json hooks; duplicates are "
                           "de-duplicated by the airbag."),
    "copilot": Adapter("copilot", "GitHub Copilot CLI", ("copilot",), render_copilot,
                       ".github/hooks/agent-vcr-airbag.json",
                       "~/.copilot/hooks/agent-vcr-airbag.json",
                       note="Copilot loads hooks at startup; restart running sessions."),
    "codex": Adapter("codex", "OpenAI Codex CLI", ("codex",), render_codex,
                     ".codex/hooks.json", "~/.codex/hooks.json",
                     note="Codex asks you to review new hooks: run /hooks in Codex once. Codex "
                          "has no 'ask', so irreversible commands are blocked until you run "
                          "`vcr airbag allow`."),
    "openhands": Adapter("openhands", "OpenHands", ("openhands",), render_openhands,
                         ".openhands/hooks.json", "~/.openhands/hooks.json",
                         note="SDK agents must load it: Conversation(..., hook_config="
                              "HookConfig.load(working_dir=ws)); for confirmation prompts use "
                              "agent_vcr.integrations.openhands_airbag."),
    "devin": Adapter("devin", "Devin Local / Devin Desktop (ex-Windsurf)", ("devin",),
                     render_devin, ".devin/hooks.v1.json", "~/.devin/hooks.v1.json"),
    "opencode": Adapter("opencode", "OpenCode (plugin API v1)", ("opencode",), render_opencode,
                        ".opencode/plugin/agent-vcr-airbag.js",
                        "~/.config/opencode/plugin/agent-vcr-airbag.js",
                        note="OpenCode plugins cannot ask, so irreversible commands are blocked "
                             "until `vcr airbag allow`. OpenCode 2.x plugins are not supported "
                             "yet."),
    "cline": Adapter("cline", "Cline", ("cline",), render_cline,
                     ".clinerules/hooks/PreToolUse", "~/Documents/Cline/Hooks/PreToolUse",
                     note="Cline cancels the whole task for irreversible commands (it has no "
                          "'ask'). Cline CLI uses .cline/hooks/PreToolUse.sh instead."),
}


def install(name: str, root: str | Path, user: bool = False) -> tuple[Path, str]:
    adapter = ADAPTERS[name]
    path = adapter.config_path(root, user)
    _INSTALLERS[name](path)
    return path, adapter.note


def uninstall(name: str, root: str | Path, user: bool = False) -> Path | None:
    adapter = ADAPTERS[name]
    try:
        path = adapter.config_path(root, user)
    except ValueError:
        return None
    if not path.exists():
        return None
    if name in {"copilot", "cline", "opencode"}:
        if MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
            return None
        path.unlink()
        return path
    if name == "devin":
        holder = {"hooks": _read_json(path)}
        if not _is_ours(holder):
            return None
        for event in list(holder["hooks"]):
            if isinstance(holder["hooks"].get(event), list):
                _merge_event_list(holder, event, None)
        _write_json(path, holder["hooks"])
        return path
    try:
        settings = _read_json(path)
    except (ValueError, json.JSONDecodeError):
        return None
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict) or not _is_ours(hooks):
        return None
    for event in list(hooks):
        _merge_event_list(settings, event, None)
    if not settings.get("hooks"):
        settings.pop("hooks", None)
    _write_json(path, settings)
    return path


def detect_available() -> list[str]:
    return [name for name, a in ADAPTERS.items() if any(shutil.which(b) for b in a.binaries)]


def detect_installed(root: str | Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, adapter in ADAPTERS.items():
        for user in (False, True):
            try:
                path = adapter.config_path(root, user)
            except ValueError:
                continue
            with contextlib.suppress(OSError):
                if path.exists() and MARKER.strip() in path.read_text(encoding="utf-8"):
                    found[name] = str(path)
                    break
    return found
