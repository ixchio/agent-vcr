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


def _interpreter(shell: str, python: str | None = None) -> str:
    """This interpreter, spelled so the agent's hook shell runs it.

    POSIX: ``shlex.quote``. Windows: agents run hooks through Git Bash
    (Claude Code), cmd.exe or PowerShell, and ``shlex.quote``'s single quotes
    mean nothing to cmd and turn the path into a string literal in PowerShell.
    Forward slashes work in all three, double quotes (only when needed) in
    bash and cmd; a PowerShell-only command uses its call operator.
    """
    python = python or sys.executable or "python3"
    if sys.platform != "win32":
        return shlex.quote(python)
    if shell == "powershell":
        return "& '" + python.replace("'", "''") + "'"
    path = python.replace(os.sep, "/")
    return path if _PLAIN_PATH.fullmatch(path) else f'"{path}"'


#: What a guarded hook prints when its interpreter is gone, per agent that needs an answer.
#: Cursor requires one; "allow" there means "no objection" (see ``render_cursor``).
_MISSING_INTERPRETER_ANSWER = {"cursor": '{"permission":"allow"}'}


def _guard(python: str, command: str, fallback: str | None = None) -> str:
    """``command`` that becomes a silent no-op when ``python`` no longer exists (POSIX sh).

    A pipx reinstall, an upgrade that moves the venv, or a teammate without
    Agent VCR must not turn every tool call into a failing hook. ``test`` and
    ``exec`` work in sh, bash, zsh, dash and fish. Self-heal with
    ``vcr airbag install`` or ``vcr doctor --fix``.
    """
    otherwise = f"echo {shlex.quote(fallback)}" if fallback else "true"
    return f"test -x {python} && exec {command} || {otherwise}"


def hook_command(agent: str, option: str = "", shell: str = "default",
                 python: str | None = None) -> str:
    """The hook command line for ``agent`` (``shell="powershell"`` for PowerShell-only hooks).

    On POSIX it is guarded (see :func:`_guard`). On Windows the agent may run
    it through Git Bash, cmd.exe or PowerShell, which share no guard syntax,
    so it stays a plain command there; ``vcr doctor`` finds a stale one.
    """
    interpreter = _interpreter(shell, python)
    command = f"{interpreter} {MARKER}{agent} pre{(' ' + option) if option else ''}"
    if sys.platform == "win32" or shell == "powershell":
        return command
    return _guard(interpreter, command, _MISSING_INTERPRETER_ANSWER.get(agent))


STATUSLINE_MODULE = "agent_vcr.airbag.statusline"


def statusline_command(python: str | None = None) -> str:
    """Claude Code ``statusLine`` command (prints nothing when the interpreter is gone)."""
    interpreter = _interpreter("default", python)
    command = f"{interpreter} -m {STATUSLINE_MODULE}"
    return command if sys.platform == "win32" else _guard(interpreter, command)


def hook_interpreter(command: str) -> str | None:
    """The interpreter path inside a hook command this module wrote (any spelling)."""
    marker = MARKER.strip() if MARKER.strip() in command else f"-m {STATUSLINE_MODULE}"
    idx = command.find(marker)
    if idx < 0:
        return None
    prefix = command[:idx].strip()
    if "exec " in prefix:
        prefix = prefix.rsplit("exec ", 1)[1].strip()
    if prefix.startswith("& '") and prefix.endswith("'"):
        return prefix[3:-1].replace("''", "'")
    if prefix.startswith('"') and prefix.endswith('"'):
        return prefix[1:-1]
    try:
        parts = shlex.split(prefix, posix=True)
    except ValueError:
        return None
    return parts[-1] if parts else None


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


def _config_home() -> Path:
    """XDG config home (OpenCode, Devin on POSIX): ``$XDG_CONFIG_HOME`` or ``~/.config``."""
    raw = os.environ.get("XDG_CONFIG_HOME")
    return Path(raw).expanduser() if raw else Path("~/.config").expanduser()


def _documents_dir() -> Path:
    """The user's Documents folder, as Cline finds it (``xdg-user-dir DOCUMENTS`` on Linux)."""
    home = Path("~").expanduser()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, UnicodeDecodeError):
            text = (_config_home() / "user-dirs.dirs").read_text(encoding="utf-8")
            match = re.search(r'^XDG_DOCUMENTS_DIR="([^"]+)"', text, re.M)
            if match:
                value = match.group(1).replace("$HOME", str(home))
                if Path(value) != home:
                    return Path(value)
    return home / "Documents"


def _expand_user(template: str) -> Path:
    """User-level config path: ``~``, ``{config}``, ``{documents}``, ``{copilot}``, ``{devin}``."""
    if "{config}" in template:
        template = template.replace("{config}", str(_config_home()))
    if "{documents}" in template:
        template = template.replace("{documents}", str(_documents_dir()))
    if "{copilot}" in template:
        template = template.replace(
            "{copilot}", os.environ.get("COPILOT_HOME") or str(Path("~/.copilot").expanduser()))
    if "{devin}" in template:
        appdata = os.environ.get("APPDATA") or ""
        base = Path(appdata) if sys.platform == "win32" and appdata else _config_home()
        template = template.replace("{devin}", str(base / "devin"))
    return Path(template).expanduser()


@dataclass
class Adapter:
    name: str
    title: str
    binaries: tuple[str, ...]
    render: Callable[[Decision, ToolCall, str], tuple[str, int]]
    #: Per-developer file inside the project (``--project``).
    project_config: str | None
    #: User-level file (the default): one install protects every project.
    user_config: str | None
    note: str = ""
    #: Where older versions installed (cleaned up by uninstall / migration).
    legacy_project: tuple[str, ...] = ()
    legacy_user: tuple[str, ...] = ()
    #: One-time step the user must do in the agent before hooks run (shown after install).
    required_step: str = ""

    def parse(self, payload: dict[str, Any]) -> ToolCall | None:
        return parse_payload(self.name, payload)

    def config_path(self, root: str | Path, user: bool) -> Path:
        rel = self.user_config if user else self.project_config
        if rel is None:
            raise ValueError(f"{self.title} has no {'user' if user else 'project'}-level config")
        return _expand_user(rel) if user else Path(root) / rel

    def locations(self, root: str | Path) -> list[tuple[str, Path, bool]]:
        """Every place this adapter installs or once installed: ``(scope, path, legacy)``."""
        out: list[tuple[str, Path, bool]] = []
        if self.user_config:
            out.append(("user", self.config_path(root, True), False))
        if self.project_config:
            out.append(("project", self.config_path(root, False), False))
        out.extend(("project", Path(root) / rel, True) for rel in self.legacy_project)
        out.extend(("user", _expand_user(rel), True) for rel in self.legacy_user)
        return out


def _install_claude(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
        "hooks": [{"type": "command", "command": hook_command("claude"), "timeout": 30}],
    })
    if statusline and "statusLine" not in settings:  # never overwrite the user's own
        settings["statusLine"] = {"type": "command", "command": statusline_command(),
                                  "padding": 0}
    elif _is_our_statusline(settings.get("statusLine")):
        settings["statusLine"] = {**settings["statusLine"], "command": statusline_command()}
    _write_json(path, settings)


def _is_our_statusline(value: Any) -> bool:
    return isinstance(value, dict) and f"-m {STATUSLINE_MODULE}" in str(value.get("command"))


def _install_gemini(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    option = "ask" if _gemini_supports_ask() else "deny"
    _merge_event_list(settings, "BeforeTool", {
        "matcher": "run_shell_command|write_file|replace",
        "hooks": [{"name": "agent-vcr-airbag", "type": "command",
                   "command": hook_command("gemini", option), "timeout": 30000}],
    })
    _write_json(path, settings)


def _install_cursor(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    settings.setdefault("version", 1)
    _merge_event_list(settings, "beforeShellExecution",
                      {"command": hook_command("cursor", "shell"), "timeout": 30})
    _merge_event_list(settings, "preToolUse",
                      {"command": hook_command("cursor", "tool"), "matcher": "Write|Delete",
                       "timeout": 30})
    _write_json(path, settings)


def _install_copilot(path: Path, statusline: bool = False) -> None:
    # A dedicated file: Copilot loads every *.json in the hooks directory.
    _write_json(path, {"version": 1, "hooks": {"preToolUse": [{
        "type": "command",
        "bash": hook_command("copilot"),
        "powershell": hook_command("copilot", shell="powershell"),
        "matcher": "bash|powershell|create|edit|str_replace_editor|apply_patch",
        "timeoutSec": 30,
    }]}})


def _install_codex(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "Bash|apply_patch",
        "hooks": [{"type": "command", "command": hook_command("codex"), "timeout": 30}],
    })
    _write_json(path, settings)


def _install_openhands(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    _merge_event_list(settings, "PreToolUse", {
        "matcher": "terminal|file_editor|apply_patch|write_file|edit|planning_file_editor",
        "hooks": [{"type": "command", "command": hook_command("openhands"), "timeout": 60}],
    })
    _write_json(path, settings)


def _devin_entry() -> dict[str, Any]:
    return {"matcher": "exec|write|edit|apply_patch",
            "hooks": [{"type": "command", "command": hook_command("devin"), "timeout": 30}]}


def _install_devin(path: Path, statusline: bool = False) -> None:
    settings = _read_json(path)
    if path.name == "hooks.v1.json":  # project file: no top-level "hooks" wrapper
        holder = {"hooks": settings}
        _merge_event_list(holder, "PreToolUse", _devin_entry())
        _write_json(path, holder["hooks"])
        return
    _merge_event_list(settings, "PreToolUse", _devin_entry())  # config.json: {"hooks": {...}}
    _write_json(path, settings)


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
    // Interpreter gone (pipx reinstall, upgrade) or any failure: never break OpenCode.
    if (res.error || res.status !== 0 || !res.stdout) return;
    let answer;
    try {{ answer = JSON.parse(res.stdout); }} catch {{ return; }}
    if (answer.decision === "deny") throw new Error(answer.reason || "Blocked by Agent VCR airbag");
    if (answer.message) console.error(answer.message);
  }},
}});
"""
_OPENCODE_PYTHON = re.compile(r"^const PYTHON = (\".*\");$", re.M)


def _install_opencode(path: Path, statusline: bool = False) -> None:
    if path.exists() and MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
        raise ValueError(f"{path} already exists and is not an Agent VCR plugin")
    _write_text(path, _OPENCODE_PLUGIN.format(marker=MARKER.strip(), module=HOOK_MODULE,
                                              python=json.dumps(sys.executable or "python3")))


_CLINE_SCRIPT = """#!/bin/sh
# Agent VCR airbag ({marker}) — snapshot before every Cline tool call.
# Generated by `vcr airbag install cline`. If the interpreter is gone, it answers
# "no objection" instead of failing every tool call; fix with `vcr doctor --fix`.
test -x {python} || {{ echo '{{"cancel": false}}'; exit 0; }}
exec {command}
"""


def _install_cline(path: Path, statusline: bool = False) -> None:
    if sys.platform == "win32":
        raise ValueError("Cline hooks are not supported on Windows (Cline wants PowerShell "
                         ".ps1 hooks there)")
    if path.exists() and MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
        raise ValueError(f"{path} already exists and is not an Agent VCR hook; merge manually")
    python = _interpreter("default")
    _write_text(path, _CLINE_SCRIPT.format(marker=MARKER.strip(), python=python,
                                           command=f"{python} {MARKER}cline pre"))
    path.chmod(0o755)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _gemini_supports_ask() -> bool:
    version = gemini_version()
    return version is not None and version >= (0, 36)


def gemini_version() -> tuple[int, int] | None:
    exe = shutil.which("gemini")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20,
                             stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


_INSTALLERS: dict[str, Callable[..., None]] = {
    "claude": _install_claude, "gemini": _install_gemini, "cursor": _install_cursor,
    "copilot": _install_copilot, "codex": _install_codex, "openhands": _install_openhands,
    "devin": _install_devin, "cline": _install_cline, "opencode": _install_opencode,
}

#: Agents whose hook files are standalone (deleted whole on uninstall).
_STANDALONE = frozenset({"copilot", "cline", "opencode"})

ADAPTERS: dict[str, Adapter] = {
    "claude": Adapter("claude", "Claude Code", ("claude",), render_claude,
                      ".claude/settings.local.json", "~/.claude/settings.json",
                      note="Hooks run in folders you trusted in Claude Code (its usual "
                           "first-run prompt).",
                      legacy_project=(".claude/settings.json",)),
    "gemini": Adapter("gemini", "Gemini CLI", ("gemini",), render_gemini,
                      ".gemini/settings.json", "~/.gemini/settings.json",
                      note="Gemini only runs hooks in trusted folders, user-level ones too.",
                      required_step="Gemini CLI: run /trust in each project folder once "
                                    "(Gemini skips every hook in untrusted folders)."),
    "cursor": Adapter("cursor", "Cursor", ("cursor", "cursor-agent"), render_cursor,
                      ".cursor/hooks.json", "~/.cursor/hooks.json",
                      note="Cursor also runs .claude/settings.json hooks; duplicates are "
                           "de-duplicated by the airbag."),
    "copilot": Adapter("copilot", "GitHub Copilot CLI", ("copilot",), render_copilot,
                       ".github/hooks/agent-vcr-airbag.json",
                       "{copilot}/hooks/agent-vcr-airbag.json",
                       note="Copilot loads hooks at startup; restart running sessions."),
    "codex": Adapter("codex", "OpenAI Codex CLI", ("codex",), render_codex,
                     ".codex/hooks.json", "~/.codex/hooks.json",
                     note="Codex has no 'ask', so irreversible commands are blocked until you "
                          "run `vcr airbag allow`.",
                     required_step="Codex: run /hooks in Codex once and trust the airbag hook "
                                   "(Codex skips new or changed hooks until you do)."),
    "openhands": Adapter("openhands", "OpenHands", ("openhands",), render_openhands,
                         ".openhands/hooks.json", "~/.openhands/hooks.json",
                         note="OpenHands reads ~/.openhands/hooks.json only when the project "
                              "has no .openhands/hooks.json. SDK agents must load it: "
                              "Conversation(..., hook_config=HookConfig.load(working_dir=ws)); "
                              "for confirmation prompts use "
                              "agent_vcr.integrations.openhands_airbag."),
    "devin": Adapter("devin", "Devin Local / Devin Desktop (ex-Windsurf)", ("devin",),
                     render_devin, ".devin/hooks.v1.json", "{devin}/config.json",
                     legacy_user=("~/.devin/hooks.v1.json",)),
    "opencode": Adapter("opencode", "OpenCode (plugin API v1)", ("opencode",), render_opencode,
                        ".opencode/plugin/agent-vcr-airbag.js",
                        "{config}/opencode/plugin/agent-vcr-airbag.js",
                        note="OpenCode plugins cannot ask, so irreversible commands are blocked "
                             "until `vcr airbag allow`. OpenCode 2.x plugins are not supported "
                             "yet."),
    "cline": Adapter("cline", "Cline", ("cline",), render_cline,
                     ".clinerules/hooks/PreToolUse", "{documents}/Cline/Hooks/PreToolUse",
                     note="Cline cancels the whole task for irreversible commands (it has no "
                          "'ask'). Cline CLI uses .cline/hooks/PreToolUse.sh instead."),
}


def install(name: str, root: str | Path, user: bool = False,
            statusline: bool = False) -> tuple[Path, str]:
    """Install ``name``'s hook (project-level unless ``user``); returns (path, note)."""
    adapter = ADAPTERS[name]
    path = adapter.config_path(root, user)
    _INSTALLERS[name](path, statusline=statusline)
    return path, adapter.note


def _remove_ours(name: str, path: Path, project: bool) -> bool:
    """Remove only the airbag's entries from ``path``; True when something was removed."""
    if not path.exists():
        return False
    if name in _STANDALONE:
        if MARKER.strip() not in path.read_text(errors="replace", encoding="utf-8"):
            return False
        path.unlink()
        _prune_dir(path.parent, project)
        return True
    try:
        settings = _read_json(path)
    except (ValueError, json.JSONDecodeError, OSError):
        return False
    if name == "devin" and path.name == "hooks.v1.json":
        holder = {"hooks": settings}
        if not _is_ours(holder):
            return False
        for event in list(holder["hooks"]):
            if isinstance(holder["hooks"].get(event), list):
                _merge_event_list(holder, event, None)
        settings = holder["hooks"]
    else:
        hooks = settings.get("hooks")
        ours_status = name == "claude" and _is_our_statusline(settings.get("statusLine"))
        has_hooks = isinstance(hooks, dict) and _is_ours(hooks)
        if not has_hooks and not ours_status:
            return False
        if has_hooks:
            for event in list(hooks or {}):
                _merge_event_list(settings, event, None)
            if not settings.get("hooks"):
                settings.pop("hooks", None)
        if ours_status:
            settings.pop("statusLine", None)
    if project and settings in ({}, {"version": 1}):
        path.unlink()  # nothing but what the airbag wrote: leave no file behind in the repo
        _prune_dir(path.parent, project)
    else:
        _write_json(path, settings)
    return True


def _prune_dir(directory: Path, project: bool) -> None:
    """Remove an emptied per-agent directory inside a project (``.codex/``, ...)."""
    if project:
        with contextlib.suppress(OSError):
            directory.rmdir()


def uninstall_paths(name: str, root: str | Path, user: bool = False) -> list[Path]:
    """Remove the airbag's hooks for ``name`` at one scope (incl. older locations)."""
    scope = "user" if user else "project"
    removed: list[Path] = []
    for where, path, _legacy in ADAPTERS[name].locations(root):
        if where == scope and path not in removed and _remove_ours(name, path, not user):
            removed.append(path)
    return removed


def remove_site(site: HookSite) -> bool:
    """Remove the airbag's entries from one found hook file."""
    return _remove_ours(site.agent, site.path, site.scope == "project")


def uninstall(name: str, root: str | Path, user: bool = False) -> Path | None:
    removed = uninstall_paths(name, root, user=user)
    return removed[0] if removed else None


@dataclass
class HookSite:
    """One installed airbag hook file."""

    agent: str
    scope: str  # "user" | "project"
    path: Path
    commands: list[str]
    #: Installed where an older version put it (e.g. 1.7.0's .claude/settings.json).
    legacy: bool = False

    @property
    def interpreters(self) -> list[str]:
        found = [hook_interpreter(c) for c in self.commands]
        return sorted({p for p in found if p})

    @property
    def missing_interpreters(self) -> list[str]:
        return [p for p in self.interpreters if not _runnable(p)]

    @property
    def guarded(self) -> bool:
        """Every command degrades to a no-op when its interpreter is gone (POSIX)."""
        return all(c.startswith("test -x ") or c.startswith("exec ") or self.agent == "opencode"
                   for c in self.commands)

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "scope": self.scope, "path": str(self.path),
                "interpreters": self.interpreters, "missing": self.missing_interpreters,
                "guarded": self.guarded, "legacy": self.legacy}


def _runnable(path: str) -> bool:
    candidate = path if os.path.isabs(path) else shutil.which(path)
    return bool(candidate) and os.path.isfile(str(candidate)) and \
        os.access(str(candidate), os.X_OK)


def _commands_in(name: str, path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace", encoding="utf-8")
    except OSError:
        return []
    if MARKER.strip() not in text and STATUSLINE_MODULE not in text:
        return []
    if name == "opencode":
        match = _OPENCODE_PYTHON.search(text)
        if match is None:
            return []
        with contextlib.suppress(json.JSONDecodeError):
            return [f"{shlex.quote(str(json.loads(match.group(1))))} {MARKER}opencode pre"]
        return []
    if name == "cline":
        return [line.strip() for line in text.splitlines()
                if line.startswith("exec ") and MARKER.strip() in line]
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and (MARKER.strip() in node or
                                        f"-m {STATUSLINE_MODULE}" in node):
            out.append(node)

    walk(data)
    return out


def find_hooks(root: str | Path) -> list[HookSite]:
    """Every airbag hook installed for ``root`` (project files) or for this user."""
    sites: list[HookSite] = []
    for name, adapter in ADAPTERS.items():
        seen: set[Path] = set()
        for scope, path, legacy in adapter.locations(root):
            if path in seen:
                continue
            seen.add(path)
            commands = _commands_in(name, path)
            if commands:
                sites.append(HookSite(name, scope, path, commands, legacy))
    return sites


def detect_available() -> list[str]:
    return [name for name, a in ADAPTERS.items() if any(shutil.which(b) for b in a.binaries)]


def detect_installed(root: str | Path) -> dict[str, str]:
    """``{agent: path}`` of one installed hook per agent (project first, then user)."""
    found: dict[str, str] = {}
    for site in sorted(find_hooks(root), key=lambda s: s.scope != "project"):
        found.setdefault(site.agent, str(site.path))
    return found
