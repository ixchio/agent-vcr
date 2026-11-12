"""Airbag adapters: real hook subprocess per agent, payloads from each agent's docs.

Payload shapes come from the official docs / installed sources recorded in
docs/airbag.md. Each test runs ``python -m agent_vcr.airbag.hook`` exactly
like the agent would (JSON on stdin, answer on stdout + exit code).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.airbag import agents
from agent_vcr.airbag.agents import ADAPTERS, install, parse_payload, uninstall
from agent_vcr.airbag.store import ShadowStore

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")
SRC = str(Path(__file__).resolve().parents[2] / "src")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
    return proj


def run_hook(agent: str, payload: dict[str, Any], project: Path, *extra: str,
             env: dict[str, str] | None = None) -> tuple[str, int]:
    full_env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
                **(env or {})}
    for var in ("CLAUDE_PROJECT_DIR", "GEMINI_PROJECT_DIR", "CURSOR_PROJECT_DIR",
                "OPENHANDS_PROJECT_DIR"):
        if env is None or var not in env:
            full_env.pop(var, None)
    proc = subprocess.run(
        [sys.executable, "-m", "agent_vcr.airbag.hook", agent, "pre", *extra],
        input=json.dumps(payload), capture_output=True, encoding="utf-8", timeout=60,
        cwd=str(project), env=full_env,
    )
    assert "Traceback" not in proc.stderr, proc.stderr
    return proc.stdout.strip(), proc.returncode


def snapshots(project: Path) -> int:
    return len(ShadowStore(project).snapshots())


# ── payload → answer, per agent ──────────────────────────────────────────

def test_claude_code(project: Path) -> None:
    base = {"session_id": "s1", "cwd": str(project), "hook_event_name": "PreToolUse",
            "permission_mode": "default", "tool_name": "Bash"}
    out, code = run_hook("claude", {**base, "tool_use_id": "t1",
                                    "tool_input": {"command": "rm -rf src"}}, project)
    assert code == 0 and "vcr undo" in json.loads(out)["systemMessage"]
    assert "hookSpecificOutput" not in json.loads(out)  # never grants permission
    assert snapshots(project) == 1
    out, _ = run_hook("claude", {**base, "tool_use_id": "t2",
                                 "tool_input": {"command": "git push --force"}}, project)
    decision = json.loads(out)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "ask" and decision["permissionDecisionReason"]
    out, _ = run_hook("claude", {**base, "tool_use_id": "t3",
                                 "tool_input": {"command": "ls -la"}}, project)
    assert out == ""


def test_claude_edit_tool_snapshots(project: Path) -> None:
    payload = {"session_id": "s", "cwd": str(project), "tool_name": "Write",
               "tool_input": {"file_path": str(project / "src" / "app.py"), "content": "x"}}
    out, code = run_hook("claude", payload, project)
    assert code == 0 and out == "" and snapshots(project) == 1


def test_codex_denies_instead_of_ask(project: Path) -> None:
    payload = {"session_id": "s", "turn_id": "u", "cwd": str(project),
               "hook_event_name": "PreToolUse", "model": "gpt-5", "permission_mode": "default",
               "tool_name": "Bash", "tool_input": {"command": "terraform destroy -auto-approve"},
               "tool_use_id": "c1"}
    out, code = run_hook("codex", payload, project)
    decision = json.loads(out)["hookSpecificOutput"]
    assert code == 0 and decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"]  # Codex fails open on an empty reason


def test_codex_apply_patch_paths(project: Path) -> None:
    patch = "*** Begin Patch\n*** Update File: src/app.py\n@@\n-print(1)\n+print(2)\n*** End Patch"
    call = parse_payload("codex", {"cwd": str(project), "tool_name": "apply_patch",
                                   "tool_input": {"command": patch}})
    assert call is not None and call.kind == "edit" and call.paths == ["src/app.py"]


def test_gemini_ask_and_deny_modes(project: Path) -> None:
    payload = {"session_id": "s", "cwd": str(project), "hook_event_name": "BeforeTool",
               "timestamp": "2026-11-11T00:00:00Z", "tool_name": "run_shell_command",
               "tool_input": {"command": "npm publish", "dir_path": str(project)}}
    out, _ = run_hook("gemini", payload, project, "ask")
    assert json.loads(out)["decision"] == "ask"
    out, _ = run_hook("gemini", {**payload, "session_id": "s2",
                                 "tool_input": {"command": "npm unpublish x"}}, project, "deny")
    answer = json.loads(out)
    assert answer["decision"] == "deny" and answer["reason"]


def test_gemini_write_file(project: Path) -> None:
    payload = {"session_id": "s", "cwd": str(project), "tool_name": "write_file",
               "tool_input": {"file_path": "src/app.py", "content": "x"}}
    out, code = run_hook("gemini", payload, project, "ask")
    assert code == 0 and out == "" and snapshots(project) == 1


def test_cursor_shell_must_always_answer(project: Path) -> None:
    common = {"conversation_id": "c", "generation_id": "g", "hook_event_name":
              "beforeShellExecution", "workspace_roots": [str(project)], "sandbox": False}
    out, code = run_hook("cursor", {**common, "command": "pytest -q", "cwd": str(project)},
                         project, "shell")
    assert code == 0 and json.loads(out) == {"permission": "allow"}  # empty output would block
    out, _ = run_hook("cursor", {**common, "command": "rm -rf ~/elsewhere",
                                 "cwd": str(project)}, project, "shell")
    answer = json.loads(out)
    assert answer["permission"] == "deny" and answer["agent_message"]


def test_copilot_tool_args_is_a_json_string(project: Path) -> None:
    payload = {"sessionId": "s", "timestamp": 1704614400000, "cwd": str(project),
               "toolName": "bash", "toolArgs": json.dumps({"command": "git push -f"})}
    out, code = run_hook("copilot", payload, project)
    answer = json.loads(out.splitlines()[-1])
    assert code == 0 and answer["permissionDecision"] == "ask"
    out, _ = run_hook("copilot", {**payload, "toolArgs": json.dumps({"command": "rm -rf src"})},
                      project)
    assert json.loads(out)["type"] == "progress"


def test_openhands_sdk_hook(project: Path) -> None:
    payload = {"event_type": "PreToolUse", "tool_name": "terminal",
               "tool_input": {"command": "psql -h db.prod -c 'DROP TABLE x'", "kind":
                              "TerminalAction"}, "session_id": "s", "working_dir": str(project)}
    out, code = run_hook("openhands", payload, project,
                         env={"OPENHANDS_PROJECT_DIR": str(project)})
    assert code == 0 and json.loads(out)["decision"] == "deny"
    view = {**payload, "tool_name": "file_editor",
            "tool_input": {"command": "view", "path": "src/app.py"}}
    assert run_hook("openhands", view, project)[0] == ""


def test_cline_vscode_and_cli_shapes(project: Path) -> None:
    vscode = {"clineVersion": "3.40", "hookName": "PreToolUse", "taskId": "t",
              "workspaceRoots": [str(project)],
              "preToolUse": {"toolName": "run_commands",
                             "parameters": {"commands": ["ls", "kubectl delete ns prod"]}}}
    out, _ = run_hook("cline", vscode, project)
    assert json.loads(out)["cancel"] is True
    cli = {"hookName": "tool_call", "workspaceRoots": [str(project)],
           "tool_call": {"id": "x", "name": "editor", "input": {"path": "src/app.py",
                                                               "old_text": "1", "new_text": "2"}}}
    out, _ = run_hook("cline", cli, project)
    assert json.loads(out) == {"cancel": False}


def test_devin_blocks(project: Path) -> None:
    payload = {"hook_event_name": "PreToolUse", "tool_name": "exec", "session_id": "s",
               "cwd": str(project), "tool_input": {"command": "gh repo delete me/x --yes"}}
    out, _ = run_hook("devin", payload, project)
    assert json.loads(out)["decision"] == "block"


def test_mcp_and_unknown_tools_are_ignored(project: Path) -> None:
    for payload in ({"tool_name": "mcp__github__create_issue", "tool_input": {}},
                    {"tool_name": "WebFetch", "tool_input": {"url": "https://x"}}):
        assert run_hook("claude", {**payload, "cwd": str(project)}, project) == ("", 0)


def test_hook_never_breaks_the_agent(project: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "agent_vcr.airbag.hook", "claude", "pre"],
                          input="not json", capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": SRC}, encoding="utf-8", errors="replace")
    assert proc.returncode == 0 and proc.stdout == ""
    home = project.parent / "home-is-a-file"
    home.write_text("x", encoding="utf-8")
    out, code = run_hook("claude", {"cwd": str(project), "tool_name": "Bash",
                                    "tool_input": {"command": "rm -rf src"}}, project,
                         env={"AGENT_VCR_AIRBAG_HOME": str(home)})
    assert code == 0 and "could not snapshot" in json.loads(out)["systemMessage"]


# ── installers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(set(ADAPTERS) - {"cline", "opencode"}))
def test_install_is_idempotent_and_keeps_user_hooks(name: str, project: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agents, "_gemini_supports_ask", lambda: True)
    path = ADAPTERS[name].config_path(project, user=False)
    user_hook: dict[str, Any] | None = None
    if name in {"claude", "codex", "openhands"}:
        user_hook = {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-lint"}]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"hooks": {"PreToolUse": [user_hook]}, "other": 1}),
                        encoding="utf-8")
    install(name, project)
    install(name, project)
    text = path.read_text(encoding="utf-8")
    # cursor: shell + tool hooks; copilot: bash + powershell variants of one hook
    assert text.count(agents.MARKER.strip()) == (2 if name in {"cursor", "copilot"} else 1)
    data = json.loads(text)
    if user_hook is not None:
        assert user_hook in data["hooks"]["PreToolUse"] and data["other"] == 1
    assert uninstall(name, project) == path
    remaining = path.read_text(encoding="utf-8") if path.exists() else ""
    assert agents.MARKER.strip() not in remaining
    if user_hook is not None:
        assert user_hook in json.loads(remaining)["hooks"]["PreToolUse"]


@pytest.mark.skipif(sys.platform == "win32", reason="Cline hooks are POSIX scripts")
def test_cline_installs_an_executable_script(project: Path) -> None:
    path, _ = install("cline", project)
    assert os.access(path, os.X_OK) and agents.MARKER.strip() in path.read_text(encoding="utf-8")
    proc = subprocess.run([str(path)], input=json.dumps({
        "workspaceRoots": [str(project)],
        "preToolUse": {"toolName": "execute_command", "parameters": {"command": "rm -rf src"}},
    }), capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": SRC}, encoding="utf-8", errors="replace")
    assert json.loads(proc.stdout)["cancel"] is False
    assert uninstall("cline", project) == path and not path.exists()


def _run_in_shell(argv_or_command: Any, project: Path, shell: bool,
                  payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        payload = {"cwd": str(project), "tool_name": "Bash",
                   "tool_input": {"command": "rm -rf src"}}
    proc = subprocess.run(argv_or_command, shell=shell, input=json.dumps(payload),
                          capture_output=True, encoding="utf-8", timeout=60, cwd=str(project),
                          env={**os.environ, "PYTHONPATH": SRC})
    assert proc.returncode == 0, proc.stderr
    return dict(json.loads(proc.stdout))


def test_hook_command_uses_this_interpreter(project: Path) -> None:
    command = agents.hook_command("claude")
    if sys.platform == "win32":
        # Forward slashes, double quotes only when needed: valid in Git Bash, cmd and pwsh.
        assert command.lstrip('"').startswith(sys.executable.replace(os.sep, "/"))
    else:
        assert command.startswith(sys.executable if " " not in sys.executable else "'")
    # The command really runs in the platform shell (/bin/sh, or cmd.exe on Windows).
    assert "vcr undo" in _run_in_shell(command, project, shell=True)["systemMessage"]
    assert snapshots(project) == 1


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell") is None,
                    reason="Windows PowerShell")
def test_powershell_hook_command_runs_under_powershell(project: Path) -> None:
    command = agents.hook_command("copilot", shell="powershell")
    payload = {"sessionId": "s", "timestamp": 1704614400000, "cwd": str(project),
               "toolName": "powershell", "toolArgs": json.dumps({"command": "rm -rf src"})}
    out = _run_in_shell(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                        project, shell=False, payload=payload)
    assert out["type"] == "progress"
    assert snapshots(project) == 1


_DRIVER = """
const [plugin, dir] = process.argv.slice(2);
const { pathToFileURL } = await import("node:url");
const mod = await import(pathToFileURL(plugin).href);
const hooks = await mod.AgentVcrAirbag({ directory: dir, worktree: dir });
const before = hooks["tool.execute.before"];
await before({ tool: "bash", sessionID: "s", callID: "c1" }, { args: { command: "rm -rf src" } });
await before({ tool: "read", sessionID: "s", callID: "c2" }, { args: { filePath: "x" } });
try {
  await before({ tool: "bash", sessionID: "s", callID: "c3" }, { args: { command: "npm publish" } });
  console.log("NOT-BLOCKED");
} catch (e) { console.log("BLOCKED " + e.message); }
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node required to run the plugin")
def test_opencode_plugin_runs_under_node(project: Path, tmp_path: Path) -> None:
    path, _ = install("opencode", project)
    driver = tmp_path / "drive.mjs"
    driver.write_text(_DRIVER, encoding="utf-8")
    proc = subprocess.run(["node", str(driver), str(path), str(project)], capture_output=True,
                          text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": SRC + os.pathsep +
                               os.environ.get("PYTHONPATH", "")},
                                   encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr
    assert "BLOCKED Agent VCR airbag" in proc.stdout
    assert snapshots(project) == 1
    assert uninstall("opencode", project) == path and not path.exists()


def test_opencode_payload_shapes(project: Path) -> None:
    edit = parse_payload("opencode", {"tool_name": "edit", "cwd": str(project),
                                      "tool_input": {"filePath": "src/app.py"}})
    patch = parse_payload("opencode", {"tool_name": "apply_patch", "cwd": str(project),
                                       "tool_input": {"patchText": "*** Add File: new.py\n+x"}})
    assert edit is not None and edit.paths == ["src/app.py"]
    assert patch is not None and patch.paths == ["new.py"]
