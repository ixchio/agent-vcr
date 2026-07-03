"""Tests for the Agent VCR CLI helpers."""

from __future__ import annotations

import json

from agent_vcr import cli
from agent_vcr.cli import CLAUDE_CODE_EVENTS, write_claude_code_hooks
from agent_vcr.hooks.claude_code import record_hook_event
from agent_vcr.player import VCRPlayer


def test_init_claude_code_writes_expected_hooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    settings_path = write_claude_code_hooks(tmp_path)

    assert settings_path == tmp_path / ".claude" / "settings.json"
    data = json.loads(settings_path.read_text())
    assert set(CLAUDE_CODE_EVENTS).issubset(data["hooks"])

    for event in CLAUDE_CODE_EVENTS:
        entries = data["hooks"][event]
        assert entries
        assert entries[0]["hooks"][0]["type"] == "command"
        assert f"agent_vcr.hooks.claude_code {event}" in entries[0]["hooks"][0]["command"]


def test_init_claude_code_is_idempotent(tmp_path):
    write_claude_code_hooks(tmp_path)
    write_claude_code_hooks(tmp_path)

    data = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    for event in CLAUDE_CODE_EVENTS:
        commands = [hook["command"] for entry in data["hooks"][event] for hook in entry["hooks"]]
        assert commands.count(f"python -m agent_vcr.hooks.claude_code {event}") == 1


def test_claude_code_hook_records_vcr_frame(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = record_hook_event(
        "PostToolUse",
        {
            "session_id": "claude-session",
            "tool_name": "Write",
            "file_path": "src/app.py",
            "cwd": str(tmp_path),
        },
    )

    player = VCRPlayer.load(path)
    assert player.session.session_id == "claude-code-hooks"
    assert len(player.frames) == 1
    assert player.frames[0].node_name == "claude:PostToolUse"
    assert player.frames[0].output_state["tool_name"] == "Write"
    assert player.frames[0].output_state["file_path"] == "src/app.py"


def test_cli_keeps_vcr_file_shortcut(monkeypatch):
    launched = []
    monkeypatch.setattr(cli, "_run_tui", lambda filepath: launched.append(filepath))

    cli.main(["run.vcr"])

    assert launched == ["run.vcr"]
