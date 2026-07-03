"""Command-line entry point for Agent VCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CLAUDE_CODE_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "FileChanged",
    "Stop",
    "SessionEnd",
)


def build_claude_code_hooks() -> dict[str, list[dict[str, Any]]]:
    """Build a Claude Code hooks config that records lifecycle events to VCR."""
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event in CLAUDE_CODE_EVENTS:
        entry: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": f"python -m agent_vcr.hooks.claude_code {event}",
                }
            ]
        }
        if event in {"PreToolUse", "PostToolUse"}:
            entry["matcher"] = "*"
        hooks[event] = [entry]
    return hooks


def write_claude_code_hooks(project_dir: str | Path = ".") -> Path:
    """Create or update .claude/settings.json with Agent VCR hooks."""
    root = Path(project_dir)
    claude_dir = root / ".claude"
    settings_path = claude_dir / "settings.json"
    claude_dir.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {}
    if settings_path.exists() and settings_path.stat().st_size > 0:
        with open(settings_path) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{settings_path} must contain a JSON object")
        settings = loaded

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{settings_path} field 'hooks' must be a JSON object")

    for event, entries in build_claude_code_hooks().items():
        existing = hooks.setdefault(event, [])
        if not isinstance(existing, list):
            raise ValueError(f"{settings_path} hooks.{event} must be a JSON array")

        existing_commands = {
            hook.get("command")
            for entry in existing
            if isinstance(entry, dict)
            for hook in entry.get("hooks", [])
            if isinstance(hook, dict)
        }
        for entry in entries:
            command = entry["hooks"][0]["command"]
            if command not in existing_commands:
                existing.append(entry)

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    return settings_path


def _run_tui(filepath: str) -> None:
    from agent_vcr.tui import VCRApp

    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    app = VCRApp(str(path))
    app.run()


def main(argv: list[str] | None = None) -> None:
    """Run the Agent VCR CLI."""
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if raw_args and raw_args[0] not in {"init", "tui", "-h", "--help"}:
        _run_tui(raw_args[0])
        return

    parser = argparse.ArgumentParser(description="Agent VCR")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize Agent VCR integrations")
    init_parser.add_argument(
        "--claude-code",
        action="store_true",
        help="Install Claude Code hooks into .claude/settings.json",
    )
    init_parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to initialize (default: current directory)",
    )

    tui_parser = subparsers.add_parser("tui", help="Open the terminal debugger")
    tui_parser.add_argument("file", help="Path to the .vcr file to replay")

    args = parser.parse_args(raw_args)

    if args.command == "init":
        if not args.claude_code:
            parser.error("init requires --claude-code")
        path = write_claude_code_hooks(args.project_dir)
        print(f"Claude Code hooks installed: {path}")
        return

    if args.command == "tui":
        _run_tui(args.file)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
