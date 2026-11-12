"""``vcr airbag statusline``: one line for Claude Code's ``statusLine``.

Claude Code runs the status line command after each message with a JSON
object on stdin (``workspace.current_dir``, ``cwd``, ...). This prints e.g.
``🛟 airbag · 12 snapshots · last: rm -rf build (2m ago)`` or
``🛟 airbag off here (not a git repo)``.

It must be fast (well under 100 ms) and must never change anything: no
snapshot, no store creation; the only subprocess is the read-only
``git rev-parse`` that finds the project. It reads the store's refs and the
tail of its log directly. Run as ``python -m agent_vcr.airbag.statusline`` (what
``vcr airbag install claude --statusline`` writes) it skips the CLI parser.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from agent_vcr._platform import utf8_stdio

ICON = "🛟"


def read_payload(stream: Any = None, timeout: float = 0.2) -> dict[str, Any]:
    """Claude Code's statusLine JSON from stdin; ``{}`` for a terminal or no input."""
    stream = stream if stream is not None else sys.stdin
    if stream is None:
        return {}
    with contextlib.suppress(OSError, ValueError, AttributeError):
        if stream.isatty():
            return {}
    if os.name == "posix":
        with contextlib.suppress(OSError, ValueError, AttributeError, TypeError):
            import select

            ready, _, _ = select.select([stream], [], [], timeout)
            if not ready:
                return {}
    try:
        raw = stream.read()
    except (OSError, ValueError):
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        data = json.loads(raw) if raw and raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def directory_of(payload: dict[str, Any]) -> str:
    workspace = payload.get("workspace")
    current = workspace.get("current_dir") if isinstance(workspace, dict) else None
    for value in (current, payload.get("cwd")):
        if isinstance(value, str) and value.strip():
            return value
    return os.getcwd()


def _ago(ts: float) -> str:
    delta = max(0.0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return f"{int(delta)}s ago"


def count_snapshots(git_dir: Path) -> int:
    """Snapshot refs (``refs/airbag/<n>``), loose and packed, without running git."""
    names: set[str] = set()
    loose = git_dir / "refs" / "airbag"
    with contextlib.suppress(OSError):
        names.update(p.name for p in loose.iterdir() if p.is_file())
    with contextlib.suppress(OSError):
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].startswith("refs/airbag/"):
                names.add(parts[1].rsplit("/", 1)[1])
    return len(names)


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


#: Same values as ``core.Policy.from_env`` (not imported: the classifier is slow to load).
_OFF = frozenset({"0", "off", "false", "no", "disabled"})


def _describe(event: dict[str, Any], root: Path) -> str:
    command = str(event.get("command") or "").strip()
    if command:
        return command
    shown = []
    for raw in list(event.get("paths") or [])[:2]:
        path = str(raw)
        with contextlib.suppress(ValueError):
            rel = os.path.relpath(path, root)
            path = path if rel.startswith("..") else rel
        shown.append(path.replace(os.sep, "/"))
    return " ".join(p for p in (str(event.get("tool") or ""), ", ".join(shown)) if p) or "?"


def render(directory: str) -> str:
    from agent_vcr.airbag.scope import resolve
    from agent_vcr.airbag.store import AirbagError, ShadowStore

    if os.environ.get("AGENT_VCR_AIRBAG", "").lower() in _OFF:
        return f"{ICON} airbag disabled (AGENT_VCR_AIRBAG)"
    scope = resolve(directory)
    if not scope.protected:
        why = {"plain": "not a git repo", "missing": "folder is gone"}.get(
            scope.kind, scope.unsafe or scope.kind)
        return f"{ICON} airbag off here ({why})"
    try:
        store = ShadowStore(scope.root)
    except AirbagError:
        return f"{ICON} airbag off here"
    count = count_snapshots(store.git_dir)
    if not count:
        return f"{ICON} airbag · armed · no snapshots yet"
    parts = [f"{ICON} airbag", f"{count} snapshot{'s' * (count != 1)}"]
    last = next((e for e in reversed(store.recent_events())
                 if e.get("kind") == "action" and e.get("snapshot") is not None), None)
    if last is not None:
        what = _short(_describe(last, store.root), 40)
        parts.append(f"last: {what} ({_ago(float(last.get('time', 0)))})")
    return " · ".join(parts)


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    try:
        line = render(directory_of(read_payload()))
    except Exception:  # noqa: BLE001 - a status line must never show a traceback
        line = f"{ICON} airbag"
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
