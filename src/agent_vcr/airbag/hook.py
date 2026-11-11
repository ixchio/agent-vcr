"""Hook entry point: ``python -m agent_vcr.airbag.hook <agent> [pre|post]``.

Reads the agent's hook JSON on stdin, runs the airbag, and answers in that
agent's own format. Contract with every agent: this process must never
crash the agent and never grant a permission — any internal error results
in "no opinion" (exit 0, no decision), except that an irreversible command
is still escalated.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    agent = args[0] if args else "claude"
    event = args[1] if len(args) > 1 else "pre"
    options = args[2] if len(args) > 2 else ""  # per-agent mode chosen by the installer
    raw = sys.stdin.read()
    try:
        payload: Any = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print("agent-vcr airbag: hook input was not JSON; doing nothing", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        from agent_vcr.airbag.agents import ADAPTERS
        from agent_vcr.airbag.core import Airbag

        adapter = ADAPTERS.get(agent)
        if adapter is None:
            print(f"agent-vcr airbag: unknown agent {agent!r}", file=sys.stderr)
            return 0
        if event != "pre":
            return 0  # post-tool events need no action: the next pre-snapshot captures results
        call = adapter.parse(payload)
        if call is None:
            return 0
        decision = Airbag.for_call(call).before(call)
        out, code = adapter.render(decision, call, options)
    except Exception:  # noqa: BLE001 - never break the agent
        print("agent-vcr airbag: internal error (tool call allowed to continue):",
              file=sys.stderr)
        traceback.print_exc(limit=3, file=sys.stderr)
        return 0
    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
        sys.stdout.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
