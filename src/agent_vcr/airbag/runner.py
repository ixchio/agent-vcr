"""Protection for agents without a hook system: ``vcr airbag run -- <agent>``.

Takes a snapshot before the agent starts, then every ``interval`` seconds
while it runs (a no-op when nothing changed), and once more when it exits.
Each snapshot that captured new changes becomes an undo point. This cannot
intercept a command *before* it runs, so at most ``interval`` seconds of
work can be lost — use a real hook integration when the agent has one.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from agent_vcr.airbag.core import Airbag, Policy
from agent_vcr.airbag.store import AirbagError


def run_with_snapshots(argv: list[str], project_dir: str | Path = ".",
                       interval: float = 2.0) -> int:
    bag = Airbag(project_dir, policy=Policy.from_env())
    agent = Path(argv[0]).name
    stop = threading.Event()
    warned: list[str] = []

    def take(label: str) -> None:
        try:
            before = bag.store.latest()
            snap = bag.store.snapshot(label=label, meta={"agent": agent, "periodic": True})
        except (AirbagError, OSError) as err:
            if not warned:
                warned.append(str(err))
                print(f"[airbag] snapshot failed: {err}", file=sys.stderr)
            return
        if before is None or snap.seq != before.seq:
            bag.store.log_event({
                "kind": "action", "agent": agent, "tool": "periodic",
                "command": f"{label} ({shlex.join(argv)[:80]})", "level": "write",
                "snapshot": snap.seq,
            })

    take("before agent start")
    print(f"[airbag] protecting {bag.root} while `{shlex.join(argv)}` runs "
          f"(snapshot every {interval:g}s; undo with `vcr undo`)", file=sys.stderr)

    def loop() -> None:
        while not stop.wait(interval):
            take("periodic")

    thread = threading.Thread(target=loop, name="airbag-periodic", daemon=True)
    thread.start()
    env = {**os.environ, "AGENT_VCR_AIRBAG_ACTIVE": "1"}
    try:
        proc = subprocess.run(argv, env=env)
        code = proc.returncode
    except FileNotFoundError:
        print(f"[airbag] command not found: {argv[0]}", file=sys.stderr)
        code = 127
    except KeyboardInterrupt:
        code = 130
    finally:
        stop.set()
        thread.join(timeout=interval + 5)
        take("agent exited")
    return code
