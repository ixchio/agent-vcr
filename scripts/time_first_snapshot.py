"""TEMPORARY (CI timing probe): first airbag snapshot of a 20k-file project."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

from agent_vcr.airbag.store import ShadowStore


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="vcr-timing-"))
    root = base / "proj"
    for d in range(200):
        folder = root / f"pkg{d:03d}"
        folder.mkdir(parents=True)
        for f in range(100):
            (folder / f"mod{f:03d}.py").write_bytes(f"# {d} {f}\nVALUE = {d * f}\n".encode())
    (root / ".gitignore").write_bytes(b".env\n")
    (root / ".env").write_bytes(b"E=1\n")
    for attempt in ("cold", "warm-home"):
        home = base / f"home-{attempt}"
        start = time.perf_counter()
        snap = ShadowStore(root, home=home).snapshot(label="first")
        first = time.perf_counter() - start
        start = time.perf_counter()
        ShadowStore(root, home=home).snapshot(label="unchanged")
        again = time.perf_counter() - start
        print(f"TIMING {sys.platform} cpus={os.cpu_count()} {attempt}: first snapshot of "
              f"{snap.files} files {first:.2f}s, unchanged re-snapshot {again:.2f}s", flush=True)


if __name__ == "__main__":
    main()
