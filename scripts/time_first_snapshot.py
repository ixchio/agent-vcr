"""TEMPORARY (CI timing probe): first airbag snapshot of a 20k-file project."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agent_vcr.airbag import store as store_mod
from agent_vcr.airbag.store import ShadowStore

N_DIRS, PER_DIR = 200, 100


def make_project(base: Path) -> Path:
    root = base / "proj"
    for d in range(N_DIRS):
        folder = root / f"pkg{d:03d}"
        folder.mkdir(parents=True)
        for f in range(PER_DIR):
            (folder / f"mod{f:03d}.py").write_bytes(f"# {d} {f}\nVALUE = {d * f}\n".encode())
    (root / ".gitignore").write_bytes(b".env\n")
    (root / ".env").write_bytes(b"E=1\n")
    return root


def timed_snapshot(root: Path, home: Path, label: str, extra_config: list[str]) -> None:
    phases: dict[str, float] = {}
    original_git = ShadowStore._git
    original_prewrite = ShadowStore._prewrite_blobs

    def git(self: ShadowStore, *args: str, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original_git(self, *extra_config, *args, **kwargs)
        finally:
            key = "git " + args[0]
            phases[key] = phases.get(key, 0.0) + time.perf_counter() - start

    def prewrite(self: ShadowStore, regular: list[str]) -> None:
        start = time.perf_counter()
        try:
            original_prewrite(self, regular)
        finally:
            phases["prewrite"] = time.perf_counter() - start

    ShadowStore._git = git  # type: ignore[method-assign]
    ShadowStore._prewrite_blobs = prewrite  # type: ignore[method-assign]
    try:
        start = time.perf_counter()
        try:
            snap = ShadowStore(root, home=home, git_timeout=900).snapshot(label="first")
            files = snap.files
        except Exception as err:  # noqa: BLE001
            files = -1
            print("  error:", type(err).__name__, str(err)[:200])
        total = time.perf_counter() - start
    finally:
        ShadowStore._git = original_git  # type: ignore[method-assign]
        ShadowStore._prewrite_blobs = original_prewrite  # type: ignore[method-assign]
    top = ", ".join(f"{k} {v:.1f}s" for k, v in sorted(phases.items(), key=lambda kv: -kv[1])[:5])
    print(f"TIMING {sys.platform} cpus={os.cpu_count()} [{label}] {files} files: "
          f"{total:.1f}s ({top})", flush=True)


def raw_git_add(root: Path, base: Path, label: str, config: list[str]) -> None:
    git_dir = base / ("raw-" + label.replace(" ", "-"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({"GIT_DIR": str(git_dir), "GIT_WORK_TREE": str(root),
                "GIT_INDEX_FILE": str(git_dir / "index"), "GIT_CONFIG_NOSYSTEM": "1"})
    subprocess.run(["git", "init", "-q", "--bare", str(git_dir)], check=True)
    subprocess.run(["git", "--git-dir", str(git_dir), "config", "core.bare", "false"],
                   check=True)
    start = time.perf_counter()
    subprocess.run(["git", *config, "add", "-A"], cwd=root, env=env, check=True, timeout=900)
    print(f"TIMING {sys.platform} [raw git add -A, {label}]: "
          f"{time.perf_counter() - start:.1f}s", flush=True)


def main() -> None:
    bases = [("system-temp", Path(tempfile.mkdtemp(prefix="vcr-timing-")))]
    if os.environ.get("RUNNER_TEMP"):
        bases.append(("runner-temp", Path(tempfile.mkdtemp(dir=os.environ["RUNNER_TEMP"]))))
    fscache = ["-c", "core.fscache=true"] if sys.platform == "win32" else []
    for where, base in bases:
        start = time.perf_counter()
        root = make_project(base)
        print(f"TIMING {sys.platform} [{where}] writing 20k files: "
              f"{time.perf_counter() - start:.1f}s", flush=True)
        timed_snapshot(root, base / "h1", f"{where} default", [])
        jobs = store_mod._PREWRITE_JOBS
        store_mod._PREWRITE_JOBS = 1  # prewrite off
        timed_snapshot(root, base / "h2", f"{where} prewrite-off", [])
        store_mod._PREWRITE_JOBS = jobs
        if fscache:
            timed_snapshot(root, base / "h3", f"{where} fscache", fscache)
        raw_git_add(root, base, f"{where} plain", [])
        if fscache:
            raw_git_add(root, base, f"{where} fscache", fscache)


if __name__ == "__main__":
    main()
