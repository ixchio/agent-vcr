"""Airbag hook latency guards: the hook runs before every agent tool call.

The first snapshot of a project used to take seconds on big repositories
(``git add`` with one pathspec per file is superlinear), stalling the agent's
first tool call. These benchmarks keep the hook path in check with hard limits
generous enough for shared CI runners; Windows and macOS get extra headroom.
Run locally:  python3 -m pytest tests/benchmarks/test_airbag_latency.py -v --no-cov
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.airbag import store as store_mod
from agent_vcr.airbag.core import Airbag, ToolCall
from agent_vcr.airbag.store import ShadowStore

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

SRC = str(Path(__file__).resolve().parents[2] / "src")
#: Slower process start and file creation on Windows/macOS runners.
SLACK = 3.0 if sys.platform == "win32" else 1.5 if sys.platform == "darwin" else 1.0
FILES = 5000  # tracked files in the fixture repository


def _git(cwd: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "-c", "user.email=b@example.com", "-c", "user.name=b", *args],
                   cwd=cwd, check=True, capture_output=True, env=env, timeout=120)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A committed repository: 50 dirs x 100 small files, plus ignored node_modules."""
    root = tmp_path_factory.mktemp("bench") / "repo"
    body = "".join(f"value_{i} = {i}\n" for i in range(20))
    for d in range(FILES // 100):
        folder = root / "src" / f"pkg{d:03d}"
        folder.mkdir(parents=True)
        for f in range(100):
            (folder / f"mod{f:03d}.py").write_text(f"# {d}/{f}\n{body}", encoding="utf-8")
    deps = root / "node_modules" / "dep"
    deps.mkdir(parents=True)
    for f in range(1000):
        (deps / f"f{f}.js").write_text("module.exports = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("node_modules/\n.env\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


class TestAirbagHookLatency:
    """Hard limits on what an agent waits for before a tool call runs."""

    def test_first_snapshot_of_5k_file_repo(self, benchmark: Any, repo: Path,
                                             tmp_path: Path) -> None:
        """First snapshot (every file is new to the store) must average < 2s."""
        homes = iter(range(1000))

        def fresh_store() -> tuple[tuple[ShadowStore], dict[str, Any]]:
            return (ShadowStore(repo, home=tmp_path / f"home{next(homes)}"),), {}

        snap = benchmark.pedantic(lambda store: store.snapshot(label="first"),
                                  setup=fresh_store, rounds=3, iterations=1)
        assert snap.files == FILES + 2  # + .gitignore and the precious (ignored) .env
        assert benchmark.stats["mean"] < 2.0 * SLACK

    def test_snapshot_after_one_change(self, benchmark: Any, repo: Path,
                                       tmp_path: Path) -> None:
        """A later snapshot after editing one file must average < 0.5s."""
        store = ShadowStore(repo, home=tmp_path / "home")
        store.snapshot(label="first")
        target = repo / "src" / "pkg000" / "mod000.py"
        original = target.read_text(encoding="utf-8")
        counter = iter(range(1000))

        def edit() -> tuple[tuple[()], dict[str, Any]]:
            target.write_text(f"{original}# edit {next(counter)}\n", encoding="utf-8")
            return (), {}

        try:
            benchmark.pedantic(lambda: store.snapshot(label="edit"), setup=edit,
                               rounds=5, iterations=1)
        finally:
            target.write_text(original, encoding="utf-8")
        assert benchmark.stats["mean"] < 0.5 * SLACK

    def test_read_only_hook_decision(self, benchmark: Any, repo: Path, tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """Deciding that a read-only command needs nothing must average < 0.25s."""
        monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("AGENT_VCR_AIRBAG", raising=False)
        call = ToolCall(agent="bench", tool_name="Bash", kind="shell", cwd=str(repo),
                        command="ls -la src && git status && grep -rn value_3 src | head")
        decision = benchmark.pedantic(lambda: Airbag.for_call(call).before(call),
                                      rounds=20, iterations=1, warmup_rounds=1)
        assert decision.action == "continue" and decision.snapshot is None
        assert not (tmp_path / "home").exists() or not any((tmp_path / "home").iterdir())
        assert benchmark.stats["mean"] < 0.25 * SLACK

    def test_read_only_hook_process(self, benchmark: Any, repo: Path, tmp_path: Path) -> None:
        """The whole hook process (Python start + imports + decision) must average < 1.5s."""
        env = {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
               "AGENT_VCR_AIRBAG_HOME": str(tmp_path / "home")}
        env.pop("CLAUDE_PROJECT_DIR", None)
        payload = json.dumps({
            "session_id": "bench", "cwd": str(repo), "hook_event_name": "PreToolUse",
            "permission_mode": "default", "tool_name": "Bash", "tool_use_id": "t1",
            "tool_input": {"command": "cat README.md | wc -l"},
        })

        def run_hook() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-m", "agent_vcr.airbag.hook", "claude", "pre"],
                input=payload, capture_output=True, encoding="utf-8", timeout=60,
                cwd=str(repo), env=env,
            )

        proc = benchmark.pedantic(run_hook, rounds=5, iterations=1, warmup_rounds=1)
        assert proc.returncode == 0 and "Traceback" not in proc.stderr, proc.stderr
        assert benchmark.stats["mean"] < 1.5 * SLACK


@pytest.fixture(scope="module")
def app_with_ignored_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small app with what tends to be git-ignored: a 5 MB SQLite database in
    instance/, a 20k-file vendor/ tree and a 5k-file data/ folder with a .env."""
    root = tmp_path_factory.mktemp("bench-ignored") / "app"
    (root / "instance").mkdir(parents=True)
    con = sqlite3.connect(str(root / "instance" / "app.db"))
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v BLOB)")
    con.executemany("INSERT INTO t (v) VALUES (?)", [(os.urandom(1000),) for _ in range(5000)])
    con.commit()
    con.close()
    for top, count in (("vendor", 20000), ("data", 5000)):
        for i in range(count):
            folder = root / top / f"pkg{i // 500:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"f{i}.txt").write_text("x\n", encoding="utf-8")
    (root / "data" / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    (root / "vendor" / "pkg000" / ".env").write_text("THEIRS=1\n", encoding="utf-8")
    (root / "app.py").write_text("print(1)\n", encoding="utf-8")
    (root / ".gitignore").write_text("instance/\nvendor/\ndata/\n.env\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


class TestAirbagIgnoredDataLatency:
    """What ignored databases and folders cost on every tool call."""

    def test_snapshot_with_unchanged_sqlite_and_big_ignored_trees(
            self, benchmark: Any, app_with_ignored_data: Path, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """Later snapshot, database untouched, after one edit: must average < 0.3s."""
        monkeypatch.setattr(store_mod, "_SQLITE_RACY_NS", 0)  # the db is fresh in CI
        root = app_with_ignored_data
        time.sleep(0.05)
        store = ShadowStore(root, home=tmp_path / "home")
        first = store.snapshot(label="first")
        store.snapshot(label="warm")
        counter = iter(range(1000))
        backups: list[str] = []
        real_connect = sqlite3.connect

        def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
            backups.append(str(database))
            return real_connect(database, *args, **kwargs)

        monkeypatch.setattr(store_mod.sqlite3, "connect", connect)

        def edit() -> tuple[tuple[()], dict[str, Any]]:
            (root / "app.py").write_text(f"print({next(counter)})\n", encoding="utf-8")
            return (), {}

        snap = benchmark.pedantic(lambda: store.snapshot(label="edit"), setup=edit,
                                  rounds=5, iterations=1)
        tree = store._git("ls-tree", "-r", "--name-only", snap.tree).stdout.decode().split()
        assert {"instance/app.db", "data/.env"} <= set(tree)
        assert not any(name.startswith("vendor/") for name in tree)
        assert first.meta.get("notes") is None
        assert backups == []  # the database backup was reused every time
        assert benchmark.stats["mean"] < 0.3 * SLACK
