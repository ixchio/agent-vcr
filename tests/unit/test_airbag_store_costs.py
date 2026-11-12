"""Per-snapshot costs that must stay bounded: unreadable precious files, the
precious-file search inside ignored directories, and SQLite backups.

Real git, real SQLite, temporary directories.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.airbag import store as store_mod
from agent_vcr.airbag.store import ShadowStore, Snapshot

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

WINDOWS = sys.platform == "win32"


def _write(path: Path, text: str = "x\n", mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _names(store: ShadowStore, snap: Snapshot) -> set[str]:
    out = store._git("ls-tree", "-r", "--name-only", "-z", snap.tree).stdout
    return {raw.decode("utf-8", "surrogateescape") for raw in out.split(b"\0") if raw}


def _spy_fast_path(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    results: list[bool] = []
    original = ShadowStore._stage_with_update_index

    def spy(self: ShadowStore, *args: Any) -> bool:
        ok = original(self, *args)
        results.append(ok)
        return ok

    monkeypatch.setattr(ShadowStore, "_stage_with_update_index", spy)
    return results


# ── 1. an unreadable precious file must not disable the fast path ────────

def test_unreadable_precious_file_is_skipped_and_reported(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if WINDOWS or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("needs POSIX permissions and a non-root user")
    root = tmp_path / "proj"
    _write(root / "app.py", "v1\n")
    _write(root / ".gitignore", ".env\nsecrets/\n")
    _write(root / ".env", "OK=1\n")
    _write(root / "secrets" / ".env", "ROOT_ONLY=1\n", mode=0)
    calls = _spy_fast_path(monkeypatch)
    store = ShadowStore(root, home=tmp_path / "home")
    try:
        first = store.snapshot(label="first")
        _write(root / "app.py", "v2\n")
        second = store.snapshot(label="second")
    finally:
        (root / "secrets" / ".env").chmod(0o600)
    assert calls == [True, True]  # never the slow `git add` fallback
    assert ".env" in _names(store, first) and "app.py" in _names(store, second)
    assert "secrets/.env" not in _names(store, second)
    for snap in (first, second):
        assert snap.meta["skipped_unreadable"] == ["secrets/.env"]
        assert any("could not be read" in note for note in snap.meta["notes"])
    assert store.info()["notes"] == second.meta["notes"]
    _write(root / "app.py", "v3\n")  # readable again: picked up
    assert "secrets/.env" in _names(store, store.snapshot(label="third"))


# ── 2. precious search inside ignored directories ────────────────────────

def test_third_party_trees_are_not_searched(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    _write(root / "app.py")
    _write(root / ".gitignore", "vendor/\ndata/\npackages/*/lib/\n")
    _write(root / "vendor" / "pkg" / ".env", "THEIRS=1\n")
    _write(root / "packages" / "a" / "lib" / "Pods" / "x.db", "theirs\n")
    _write(root / "data" / ".env", "MINE=1\n")
    _write(root / "packages" / "a" / "package.json", "{}\n")
    _write(root / "packages" / "a" / "lib" / "cfg" / ".env", "MINE=2\n")
    store = ShadowStore(root, home=tmp_path / "home")
    names = _names(store, store.snapshot(label="first"))
    assert {"data/.env", "packages/a/lib/cfg/.env"} <= names
    assert "vendor/pkg/.env" not in names and "packages/a/lib/Pods/x.db" not in names


def test_precious_search_is_bounded_and_resumes(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_PRECIOUS_DIR_LIMIT", 3)
    root = tmp_path / "proj"
    _write(root / "app.py", "0\n")
    _write(root / ".gitignore", "packages/*/lib/\n")
    for n in range(7):
        _write(root / "packages" / f"p{n}" / "package.json", "{}\n")
        _write(root / "packages" / f"p{n}" / "lib" / "cfg" / ".env", f"N={n}\n")
    store = ShadowStore(root, home=tmp_path / "home")
    wanted = {f"packages/p{n}/lib/cfg/.env" for n in range(7)}
    found: set[str] = set()
    for step in range(3):
        _write(root / "app.py", f"{step}\n")
        snap = store.snapshot(label=f"s{step}")
        found |= _names(store, snap) & wanted
        if step == 0:
            assert len(found) == 3
            assert any("in 3 of 7 ignored folders" in n for n in snap.meta["notes"])
            assert "in 3 of 7" in store.info()["notes"][0]
    assert found == wanted  # every directory is searched within a few snapshots
    assert not snap.meta.get("notes")


def test_precious_search_respects_the_time_budget(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_mod, "_PRECIOUS_TIME_BUDGET", 0.0)
    root = tmp_path / "proj"
    _write(root / ".gitignore", "d*/\n")
    for n in range(40):
        _write(root / f"d{n:02d}" / ".env", f"N={n}\n")
    _write(root / "app.py")
    store = ShadowStore(root, home=tmp_path / "home")
    snap = store.snapshot(label="first")
    per_call = 100 // len(store.precious)
    assert len(_names(store, snap)) == 2 + per_call  # .gitignore, app.py + one batch
    assert any(f"in {per_call} of 40" in n for n in snap.meta["notes"])


# ── 3. SQLite backups are reused while the database is unchanged ─────────

def _make_db(path: Path, *, wal: bool = False, rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO t (v) VALUES (?)", [(f"value-{i:04d}",) for i in range(rows)])
    con.commit()
    con.close()


def _db_values(store: ShadowStore, snap: Snapshot, rel: str, tmp: Path) -> list[str]:
    dest = tmp / f"checkout-{snap.seq}"
    store.materialize(snap, dest)
    con = sqlite3.connect(str(dest / rel))
    try:
        return [row[0] for row in con.execute("SELECT v FROM t ORDER BY id")]
    finally:
        con.close()


def _count_backups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    made: list[str] = []
    real_connect = sqlite3.connect

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(database).startswith("file:") and "mode=ro" in str(database):
            made.append(str(database))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(store_mod.sqlite3, "connect", connect)
    return made


@pytest.fixture
def backups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count online backups; trust timestamps right away (no 2 s racy window)."""
    monkeypatch.setattr(store_mod, "_SQLITE_RACY_NS", 0)
    return _count_backups(monkeypatch)


def _tick() -> None:
    time.sleep(0.05)  # past the filesystem's timestamp granularity


@pytest.mark.parametrize("wal", [False, True], ids=["rollback", "wal"])
def test_unchanged_sqlite_is_not_backed_up_again(tmp_path: Path, backups: list[str],
                                                 wal: bool) -> None:
    root = tmp_path / "proj"
    _write(root / "app.py", "0\n")
    _write(root / ".gitignore", "instance/\n")
    _make_db(root / "instance" / "app.db", wal=wal)
    _tick()
    store = ShadowStore(root, home=tmp_path / "home")
    first = store.snapshot(label="first")
    assert len(backups) == 1
    for step in range(3):
        _write(root / "app.py", f"{step + 1}\n")
        snap = store.snapshot(label=f"s{step}")
        assert "instance/app.db" in _names(store, snap)
    # Reused from then on. In WAL mode the first read-only backup creates the
    # (empty) -wal file, so the second snapshot backs up once more.
    assert len(backups) == (2 if wal else 1)
    blob = store._git("rev-parse", f"{snap.tree}:instance/app.db").stdout
    assert blob == store._git("rev-parse", f"{first.tree}:instance/app.db").stdout


def test_sqlite_changes_are_never_missed(tmp_path: Path, backups: list[str]) -> None:
    root = tmp_path / "proj"
    rel = "instance/app.db"
    db = root / rel
    _write(root / ".gitignore", "instance/\n")
    _write(root / "app.py")
    _make_db(db)
    _tick()
    store = ShadowStore(root, home=tmp_path / "home")
    store.snapshot(label="first")
    expected = [f"value-{i:04d}" for i in range(200)]

    # In-place write of the same length: same size, same inode.
    size = db.stat().st_size
    con = sqlite3.connect(str(db))
    con.execute("UPDATE t SET v = 'VALUE-0007' WHERE id = 8")
    con.commit()
    con.close()
    assert db.stat().st_size == size
    _tick()
    snap = store.snapshot(label="in-place")
    expected[7] = "VALUE-0007"
    assert _db_values(store, snap, rel, tmp_path) == expected

    # In-place write with the old mtime put back (ctime still moves).
    st = db.stat()
    con = sqlite3.connect(str(db))
    con.execute("UPDATE t SET v = 'VALUE-0008' WHERE id = 9")
    con.commit()
    con.close()
    os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns))
    _tick()
    snap = store.snapshot(label="mtime-reset")
    expected[8] = "VALUE-0008"
    assert _db_values(store, snap, rel, tmp_path) == expected

    # WAL mode: the change lives only in -wal while a connection stays open.
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("UPDATE t SET v = 'VALUE-0009' WHERE id = 10")
    con.commit()
    try:
        _tick()
        snap = store.snapshot(label="wal")
        expected[9] = "VALUE-0009"
        assert _db_values(store, snap, rel, tmp_path) == expected
        con.execute("UPDATE t SET v = 'VALUE-0010' WHERE id = 11")
        con.commit()
        _tick()
        snap = store.snapshot(label="wal-2")
        expected[10] = "VALUE-0010"
        assert _db_values(store, snap, rel, tmp_path) == expected
    finally:
        con.close()

    # Replaced by a different file (new inode), then deleted.
    _make_db(root / "instance" / "new.db", rows=3)
    os.replace(root / "instance" / "new.db", db)
    _tick()
    snap = store.snapshot(label="replaced")
    assert _db_values(store, snap, rel, tmp_path) == [f"value-{i:04d}" for i in range(3)]
    db.unlink()
    for suffix in ("-wal", "-shm"):
        with pytest.raises(FileNotFoundError):
            os.stat(str(db) + suffix)
    assert rel not in _names(store, store.snapshot(label="deleted"))


def test_recently_written_sqlite_is_always_backed_up(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """With coarse timestamps a write right after the backup could keep the same
    mtime, so a database modified within the racy window (2 s) is never trusted."""
    made = _count_backups(monkeypatch)
    root = tmp_path / "proj"
    _write(root / ".gitignore", "*.db\n")
    _write(root / "app.py")
    _make_db(root / "app.db")
    store = ShadowStore(root, home=tmp_path / "home")
    for step in range(3):
        _write(root / "app.py", f"{step}\n")
        store.snapshot(label=f"s{step}")
    assert len(made) == 3
    entry = store._sqlite_cache()["app.db"]
    assert not store_mod._sqlite_unchanged(str(root / "app.db"), entry)


def test_restore_after_reused_sqlite_backup(tmp_path: Path, backups: list[str]) -> None:
    root = tmp_path / "proj"
    rel = "instance/app.db"
    _write(root / ".gitignore", "instance/\n")
    _write(root / "app.py", "0\n")
    _make_db(root / rel)
    _tick()
    store = ShadowStore(root, home=tmp_path / "home")
    good = store.snapshot(label="good")
    _write(root / "app.py", "1\n")
    store.snapshot(label="reused")
    con = sqlite3.connect(str(root / rel))
    con.execute("DELETE FROM t")
    con.commit()
    con.close()
    _tick()
    store.snapshot(label="wiped")
    store.restore(good)
    con = sqlite3.connect(str(root / rel))
    try:
        assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 200
    finally:
        con.close()
    _tick()
    _write(root / "app.py", "2\n")
    after = store.snapshot(label="after-restore")
    assert _db_values(store, after, rel, tmp_path)[:1] == ["value-0000"]
