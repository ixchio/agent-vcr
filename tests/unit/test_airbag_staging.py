"""Snapshot staging: the fast ``update-index --stdin`` path must build exactly the
trees that Agent VCR 1.7.0 built with ``git add`` and literal pathspecs.

``Store170`` below is the 1.7.0 ``_snapshot_locked`` kept verbatim as a reference.
Each scenario snapshots the same project into two independent stores (one per
implementation) after every mutation step and compares ``git ls-tree -r`` (modes,
types, object ids, names), the recorded private modes and the snapshot metadata.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.airbag import store as store_mod
from agent_vcr.airbag.store import (
    DatabaseDump,
    ShadowStore,
    Snapshot,
    _literal,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

WINDOWS = sys.platform == "win32"
LIMIT = 64 * 1024  # max_file_bytes for both stores


def _can_symlink(tmp: Path) -> bool:
    try:
        os.symlink("x", tmp / ".symlink-probe")
    except (OSError, NotImplementedError):
        return False
    os.unlink(tmp / ".symlink-probe")
    return True


def _can_make_unreadable(tmp: Path) -> bool:
    if WINDOWS or (hasattr(os, "geteuid") and os.geteuid() == 0):
        return False
    probe = tmp / ".unreadable-probe"
    probe.write_text("x", encoding="utf-8")
    probe.chmod(0)
    try:
        with open(probe, "rb"):
            return False
    except OSError:
        return True
    finally:
        probe.chmod(0o600)
        probe.unlink()


class Store170(ShadowStore):
    """The 1.7.0 snapshot staging, unchanged (reference implementation)."""

    def _snapshot_locked(self, label: str, meta: dict[str, Any], *,
                         databases: list[DatabaseDump] | None = None) -> Snapshot:
        if not self.root.exists():
            self.root.mkdir(parents=True)
        skipped: list[str] = []
        changed = self._changed_paths()
        to_add: list[str] = []
        untracked_nested = [rel.rstrip("/") for rel in changed if rel.endswith("/")]
        for rel in changed:
            if rel.endswith("/"):
                continue
            full = self.root / rel
            try:
                too_big = (full.is_file() and not full.is_symlink()
                           and full.stat().st_size > self.max_file_bytes)
            except OSError:
                too_big = False
            if too_big:
                skipped.append(rel)
                continue
            to_add.append(rel)
        precious, too_big_precious = self._precious_ignored_files()
        if to_add:
            self._git(
                "add", "-A", "--ignore-errors", "--pathspec-from-file=-", "--pathspec-file-nul",
                input_bytes=b"\0".join(_literal(p) for p in to_add), check=False,
            )
        skipped.extend(too_big_precious)
        if skipped:
            self._git(
                "update-index", "-z", "--force-remove", "--stdin",
                input_bytes=b"".join(p.encode("utf-8", "surrogateescape") + b"\0"
                                     for p in skipped),
            )
        if precious:
            self._git(
                "add", "-f", "--ignore-errors", "--pathspec-from-file=-", "--pathspec-file-nul",
                input_bytes=b"\0".join(_literal(p) for p in precious), check=False,
            )
        nested = self._stage_sqlite_consistently()
        nested.extend(n for n in untracked_nested if n not in nested)
        self._remember_private_modes_170(to_add + precious)

        tree = self._git("write-tree").stdout.decode().strip()
        last = self.latest()
        if last is not None and last.tree == tree and not databases \
                and not last.meta.get("databases"):
            return last
        files = int(self._git("ls-files", "-z").stdout.count(b"\0"))
        seq = (last.seq + 1) if last is not None else 1
        record = {"seq": seq, "label": label, "time": 0.0, "files": files,
                  "skipped_large": skipped[:50], "nested_repos": nested[:50], **meta}
        commit = self._git(
            "commit-tree", tree, input_bytes=json.dumps(record, default=str).encode()
        ).stdout.decode().strip()
        self._git("update-ref", f"refs/airbag/{seq:08d}", commit)
        return Snapshot(seq=seq, commit=commit, tree=tree, time=0.0, label=label, files=files,
                        meta={"skipped_large": skipped[:50], "nested_repos": nested[:50],
                              **meta})

    def _remember_private_modes_170(self, paths: list[str]) -> None:
        private: dict[str, int] = {}
        for rel in paths:
            with contextlib.suppress(OSError):
                st = os.lstat(self.root / rel)
                if stat.S_ISREG(st.st_mode) and not st.st_mode & 0o077:
                    private[rel] = stat.S_IMODE(st.st_mode)
        if not private:
            return
        modes = self._private_modes()
        if all(modes.get(k) == v for k, v in private.items()):
            return
        modes.update(private)
        (self.dir / "modes.json").write_text(json.dumps(modes), encoding="utf-8")


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "-c", "user.email=u@example.com", "-c", "user.name=u", *args], cwd=cwd,
        check=True, capture_output=True, text=True, env=env, encoding="utf-8",
        errors="replace").stdout


def _write(path: Path, text: str | bytes = "x\n", mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _sqlite(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(rows)])
    con.commit()
    con.close()


def _nested_repo(path: Path, commit: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _write(path / "inside.txt", "nested\n")
    if commit:
        _git(path, "add", "-A")
        _git(path, "commit", "-qm", "nested")


class Pair:
    """The same project snapshotted by the 1.7.0 reference and by the current store."""

    def __init__(self, root: Path, homes: Path) -> None:
        self.old = Store170(root, home=homes / "old", max_file_bytes=LIMIT)
        self.new = ShadowStore(root, home=homes / "new", max_file_bytes=LIMIT)

    def set_config(self, key: str, value: str) -> None:
        for st in (self.old, self.new):
            st.ensure()
            st._git("config", key, value)

    def check(self, step: str) -> Snapshot:
        old = self.old.snapshot(label=step)
        new = self.new.snapshot(label=step)
        old_tree = self.old._git("ls-tree", "-r", "-z", old.tree).stdout.split(b"\0")
        new_tree = self.new._git("ls-tree", "-r", "-z", new.tree).stdout.split(b"\0")
        assert new_tree == old_tree, f"{step}: trees differ"
        assert new.tree == old.tree
        assert new.files == old.files
        assert new.meta["skipped_large"] == old.meta["skipped_large"]
        assert new.meta["nested_repos"] == old.meta["nested_repos"]
        assert self.new._private_modes() == self.old._private_modes(), f"{step}: modes differ"
        return new


def _odd_names(root: Path) -> list[str]:
    names = ["new file with spaces.txt", "ünïcödé/naïve.py", "-dash.txt", "#hash.txt",
             "per%cent.txt", "emoji-\U0001f600.txt"]
    if not WINDOWS:
        names += ['"quoted.txt', "tab\tname.txt", "line\nbreak.txt", "star*glob?[x].txt",
                  ":(glob)magic.txt", "back\\slash.txt", "trailing space "]
    for name in names:
        _write(root / name, f"{name}\n")
    return names


def _build_project(root: Path, *, git: bool, symlinks: bool, unreadable: bool) -> None:
    _write(root / "src" / "app.py", "print('v1')\n")
    _write(root / "src" / "lib" / "util.py", "def f(): pass\n")
    _write(root / "bin" / "run.sh", "#!/bin/sh\necho run\n", mode=0o755)
    _write(root / "docs" / "readme.md", "# readme\n")
    _write(root / "old_name.txt", "rename me\n")
    _write(root / "to_delete.txt", "delete me\n")
    _write(root / "swap_file", "a file that becomes a directory\n")
    _write(root / "swap_dir" / "inner.txt", "a directory that becomes a file\n")
    _write(root / "becomes_repo", "a file replaced by a nested repository\n")
    _write(root / "becomes_uncommitted_repo", "a file replaced by a repository, no commit\n")
    _write(root / "becomes_mixed", "a file replaced by a dir with a repo and a file\n")
    _write(root / "becomes_empty", "a file replaced by an empty directory\n")
    _write(root / "becomes_ignored", "a file replaced by a dir of ignored files\n")
    _write(root / "real" / "a.txt", "a directory later replaced by a symlink\n")
    _write(root / "link_target.txt", "target\n")
    _write(root / ".gitignore", ".env\n.env.*\n*.db\nsecret.env\nignored_dir/\nlogs/\n*.tmp\n")
    if symlinks:
        os.symlink("link_target.txt", root / "sym_to_file")
    if git:
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "init")
    # Uncommitted, untracked and ignored state on top of the commit.
    _write(root / "src" / "app.py", "print('v2, not committed')\n")
    _odd_names(root)
    _write(root / "big.bin", b"\0" * (LIMIT + 1))
    _write(root / "exactly_limit.bin", b"\1" * LIMIT)
    _write(root / ".env", "TOKEN=1\n", mode=0o600)
    _write(root / ".env.big", "X" * (LIMIT + 10))
    _write(root / "config" / "secret.env", "S=1\n", mode=0o640)
    _sqlite(root / "data" / "app.db", 3)
    _write(root / "data" / "not_sqlite.db", "plain text pretending\n")
    _write(root / "data" / "schema.sql", "-- keeps data/ from collapsing as ignored\n")
    _write(root / "node_modules" / "pkg" / "index.js", "module.exports = 1\n")
    _write(root / "node_modules" / "pkg" / ".env", "IGNORED=1\n")
    _write(root / "__pycache__" / "x.cpython-312.pyc", b"\0\1")
    _write(root / "ignored_dir" / "keep.txt", "ignored, not precious\n")
    _write(root / "ignored_dir" / "inner.env", "PRECIOUS=1\n")
    _write(root / "logs" / "a.log", "ignored\n")
    _write(root / "notes_private.txt", "private\n", mode=0o600)
    _write(root / "exec_new.sh", "#!/bin/sh\n", mode=0o755)
    (root / "empty_dir").mkdir()
    _nested_repo(root / "vendor" / "withcommit", commit=True)
    _nested_repo(root / "vendor" / "nocommit", commit=False)
    _nested_repo(root / "fresh" / "deep" / "repo", commit=True)
    _write(root / "fresh" / "sibling.txt", "next to a nested repo\n")
    _write(root / "worktree_like" / ".git", "gitdir: /nonexistent/path\n")
    _write(root / "worktree_like" / "file.txt", "pointer-file repo\n")
    if symlinks:
        os.symlink("src", root / "sym_dir")
        os.symlink("nowhere", root / "dangling")
        os.symlink("../link_target.txt", root / "docs" / "rel_link")
        os.symlink("big.bin", root / "link_to_big")
    if unreadable:
        _write(root / "unreadable.txt", "no access\n", mode=0)


def _mutate_1(root: Path, *, symlinks: bool, unreadable: bool) -> None:
    (root / "old_name.txt").rename(root / "new_name.txt")
    (root / "to_delete.txt").unlink()
    (root / "swap_file").unlink()
    _write(root / "swap_file" / "child.txt", "now a directory\n")
    shutil.rmtree(root / "swap_dir")
    _write(root / "swap_dir", "now a file\n")
    (root / "becomes_repo").unlink()
    _nested_repo(root / "becomes_repo", commit=True)
    (root / "becomes_uncommitted_repo").unlink()
    _nested_repo(root / "becomes_uncommitted_repo", commit=False)
    (root / "becomes_mixed").unlink()
    _nested_repo(root / "becomes_mixed" / "sub", commit=True)
    _write(root / "becomes_mixed" / "plain.txt", "plain\n")
    (root / "becomes_empty").unlink()
    (root / "becomes_empty").mkdir()
    (root / "becomes_ignored").unlink()
    _write(root / "becomes_ignored" / "x.tmp", "ignored\n")
    if not WINDOWS:
        (root / "docs" / "readme.md").chmod(0o755)
        (root / "bin" / "run.sh").chmod(0o644)
    _write(root / "src" / "app.py", "x" * (LIMIT + 100))  # tracked file grows past the limit
    _write(root / "big.bin", b"\2" * 10)  # shrinks under the limit
    _write(root / ".env", "TOKEN=2\n")
    _sqlite(root / "data" / "app.db", 5)
    _write(root / ".env.big", "small now\n")
    _nested_repo(root / "vendor" / "nocommit", commit=True)
    (root / "empty_dir" / "now_full.txt").write_text("filled\n", encoding="utf-8")
    if not WINDOWS:
        (root / "line\nbreak.txt").unlink()
        _write(root / "notes_private.txt", "still private\n")
    if symlinks:
        (root / "sym_to_file").unlink()
        _write(root / "sym_to_file", "was a symlink\n")
        (root / "src" / "lib" / "util.py").unlink()
        os.symlink("../app.py", root / "src" / "lib" / "util.py")
        (root / "dangling").unlink()
        os.symlink("link_target.txt", root / "dangling")
    if unreadable:
        (root / "unreadable.txt").chmod(0o644)
        (root / "docs" / "readme.md").chmod(0)  # tracked, changed mode, unreadable


def _mutate_2(root: Path, *, symlinks: bool, unreadable: bool) -> None:
    _write(root / "vendor" / "withcommit" / "more.txt", "more\n")
    _git(root / "vendor" / "withcommit", "add", "-A")
    _git(root / "vendor" / "withcommit", "commit", "-qm", "more")
    shutil.rmtree(root / "src")
    _write(root / ".env", "TOKEN=3\n", mode=0o600)
    (root / "new file with spaces.txt").unlink()
    _write(root / "ünïcödé" / "naïve.py", "changed\n")
    if symlinks:
        (root / "real").rename(root / "other_real")
        os.symlink("other_real", root / "real")
    if unreadable:
        (root / "docs" / "readme.md").chmod(0o644)


@pytest.fixture(params=["prewrite-off", "prewrite-on"])
def prewrite(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "prewrite-on":
        # Exercise the parallel blob writers even on this small fixture.
        monkeypatch.setattr(store_mod, "_PREWRITE_MIN_FILES", 1)
        monkeypatch.setattr(store_mod, "_PREWRITE_JOBS", 2)
        monkeypatch.setattr(os, "cpu_count", lambda: 2)
    return str(request.param)


@pytest.mark.parametrize("git", [True, False], ids=["git-repo", "plain-dir"])
def test_staging_matches_1_7_0_for_every_case(tmp_path: Path, git: bool, prewrite: str,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _spy_fast_path(monkeypatch)
    root = tmp_path / "proj"
    root.mkdir()
    symlinks = _can_symlink(tmp_path)
    unreadable = _can_make_unreadable(tmp_path)
    _build_project(root, git=git, symlinks=symlinks, unreadable=unreadable)
    pair = Pair(root, tmp_path / "homes")
    try:
        first = pair.check("first")
        assert "big.bin" in first.meta["skipped_large"]
        assert ".env.big" in first.meta["skipped_large"]
        listed = pair.new._git("ls-tree", "-r", "--name-only", "-z", first.tree).stdout
        names = {raw.decode("utf-8", "surrogateescape") for raw in listed.split(b"\0") if raw}
        assert {".env", "config/secret.env", "data/app.db", "data/not_sqlite.db",
                "exactly_limit.bin", "fresh/sibling.txt"} - names == set()
        assert not any(n.startswith(("node_modules/", "vendor/", "logs/")) for n in names)
        assert "unreadable.txt" not in names
        pair.check("unchanged")
        _mutate_1(root, symlinks=symlinks, unreadable=unreadable)
        second = pair.check("second")
        assert "src/app.py" in second.meta["skipped_large"]
        _mutate_2(root, symlinks=symlinks, unreadable=unreadable)
        pair.check("third")
        # The fast path really ran (the comparison is not git add against itself).
        assert calls == [True] * 4
    finally:
        for path in (root / "unreadable.txt", root / "docs" / "readme.md"):
            with contextlib.suppress(OSError):
                path.chmod(0o644)


def test_restore_after_fast_first_snapshot_round_trips(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    symlinks = _can_symlink(tmp_path)
    _build_project(root, git=True, symlinks=symlinks, unreadable=False)
    st = ShadowStore(root, home=tmp_path / "home", max_file_bytes=LIMIT)
    first = st.snapshot(label="first")
    expected = st._git("ls-tree", "-r", first.tree).stdout
    for entry in list(root.iterdir()):  # the agent wipes everything but .git and deps
        if entry.name in {".git", "node_modules", "vendor", "fresh"}:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    st.snapshot(label="wiped")
    st.restore(first)
    again = st.snapshot(label="restored")
    assert st._git("ls-tree", "-r", again.tree).stdout == expected
    assert (root / ".env").read_text(encoding="utf-8") == "TOKEN=1\n"
    if not WINDOWS:
        assert stat.S_IMODE((root / ".env").stat().st_mode) == 0o600


def _spy_fast_path(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    results: list[bool] = []
    original: Callable[..., bool] = ShadowStore._stage_with_update_index

    def spy(self: ShadowStore, *args: Any) -> bool:
        ok = original(self, *args)
        results.append(ok)
        return ok

    monkeypatch.setattr(ShadowStore, "_stage_with_update_index", spy)
    return results


def test_case_aliases_fall_back_to_git_add(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """With core.ignorecase, git add folds the case of new paths: never bypass it."""
    root = tmp_path / "proj"
    _write(root / "Dir" / "a.txt", "a\n")
    _write(root / "plain.txt", "p\n")
    pair = Pair(root, tmp_path / "homes")
    pair.set_config("core.ignorecase", "true")
    calls = _spy_fast_path(monkeypatch)
    pair.check("first")
    assert calls == [True]  # empty index, no aliases: fast path
    if (root / "dir").exists():
        pytest.skip("case-insensitive filesystem")
    _write(root / "dir" / "b.txt", "b\n")  # "dir" vs "Dir"
    _write(root / "plain.txt", "changed\n")
    pair.check("alias")
    assert calls[-1] is False
    _write(root / "NEW" / "x.txt", "x\n")
    _write(root / "new" / "y.txt", "y\n")  # aliases inside the batch itself
    pair.check("alias-in-batch")
    assert calls[-1] is False


def test_directory_replaced_by_symlink_is_snapshotted(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """1.7.0 staged nothing at all here: `git add` refuses a pathspec behind a
    symlink. Now the entries behind it are dropped, as `git add -A` does."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks unavailable")
    root = tmp_path / "proj"
    _write(root / "real" / "a.txt", "a\n")
    _write(root / "real" / "b.txt", "b\n")
    pair = Pair(root, tmp_path / "homes")
    calls = _spy_fast_path(monkeypatch)
    pair.check("first")
    (root / "real").rename(root / "moved")
    os.symlink("moved", root / "real")
    _write(root / "moved" / "a.txt", "changed behind the symlink\n")
    _write(root / "other.txt", "unrelated new file\n")
    snap = pair.new.snapshot(label="behind-symlink")
    tree = pair.new._git("ls-tree", "-r", snap.tree).stdout.decode()
    entries = {line.split("\t")[1]: line.split()[0] for line in tree.splitlines()}
    assert entries == {"moved/a.txt": "100644", "moved/b.txt": "100644", "real": "120000",
                       "other.txt": "100644"}
    reference = tmp_path / "reference-index"  # plain `git add -A` agrees
    env = {"GIT_INDEX_FILE": str(reference)}
    pair.new._git("add", "-A", env_extra=env)
    assert pair.new._git("write-tree", env_extra=env).stdout.decode().strip() == snap.tree
    assert calls == [True, True]


def test_private_modes_are_recorded_without_pathlib(tmp_path: Path) -> None:
    if WINDOWS:
        pytest.skip("POSIX permission bits")
    root = tmp_path / "proj"
    _write(root / "a" / "secret.txt", "s\n", mode=0o600)
    _write(root / "a" / "public.txt", "p\n", mode=0o644)
    _write(root / ".env", "E=1\n", mode=0o400)
    _write(root / ".gitignore", ".env\n")
    st = ShadowStore(root, home=tmp_path / "home")
    st.snapshot(label="first")
    assert st._private_modes() == {"a/secret.txt": 0o600, ".env": 0o400}


def test_first_snapshot_of_a_big_repo_never_uses_pathspecs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: one `git add` pathspec per file is superlinear (5 s at 20k files)."""
    root = tmp_path / "proj"
    for d in range(25):
        for f in range(100):
            _write(root / f"d{d:02d}" / f"f{f:03d}.txt", f"{d} {f}\n")
    _write(root / ".gitignore", ".env\n")
    _write(root / ".env", "E=1\n")
    commands: list[tuple[str, ...]] = []
    original = ShadowStore._git

    def spy(self: ShadowStore, *args: str, **kwargs: Any) -> Any:
        commands.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ShadowStore, "_git", spy)
    prewritten: list[int] = []
    original_prewrite = ShadowStore._prewrite_blobs

    def spy_prewrite(self: ShadowStore, regular: list[str]) -> None:
        prewritten.append(len(regular))
        original_prewrite(self, regular)

    monkeypatch.setattr(ShadowStore, "_prewrite_blobs", spy_prewrite)
    snap = ShadowStore(root, home=tmp_path / "home").snapshot(label="first")
    assert snap.files == 2502
    assert not [c for c in commands if c[0] == "add"]
    assert [c[:2] for c in commands if c[0] == "update-index"] == [("update-index", "--add")]
    assert prewritten == [2501]  # every regular file (+ .gitignore), in parallel
