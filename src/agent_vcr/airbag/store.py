"""Shadow-git snapshot store: the airbag's memory.

Each project gets a private git directory **outside** the project (under the
user's state dir), so:

* ``rm -rf`` of the whole project can still be undone;
* the user's own ``.git`` — HEAD, branches, index, stash, hooks — is never
  read or written;
* untracked files are protected too, and small "precious" files that are
  usually git-ignored (``.env``, SQLite databases) are force-included.

Snapshots are parentless commits referenced by ``refs/airbag/<seq>`` so
garbage collection is just "delete old refs, prune". Everything here uses
the standard library and the ``git`` binary only, because it runs inside a
hook on every agent tool call and must start fast.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_vcr import _platform

#: Never text mode on Windows (no CRLF translation of the JSONL log).
_O_BINARY = getattr(os, "O_BINARY", 0)

#: Directories never snapshotted (build output, caches, dependency trees).
DEFAULT_EXCLUDES = (
    "node_modules/", ".venv/", "venv/", "env/", "__pycache__/", "*.pyc",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".tox/", ".nox/",
    ".next/", ".nuxt/", ".svelte-kit/", ".turbo/", ".parcel-cache/", ".cache/",
    ".gradle/", "target/", "dist/", "build/", "coverage/", "htmlcov/",
    ".terraform/", ".DS_Store", "*.swp",
)

#: Usually git-ignored, but losing them hurts: force-included when small.
DEFAULT_PRECIOUS = (
    ".env", ".env.*", "*.env", "*.db", "*.sqlite", "*.sqlite3", "*.db3",
)

DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024
#: Refuse to protect absurdly large trees by accident (e.g. a home directory).
DEFAULT_MAX_FILES = 200_000

#: For git commands that write many objects: glibc otherwise returns zlib's
#: buffers to the OS after every blob and page-faults them back in for the
#: next one (~20 faults per file, half of ``update-index``'s time on a fresh
#: store). Other C libraries ignore these variables.
_MALLOC_ENV = {"MALLOC_TRIM_THRESHOLD_": "268435456", "MALLOC_TOP_PAD_": "16777216"}

#: Parallel blob writers for big batches (see ``ShadowStore._prewrite_blobs``). Off on
#: Windows: measured on a GitHub runner, a 20k-file first snapshot took 41 s with them and
#: 18 s without (parallel loose-object writes there are slower than one update-index).
_PREWRITE_ENABLED = not _platform.WINDOWS
_PREWRITE_JOBS = 4
_PREWRITE_MIN_FILES = 2000

#: At most this many ignored directories are searched for precious files per snapshot,
#: and for at most this long; the rest are searched by the next snapshots (rotating).
_PRECIOUS_DIR_LIMIT = 200
_PRECIOUS_TIME_BUDGET = 0.2
#: Ignored third-party trees that are never searched for precious files: their
#: ``.env``/``*.db`` files belong to someone else's package, and they can be huge.
_PRECIOUS_SKIP_DIRS = frozenset({
    "vendor", "third_party", "third-party", "Pods", "Carthage", "DerivedData",
    "bower_components", "jspm_packages", ".bundle", ".gradle", ".m2", ".ivy2", ".yarn",
    ".pnpm-store", ".npm", ".cargo", ".stack-work", "elm-stuff", ".dart_tool", ".pub-cache",
    "site-packages",
})

#: A SQLite file whose stat (and its sidecars') is unchanged since its last backup is not
#: backed up again, unless it was modified this close to that check: timestamps are
#: coarse (a jiffy on Linux, 2 s on FAT), so a later write could keep the same mtime.
_SQLITE_RACY_NS = 2_000_000_000
_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".db3")
#: Sidecars whose stat is part of a database's identity. Not ``-shm``: it only indexes
#: the WAL (no content of its own) and every reader writes to it, our backup included.
_SQLITE_SIDECARS = ("-wal", "-journal")

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class AirbagError(RuntimeError):
    """Raised for unrecoverable store problems (missing git, unsafe root...)."""


@dataclass
class Snapshot:
    """One protected state of the project."""

    seq: int
    commit: str
    tree: str
    time: float
    label: str
    files: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return f"#{self.seq}"


@dataclass
class DatabaseDump:
    """A database dump file (in the store) waiting to be attached to a snapshot."""

    path: Path
    name: str
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestoreResult:
    """What a restore changed on disk."""

    target: Snapshot
    before: Snapshot
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: Present on disk but not protected (too large, ignored, nested repo): left untouched.
    kept: list[str] = field(default_factory=list)
    #: Databases restored from a dump (see :mod:`agent_vcr.airbag.databases`).
    databases: list[str] = field(default_factory=list)
    #: Caveats about what was restored (e.g. a database restore that was not exact).
    notes: list[str] = field(default_factory=list)
    #: The subset of ``written`` that already existed and was changed back (not re-created).
    modified: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.written) + len(self.deleted)


def state_home() -> Path:
    """Where shadow stores live (outside every project)."""
    override = os.environ.get("AGENT_VCR_AIRBAG_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "agent-vcr" / "airbag"


def known_roots(home: str | Path | None = None) -> list[Path]:
    """Project roots that already have an airbag store (from each ``root.txt``)."""
    base = Path(home).expanduser() if home is not None else state_home()
    roots: list[Path] = []
    with contextlib.suppress(OSError):
        for marker in base.glob("*/root.txt"):
            with contextlib.suppress(OSError):
                text = marker.read_text(encoding="utf-8").strip()
                if text:
                    roots.append(Path(text))
    return roots


def _nearest_known_root(path: Path) -> Path | None:
    candidates = [r for r in known_roots() if r == path or r in path.parents]
    return max(candidates, key=lambda r: len(r.parts)) if candidates else None


def find_project_root(start: str | Path) -> Path:
    """The project a path belongs to.

    1. the git top-level, if ``start`` is inside a working git repository;
    2. otherwise the nearest already-protected root (so ``vcr undo`` still finds
       the snapshots after the agent deleted the project or its ``.git``);
    3. otherwise ``start`` itself.

    Uses the *user's* git read-only (``rev-parse``); nothing is written.
    """
    path = Path(os.path.abspath(Path(start).expanduser()))
    if not path.exists():
        # The agent may have deleted the project: never fall back to a parent
        # directory, which would be a different (empty) store.
        return _nearest_known_root(path) or path
    path = path.resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path), capture_output=True, text=True, timeout=10,
            env=_clean_env(), stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).resolve()
    return _nearest_known_root(path) or path


def _clean_env() -> dict[str, str]:
    """Environment without GIT_* variables that could redirect git elsewhere."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _link_or_copy(src: str, dst: str) -> object:
    """``copytree`` copy function for a ``.git`` directory: hardlink when it is safe.

    Git replaces files via rename and never edits objects in place, so on POSIX
    a hardlink is a faithful point-in-time copy. Not on Windows: git objects are
    read-only there and deleting one link (``gc`` of an old backup, an agent's
    ``rm -rf``) means clearing the read-only attribute, which every link shares,
    so the user's own ``.git`` would be modified. Windows gets real copies.
    """
    if not _platform.WINDOWS:
        try:
            os.link(src, dst)
            return dst
        except OSError:
            pass
    shutil.copy2(src, dst)
    return dst


def _is_unsafe_root(root: Path) -> str | None:
    home = Path.home().resolve()
    if root == Path(root.anchor):
        return "the filesystem root"
    if root == home:
        return "your home directory"
    if root in {home / "Desktop", home / "Documents", home / "Downloads"}:
        return f"{root.name} (too broad to protect)"
    return None


@dataclass
class _Changes:
    """What a snapshot has to stage (see ``ShadowStore._stage``)."""

    #: Modified, deleted and new non-ignored paths (minus files over the size limit).
    to_add: list[str] = field(default_factory=list)
    #: Small ignored files that are force-included (``.env``, SQLite...).
    precious: list[str] = field(default_factory=list)
    #: Regular files in ``to_add`` that can be read.
    regular: list[str] = field(default_factory=list)
    #: Regular files ``git add --ignore-errors`` would skip: we cannot read them.
    unreadable: set[str] = field(default_factory=set)
    #: Indexed paths that are a directory now.
    dirs: set[str] = field(default_factory=set)


class ShadowStore:
    """Snapshot/restore for one project directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        home: str | Path | None = None,
        excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
        precious: tuple[str, ...] = DEFAULT_PRECIOUS,
        max_file_bytes: int | None = None,
        max_files: int | None = None,
        git_timeout: float = 60.0,
    ) -> None:
        if max_file_bytes is None:
            max_file_bytes = int(float(os.environ.get(
                "AGENT_VCR_AIRBAG_MAX_FILE_MB", DEFAULT_MAX_FILE_BYTES / 1048576)) * 1048576)
        if max_files is None:
            max_files = int(os.environ.get("AGENT_VCR_AIRBAG_MAX_FILES", DEFAULT_MAX_FILES))
        self.root = Path(root).expanduser().resolve()
        reason = _is_unsafe_root(self.root)
        if reason:
            raise AirbagError(f"refusing to snapshot {reason}: {self.root}")
        base = Path(home).expanduser() if home is not None else state_home()
        digest = hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        name = _SAFE_NAME.sub("_", self.root.name or "root")[:40]
        self.dir = base / f"{name}-{digest}"
        self.git_dir = self.dir / "git"
        self.index_file = self.dir / "index"
        self.log_file = self.dir / "log.jsonl"
        self.state_file = self.dir / "state.json"
        self.excludes = excludes
        self.precious = precious
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.git_timeout = git_timeout

    # ── setup / locking ──────────────────────────────────────────────────

    def ensure(self) -> None:
        """Create the shadow git dir on first use (idempotent, safe under races).

        Several hook processes can start at once (agents run tool calls in
        parallel), so the repository is built in a temporary directory under
        the store lock and moved into place atomically.
        """
        if (self.git_dir / "HEAD").exists():
            self._ensure_attributes()
            return
        if shutil.which("git") is None:
            raise AirbagError("git is required for the airbag but was not found on PATH")
        self.dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.dir, 0o700)  # snapshots can contain .env files
        with self.lock():
            if (self.git_dir / "HEAD").exists():
                self._ensure_attributes()
                return
            tmp = self.dir / f"git.init-{os.getpid()}"
            _platform.rmtree(tmp, ignore_errors=True)
            for cmd in (
                ["git", "init", "-q", "--bare", str(tmp)],
                # Not really bare: GIT_WORK_TREE always points at the project.
                ["git", "--git-dir", str(tmp), "config", "core.bare", "false"],
            ):
                subprocess.run(
                    cmd, check=True, capture_output=True, timeout=self.git_timeout,
                    env=_clean_env(), stdin=subprocess.DEVNULL,
                )
            info = tmp / "info"
            info.mkdir(exist_ok=True)
            (info / "exclude").write_text(
                "# agent-vcr airbag defaults\n" + "\n".join(self.excludes) + "\n", encoding="utf-8"
            )
            (info / "attributes").write_text(
                "* -text -eol -filter -ident -working-tree-encoding\n", encoding="utf-8")
            _platform.replace(tmp, self.git_dir)
            (self.dir / "root.txt").write_text(str(self.root) + "\n", encoding="utf-8")

    def _ensure_attributes(self) -> None:
        """Store bytes exactly: no eol conversion, no LFS/clean filters, no encoding.

        ``$GIT_DIR/info/attributes`` overrides in-tree ``.gitattributes``, so a
        project's ``* text=auto`` cannot rewrite line endings on restore.
        """
        attrs = self.git_dir / "info" / "attributes"
        if not attrs.exists():
            attrs.parent.mkdir(parents=True, exist_ok=True)
            attrs.write_text("* -text -eol -filter -ident -working-tree-encoding\n",
                             encoding="utf-8")

    @contextlib.contextmanager
    def lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Serialize store operations across concurrent hook processes."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "lock", "a+b") as handle:
            deadline = time.monotonic() + timeout
            while not _platform.try_lock(handle):
                if time.monotonic() > deadline:
                    raise AirbagError("timed out waiting for the airbag lock")
                time.sleep(0.02)
            try:
                yield
            finally:
                _platform.unlock(handle)

    def _git(
        self, *args: str, input_bytes: bytes | None = None, check: bool = True,
        env_extra: dict[str, str] | None = None, cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        env = _clean_env()
        env.update({
            "GIT_DIR": str(self.git_dir),
            "GIT_WORK_TREE": str(self.root),
            "GIT_INDEX_FILE": str(self.index_file),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "agent-vcr airbag",
            "GIT_AUTHOR_EMAIL": "airbag@agent-vcr.local",
            "GIT_COMMITTER_NAME": "agent-vcr airbag",
            "GIT_COMMITTER_EMAIL": "airbag@agent-vcr.local",
        })
        env.update(env_extra or {})
        cmd = [
            "git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
            "-c", "core.quotepath=false", "-c", "core.fsmonitor=false",
            "-c", "gc.auto=0", "-c", "advice.addIgnoredFile=false", *args,
        ]
        if cwd is None:
            cwd = self.root if self.root.exists() else self.dir
        result = subprocess.run(
            cmd, cwd=str(cwd),
            input=input_bytes, capture_output=True, timeout=self.git_timeout,
            env=env, stdin=None if input_bytes is not None else subprocess.DEVNULL,
        )
        if check and result.returncode != 0:
            raise AirbagError(
                f"git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}"
            )
        return result

    # ── snapshot ─────────────────────────────────────────────────────────

    def snapshot(self, label: str = "snapshot", meta: dict[str, Any] | None = None,
                 *, backup_git_dir: bool = False,
                 databases: list[DatabaseDump] | None = None) -> Snapshot:
        """Record the current state of the project. Cheap when nothing changed.

        ``backup_git_dir`` also preserves the project's own ``.git`` (as
        hardlinks — near-instant, no extra space) for commands that would
        delete it, e.g. ``rm -rf`` of the project.

        ``databases`` attaches database dumps (already written into the store
        via :meth:`new_dump_path`). Such a snapshot always gets its own number,
        even when no file changed, and its metadata lists the dumps.
        """
        self.ensure()
        with self.lock():
            snap = self._snapshot_locked(label, meta or {}, databases=databases)
            if backup_git_dir:
                self._backup_git_dir(snap.seq)
            return snap

    # ── database dumps attached to snapshots ─────────────────────────────

    @property
    def db_dir(self) -> Path:
        return self.dir / "db"

    def new_dump_path(self, name: str) -> Path:
        """A fresh private (0600) file in the store for a database dump in progress."""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.dir, 0o700)
            os.chmod(self.db_dir, 0o700)
        safe = _SAFE_NAME.sub("_", name)[:40] or "db"
        fd, path = tempfile.mkstemp(dir=str(self.db_dir), prefix=f".tmp-{safe}-", suffix=".dump")
        os.close(fd)
        os.chmod(path, 0o600)
        return Path(path)

    def dump_file(self, entry: dict[str, Any]) -> Path | None:
        """The dump file a snapshot's ``databases`` entry points to (inside the store only)."""
        rel = str(entry.get("dump") or "")
        if not rel:
            return None
        path = (self.dir / rel).resolve()
        if self.db_dir.resolve() not in path.parents or not path.is_file():
            return None
        return path

    def _attach_dumps(self, seq: int, dumps: list[DatabaseDump]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for dump in dumps:
            safe = _SAFE_NAME.sub("_", dump.name)[:40] or "db"
            dest = self.db_dir / f"{seq:08d}-{safe}.dump"
            n = 1
            while dest.exists():
                n += 1
                dest = self.db_dir / f"{seq:08d}-{safe}-{n}.dump"
            _platform.replace(dump.path, dest)
            entries.append({**dump.info, "name": dump.name,
                            "dump": dest.relative_to(self.dir).as_posix(),
                            "bytes": dest.stat().st_size})
        return entries

    # ── the project's own .git (only for commands that delete it) ────────

    @property
    def gitdirs(self) -> Path:
        return self.dir / "gitdirs"

    def _backup_git_dir(self, seq: int) -> Path | None:
        src = self.root / ".git"
        if src.is_symlink():
            return None
        if src.is_file():
            # Linked worktree / submodule: `.git` is a small pointer file.
            dst_file = self.gitdirs / f"{seq:08d}.gitfile"
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_file)
            return dst_file
        if not src.is_dir():
            return None
        dst = self.gitdirs / f"{seq:08d}"
        if dst.exists():
            return dst
        tmp = self.gitdirs / f".{seq:08d}.tmp"
        _platform.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)

        shutil.copytree(src, tmp, symlinks=True, copy_function=_link_or_copy)
        _platform.replace(tmp, dst)
        return dst

    def latest_git_dir_backup(self) -> Path | None:
        if not self.gitdirs.is_dir():
            return None
        backups = sorted(p for p in self.gitdirs.iterdir()
                         if not p.name.startswith(".") and (p.is_dir() or p.suffix == ".gitfile"))
        return backups[-1] if backups else None

    def _repo_works(self, git_dir: Path | None = None) -> bool:
        """Does HEAD resolve to a commit (in the project, or in ``git_dir``)?"""
        cmd = ["git", "rev-parse", "--verify", "-q", "HEAD^{commit}"]
        if git_dir is not None:
            cmd = ["git", "--git-dir", str(git_dir), *cmd[1:]]
        try:
            proc = subprocess.run(cmd, cwd=str(self.root), capture_output=True,
                                  timeout=self.git_timeout, env=_clean_env(),
                                  stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def git_dir_needs_repair(self) -> bool:
        """True when a ``.git`` backup exists and the project's ``.git`` is missing,
        broken, or no longer contains the backed-up history (``rm -rf .git && git init``)."""
        backup = self.latest_git_dir_backup()
        if backup is None:
            return False
        target = self.root / ".git"
        if not (target.exists() or target.is_symlink()):
            return True
        if backup.suffix == ".gitfile":
            return False
        if not self._repo_works(backup):
            return False
        if not self._repo_works():
            return True
        head = self._rev_parse(backup)
        return head is not None and not self._has_commit(head)

    def _rev_parse(self, git_dir: Path) -> str | None:
        try:
            proc = subprocess.run(["git", "--git-dir", str(git_dir), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=self.git_timeout,
                                  env=_clean_env(), stdin=subprocess.DEVNULL,
                                  encoding="utf-8", errors="replace")
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    def _has_commit(self, sha: str) -> bool:
        try:
            proc = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                                  cwd=str(self.root), capture_output=True,
                                  timeout=self.git_timeout, env=_clean_env(),
                                  stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def _restore_git_dir(self) -> str | None:
        """Bring back the project's ``.git`` from the airbag's backup when needed.

        Only when ``.git`` is missing, or present but broken (HEAD no longer
        resolves — e.g. ``rm -rf .git/objects`` or ``rm -rf .git && git init``)
        while the backup still works. A replaced ``.git`` is moved aside into
        the store, never deleted. A working repository is never touched.
        """
        backup = self.latest_git_dir_backup()
        if backup is None:
            return None
        target = self.root / ".git"
        if backup.suffix == ".gitfile":
            if target.exists() or target.is_symlink():
                return None
            shutil.copy2(backup, target)
            return ".git (worktree link)"
        note = ".git/"
        if target.exists() or target.is_symlink():
            if not self.git_dir_needs_repair():
                return None
            aside = self.dir / "replaced-gitdirs" / time.strftime("%Y%m%d-%H%M%S")
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(aside))
            note = f".git/ (repaired; the damaged one is kept at {aside})"

        shutil.copytree(backup, target, symlinks=True, copy_function=_link_or_copy)
        return note

    def _snapshot_locked(self, label: str, meta: dict[str, Any], *,
                         databases: list[DatabaseDump] | None = None) -> Snapshot:
        if not self.root.exists():
            self.root.mkdir(parents=True)
        skipped: list[str] = []
        changed = self._changed_paths()
        if len(changed) > self.max_files:
            raise AirbagError(
                f"{len(changed)} files would be snapshotted (limit {self.max_files}); "
                "add excludes or raise AGENT_VCR_AIRBAG_MAX_FILES"
            )
        changes = _Changes()
        to_add = changes.to_add
        private: dict[str, int] = {}
        # Untracked nested repositories are listed as "dir/": git cannot add their
        # contents (only a gitlink, and not even that before their first commit).
        untracked_nested = [rel.rstrip("/") for rel in changed if rel.endswith("/")]
        # One lstat per path on plain strings: pathlib costs more than the syscall
        # on the first snapshot of a big repository, where every file is "changed".
        prefix = os.path.join(str(self.root), "")
        uid = os.getuid() if hasattr(os, "getuid") else -1
        sqlite_cache = self._sqlite_cache()
        sqlite_reused: set[str] = set()
        for rel in changed:
            if rel.endswith("/"):
                continue
            try:
                st = os.lstat(prefix + rel)
            except OSError:
                to_add.append(rel)  # deleted: staged as a removal
                continue
            if stat.S_ISREG(st.st_mode):
                if st.st_size > self.max_file_bytes:
                    skipped.append(rel)
                    continue
                if not st.st_mode & 0o077:
                    private[rel] = stat.S_IMODE(st.st_mode)
                if rel in sqlite_cache and _sqlite_unchanged(prefix + rel, sqlite_cache[rel]):
                    # Its consistent backup from an earlier snapshot is still staged (the
                    # entry carries no stat data, so ls-files always lists it as modified).
                    # _stage_sqlite_consistently verifies that blob.
                    sqlite_reused.add(rel)
                    continue
                if _maybe_unreadable(st, uid) and not os.access(prefix + rel, os.R_OK):
                    changes.unreadable.add(rel)
                else:
                    changes.regular.append(rel)
            elif stat.S_ISDIR(st.st_mode):
                changes.dirs.add(rel)  # an indexed file (or gitlink) is a directory now
            to_add.append(rel)
        notes: list[str] = []
        precious, too_big_precious = self._precious_ignored_files(notes)
        for rel in precious:
            with contextlib.suppress(OSError):
                st = os.lstat(prefix + rel)
                if stat.S_ISREG(st.st_mode) and _maybe_unreadable(st, uid) \
                        and not os.access(prefix + rel, os.R_OK):
                    changes.unreadable.add(rel)
        changes.precious = precious
        self._stage(changes)
        unreadable = sorted(changes.unreadable)
        if unreadable:
            notes.append(f"{len(unreadable)} file(s) could not be read and are not in this "
                         f"snapshot: {', '.join(unreadable[:5])}"
                         f"{', ...' if len(unreadable) > 5 else ''}")
        skipped.extend(too_big_precious)
        if skipped:
            # Files over the limit must not stay in the index with an old, small
            # blob: a later restore would write that stale copy over the real file.
            self._git(
                "update-index", "-z", "--force-remove", "--stdin",
                input_bytes=b"".join(p.encode("utf-8", "surrogateescape") + b"\0"
                                     for p in skipped),
            )
        nested = self._stage_sqlite_consistently(sqlite_cache, sqlite_reused)
        nested.extend(n for n in untracked_nested if n not in nested)
        private.update(_private_modes_of(prefix, precious))
        self._remember_private_modes(private)

        tree = self._git("write-tree").stdout.decode().strip()
        last = self.latest()
        if last is not None and last.tree == tree and not databases \
                and not last.meta.get("databases"):
            # Nothing changed since the previous snapshot. One carrying database dumps is
            # never shared, so a later action can't be mistaken for the one that dumped.
            return last
        files = int(self._git("ls-files", "-z").stdout.count(b"\0"))
        seq = (last.seq + 1) if last is not None else 1
        if databases:
            meta = {**meta, "databases": self._attach_dumps(seq, databases)}
        extra: dict[str, Any] = {}
        if unreadable:
            extra["skipped_unreadable"] = unreadable[:50]
        if notes:
            extra["notes"] = notes
        meta = {**extra, **meta}
        record = {"seq": seq, "label": label, "time": time.time(), "files": files,
                  "skipped_large": skipped[:50], "nested_repos": nested[:50], **meta}
        commit = self._git(
            "commit-tree", tree, input_bytes=json.dumps(record, default=str).encode()
        ).stdout.decode().strip()
        self._git("update-ref", f"refs/airbag/{seq:08d}", commit)
        snap = Snapshot(seq=seq, commit=commit, tree=tree, time=record["time"],
                        label=label, files=files,
                        meta={"skipped_large": skipped[:50], "nested_repos": nested[:50], **meta})
        self._write_state({"latest": seq})
        return snap

    def _changed_paths(self) -> list[str]:
        """Modified, deleted and new (non-ignored) paths relative to the shadow index."""
        out = self._git(
            "ls-files", "-z", "--modified", "--deleted", "--others", "--exclude-standard",
        ).stdout
        seen: dict[str, None] = {}
        for raw in out.split(b"\0"):
            if raw:
                seen[raw.decode("utf-8", "surrogateescape")] = None
        return list(seen)

    def _stage(self, changes: _Changes) -> None:
        """Bring the shadow index up to date for the changed and the precious files.

        ``changes.to_add`` are modified, deleted and new (non-ignored) paths;
        ``changes.precious`` are small ignored files that are force-included.
        The result is what ``git add -A --ignore-errors`` / ``git add -f
        --ignore-errors`` with those literal pathspecs produce, but ``git
        update-index --stdin`` gets there without pathspec matching, which is
        superlinear in the number of pathspecs (seconds for the first snapshot
        of a 20k-file repository). Whenever the fast path cannot guarantee the
        same index, or fails, the ``git add`` commands run instead.
        """
        to_add, precious = changes.to_add, changes.precious
        if not to_add and not precious:
            return
        prefix = os.path.join(str(self.root), "")
        links: dict[str, bool] = {}
        behind = {p for p in to_add if "/" in p and _behind_symlink(prefix, p, links)}
        if behind:
            # Index entries under a directory that is now a symlink. `git add`
            # refuses such a pathspec and then stages nothing at all (1.7.0 lost
            # every other change of that snapshot); `git add -A` without
            # pathspecs drops them, which is what happens here.
            self._git(
                "update-index", "-z", "--force-remove", "--stdin",
                input_bytes=b"".join(p.encode("utf-8", "surrogateescape") + b"\0"
                                     for p in sorted(behind)),
            )
            to_add = [p for p in to_add if p not in behind]
            changes.regular = [p for p in changes.regular if p not in behind]
        if self._stage_with_update_index(to_add, changes):
            return
        if to_add:
            self._git(
                "add", "-A", "--ignore-errors", "--pathspec-from-file=-", "--pathspec-file-nul",
                input_bytes=b"\0".join(_literal(p) for p in to_add), check=False,
                env_extra=_MALLOC_ENV,
            )
        if precious:
            self._git(
                "add", "-f", "--ignore-errors", "--pathspec-from-file=-", "--pathspec-file-nul",
                input_bytes=b"\0".join(_literal(p) for p in precious), check=False,
                env_extra=_MALLOC_ENV,
            )

    def _stage_with_update_index(self, to_add: list[str], changes: _Changes) -> bool:
        """``git update-index --add --remove --replace``; False = nothing done.

        Equivalent to ``git add`` for every listed path: new and modified files
        (and symlinks) are hashed with the same mode rules, missing ones are
        removed and file/directory swaps are replaced. ``git add
        --ignore-errors`` skips files it cannot read, so those are left out.
        Indexed paths that are directories now go through ``git add`` (only a
        handful of pathspecs): it turns them into gitlinks when they hold a
        repository, and adds what they contain. ``update-index`` never writes a
        partial index: on any error it exits without touching the index and
        the caller falls back to ``git add`` for everything.

        ``git add`` also folds the case of new paths into existing entries when
        ``core.ignorecase`` is set (macOS, Windows); ``update-index`` does not,
        so when any listed path could alias another one by case, this declines.
        """
        everything = to_add + changes.precious
        if self._ignorecase() and self._case_aliases(everything):
            return False
        skip = changes.unreadable | changes.dirs
        batch = sorted(p.encode("utf-8", "surrogateescape") for p in everything
                       if p not in skip)
        if batch:
            self._prewrite_blobs(changes.regular)
            proc = self._git(
                "update-index", "--add", "--remove", "--replace", "-z", "--stdin",
                input_bytes=b"".join(p + b"\0" for p in batch),
                check=False, env_extra=_MALLOC_ENV,
            )
            if proc.returncode != 0:
                return False
        dirs = [p for p in to_add if p in changes.dirs]
        if dirs:
            self._git(
                "add", "-A", "--ignore-errors", "--pathspec-from-file=-", "--pathspec-file-nul",
                input_bytes=b"\0".join(_literal(p) for p in dirs), check=False,
                env_extra=_MALLOC_ENV,
            )
        return True

    def _prewrite_blobs(self, regular: list[str]) -> None:
        """Write the blobs of many files with parallel ``git hash-object`` processes.

        Purely a speed-up for big batches (the first snapshot of a repository):
        deflating and writing thousands of loose objects is what ``update-index``
        spends its time on, and it does that on one core. ``update-index`` still
        hashes every file itself and only finds the object already present, so
        what gets staged never depends on this step; anything it misses or gets
        wrong (a file changing meanwhile) is at worst an unreferenced object.
        """
        jobs = min(_PREWRITE_JOBS, os.cpu_count() or 1)
        if not _PREWRITE_ENABLED or jobs < 2 or len(regular) < _PREWRITE_MIN_FILES:
            return
        # --stdin-paths is line-based, strips a trailing CR and C-unquotes lines
        # starting with '"'; a path it cannot open ends that worker (no harm).
        names = [rel for rel in regular
                 if "\n" not in rel and not rel.endswith("\r") and not rel.startswith('"')]
        env = _clean_env()
        env.update({"GIT_DIR": str(self.git_dir), "GIT_WORK_TREE": str(self.root),
                    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                    **_MALLOC_ENV})
        procs: list[subprocess.Popen[bytes]] = []
        tmp = Path(tempfile.mkdtemp(dir=str(self.dir), prefix="prewrite-"))
        try:
            for n in range(jobs):
                listing = tmp / f"paths-{n}"
                listing.write_bytes(b"".join(
                    rel.encode("utf-8", "surrogateescape") + b"\n" for rel in names[n::jobs]))
                with open(listing, "rb") as stdin:
                    procs.append(subprocess.Popen(
                        ["git", "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
                         "hash-object", "-w", "--stdin-paths"],
                        cwd=str(self.root), env=env, stdin=stdin,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    ))
            deadline = time.monotonic() + self.git_timeout
            for proc in procs:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except (OSError, subprocess.SubprocessError):
            pass  # update-index writes whatever is missing
        finally:
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
                    with contextlib.suppress(subprocess.SubprocessError):
                        proc.wait(timeout=5)
            _platform.rmtree(tmp, ignore_errors=True)

    def _ignorecase(self) -> bool:
        """``core.ignorecase`` of the shadow repository (set by ``git init``)."""
        try:
            text = (self.git_dir / "config").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return re.search(r"(?im)^\s*ignorecase\s*=\s*(true|yes|on|1)\s*$", text) is not None

    def _case_aliases(self, batch: list[str]) -> bool:
        """Could ``git add``'s case folding change any of these paths?

        True when a path, or one of its directories, matches a different
        spelling (ignoring case) in the shadow index or elsewhere in the batch.
        """
        out = self._git("ls-files", "-z").stdout if self.index_file.exists() else b""
        indexed = [raw.decode("utf-8", "surrogateescape") for raw in out.split(b"\0") if raw]
        seen: dict[str, str] = {}
        dirs_done: set[str] = set()
        for rel in itertools.chain(indexed, batch):
            if seen.setdefault(rel.lower(), rel) != rel:
                return True
            parent = rel.rpartition("/")[0]
            while parent and parent not in dirs_done:  # each directory spelling once
                dirs_done.add(parent)
                if seen.setdefault(parent.lower(), parent) != parent:
                    return True
                parent = parent.rpartition("/")[0]
        return False

    def _remember_private_modes(self, private: dict[str, int]) -> None:
        """Remember permission bits of private files (e.g. ``.env`` at 0600).

        Git only stores the executable bit, so a deleted-then-restored secret
        would otherwise come back group/world-readable. ``private`` maps each
        snapshotted regular file without group/other bits to its mode.
        """
        if not private:
            return
        modes = self._private_modes()
        if all(modes.get(k) == v for k, v in private.items()):
            return
        modes.update(private)
        tmp = self.dir / "modes.json.tmp"
        tmp.write_text(json.dumps(modes), encoding="utf-8")
        _platform.replace(tmp, self.dir / "modes.json")

    def _private_modes(self) -> dict[str, int]:
        try:
            data = json.loads((self.dir / "modes.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _precious_ignored_files(self, notes: list[str] | None = None,
                                ) -> tuple[list[str], list[str]]:
        """Ignored files matching the precious patterns (e.g. ``.env``): (small, too big).

        Precious files are found by listing untracked ignored files, so each is only
        looked for until it is in the shadow index; from then on ``ls-files
        --modified`` follows it. Directories ignored as a whole are searched within a
        per-snapshot budget (``_PRECIOUS_DIR_LIMIT`` directories, about
        ``_PRECIOUS_TIME_BUDGET`` seconds), resuming where the previous snapshot
        stopped, so every directory is eventually searched and a snapshot that did not
        get to all of them says so in ``notes``.
        """
        if not self.precious:
            return [], []
        out = self._git(
            "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory",
        ).stdout
        entries = [raw for raw in out.split(b"\0") if raw]
        # A whole ignored directory comes back collapsed ("data/"). Look inside it for
        # precious files only (data/app.db, config/.env), never inside dependency or
        # build trees (node_modules/, .venv/...), with one pathspec-limited listing.
        collapsed = [
            raw.decode("utf-8", "surrogateescape") for raw in entries if raw.endswith(b"/")
        ]
        look_inside = [
            d for d in collapsed
            if not any(_matches_exclude(d + "x", pat) for pat in self.excludes)
            and _PRECIOUS_SKIP_DIRS.isdisjoint(d.split("/"))
        ]
        total = len(look_inside)
        cursor = int(self.read_state().get("precious_cursor") or 0) % total if total else 0
        order = look_inside[cursor:] + look_inside[:cursor]
        searched = 0
        inside: set[bytes] = set()
        # ls-files has no --pathspec-from-file; batch to stay far below Windows'
        # 32k-character command-line limit.
        per_call = max(1, 100 // len(self.precious))
        deadline = time.monotonic() + _PRECIOUS_TIME_BUDGET
        while searched < min(total, _PRECIOUS_DIR_LIMIT):
            if searched and time.monotonic() > deadline:
                break
            chunk = order[searched:min(searched + per_call, _PRECIOUS_DIR_LIMIT)]
            specs = [f":(glob){_glob_escape(d)}**/{pat}" for d in chunk for pat in self.precious]
            inner = self._git(
                "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
                "--", *specs, check=False,
            )
            if inner.returncode == 0:
                inside.update(raw for raw in inner.stdout.split(b"\0") if raw)
            searched += len(chunk)
        next_cursor = (cursor + searched) % total if searched < total else 0
        if next_cursor != cursor:
            self._write_state({"precious_cursor": next_cursor})
        if searched < total and notes is not None:
            notes.append(f".env/SQLite files were looked for in {searched} of {total} ignored "
                         "folders; the next snapshots search the others")
        entries.extend(sorted(inside))
        result: list[str] = []
        too_big: list[str] = []
        seen: set[bytes] = set()
        for raw in entries:
            if raw.endswith(b"/") or raw in seen:
                continue  # collapsed ignored directory (handled above)
            seen.add(raw)
            rel = raw.decode("utf-8", "surrogateescape")
            name = rel.rsplit("/", 1)[-1]
            if not any(fnmatch.fnmatch(name, pat) for pat in self.precious):
                continue
            if any(_matches_exclude(rel, pat) for pat in self.excludes):
                continue
            if raw in inside and not _PRECIOUS_SKIP_DIRS.isdisjoint(rel.split("/")[:-1]):
                continue  # found in an ignored folder, but within a third-party tree
            try:
                if (self.root / rel).stat().st_size <= self.max_file_bytes:
                    result.append(rel)
                else:
                    too_big.append(rel)
            except OSError:
                continue
        return result, too_big

    def _sqlite_cache(self) -> dict[str, dict[str, Any]]:
        """``{path: {"key", "blob", "checked_ns"}}`` of the last SQLite backups."""
        try:
            data = json.loads((self.dir / "sqlite-cache.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)} \
            if isinstance(data, dict) else {}

    def _write_sqlite_cache(self, cache: dict[str, dict[str, Any]]) -> None:
        dest = self.dir / "sqlite-cache.json"
        if not cache:
            with contextlib.suppress(OSError):
                dest.unlink()
            return
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="sqlite-cache.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
            _platform.replace(tmp, dest)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _stage_sqlite_consistently(self, cache: dict[str, dict[str, Any]] | None = None,
                                   reused: set[str] | frozenset[str] = frozenset(),
                                   ) -> list[str]:
        """Replace staged SQLite files with a consistent online-backup copy.

        Copying a live SQLite file byte-for-byte can capture a torn state
        (especially in WAL mode). The backup API produces a consistent image.
        A database whose stat and sidecars are unchanged since its last backup
        (see ``_sqlite_unchanged``) keeps that backup, which is still staged.
        ``reused`` are the databases the caller left unstaged for that reason.
        Returns the nested git repositories (gitlinks) found in the index: their
        contents are *not* in the snapshot.
        """
        cache = dict(cache) if cache is not None else self._sqlite_cache()
        before = dict(cache)
        prefix = os.path.join(str(self.root), "")
        out = self._git("ls-files", "-z", "--stage").stdout
        nested: list[str] = []
        live: set[str] = set()
        for raw in out.split(b"\0"):
            if not raw:
                continue
            info, _, path_b = raw.partition(b"\t")
            rel = path_b.decode("utf-8", "surrogateescape")
            if info.startswith(b"160000"):
                nested.append(rel)
                continue
            if not rel.lower().endswith(_SQLITE_SUFFIXES):
                continue
            mode, staged = info.decode().split(" ")[:2]
            entry = cache.get(rel)
            if entry is not None and entry.get("blob") == staged \
                    and _sqlite_unchanged(prefix + rel, entry):
                live.add(rel)
                continue
            cache.pop(rel, None)
            if rel in reused:
                # Left unstaged as unchanged, yet the backup cannot be reused now (the
                # file changed meanwhile, or the staged blob is another one): stage
                # its bytes first so that a failed backup never keeps an old blob.
                self._git("update-index", "--add", "--remove", "--", rel, check=False)
            full = self.root / rel
            try:
                with open(full, "rb") as fh:
                    if fh.read(16) != _SQLITE_MAGIC:
                        continue
            except OSError:
                continue
            checked_ns = time.time_ns()
            key = _sqlite_key(prefix + rel)
            with tempfile.TemporaryDirectory(dir=str(self.dir)) as tmp:
                copy = Path(tmp) / "db"
                try:
                    src = sqlite3.connect(f"file:{full}?mode=ro", uri=True, timeout=2)
                    dst = sqlite3.connect(str(copy))
                    try:
                        src.backup(dst)
                    finally:
                        dst.close()
                        src.close()
                except sqlite3.Error:
                    continue  # keep the byte copy already staged
                blob = self._git("hash-object", "-w", str(copy)).stdout.decode().strip()
                self._git("update-index", "--cacheinfo", f"{mode},{blob},{rel}")
            if key is not None and _sqlite_key(prefix + rel) == key:
                cache[rel] = {"key": key, "blob": blob, "checked_ns": checked_ns}
                live.add(rel)
        cache = {k: v for k, v in cache.items() if k in live}
        if cache != before:
            self._write_sqlite_cache(cache)
        return nested

    # ── listing ──────────────────────────────────────────────────────────

    def snapshots(self) -> list[Snapshot]:
        """All snapshots, oldest first."""
        if not (self.git_dir / "HEAD").exists():
            return []
        out = self._git(
            "for-each-ref", "--format=%(objectname) %(tree)%00%(contents)%00%00",
            "refs/airbag/",
        ).stdout.decode("utf-8", "replace")
        snaps: list[Snapshot] = []
        for chunk in out.split("\0\0"):
            chunk = chunk.strip("\n")
            if not chunk:
                continue
            head, _, body = chunk.partition("\0")
            parts = head.split()
            if len(parts) != 2:
                continue
            try:
                meta = json.loads(body)
            except json.JSONDecodeError:
                continue
            snaps.append(Snapshot(
                seq=int(meta.pop("seq")), commit=parts[0], tree=parts[1],
                time=float(meta.pop("time", 0.0)), label=str(meta.pop("label", "")),
                files=int(meta.pop("files", 0)), meta=meta,
            ))
        snaps.sort(key=lambda s: s.seq)
        return snaps

    def latest(self) -> Snapshot | None:
        snaps = self.snapshots()
        return snaps[-1] if snaps else None

    def get(self, seq: int) -> Snapshot:
        for snap in self.snapshots():
            if snap.seq == seq:
                return snap
        raise KeyError(f"no airbag snapshot #{seq}")

    def diff(self, old: Snapshot, new: Snapshot) -> list[tuple[str, str]]:
        """``(status, path)`` changes from ``old`` to ``new`` (A/M/D/T)."""
        return [(status, path) for status, path, _ in self._raw_diff(old, new)]

    def _raw_diff(self, old: Snapshot, new: Snapshot) -> list[tuple[str, str, bool]]:
        """``(status, path, is_nested_repo)`` from ``git diff-tree --raw``."""
        out = self._git("diff-tree", "-r", "-z", "--no-renames", "--raw", old.tree, new.tree).stdout
        parts = out.split(b"\0")
        result: list[tuple[str, str, bool]] = []
        i = 0
        while i + 1 < len(parts):
            header = parts[i].decode()
            if not header.startswith(":"):
                i += 1
                continue
            fields = header[1:].split()
            path = parts[i + 1].decode("utf-8", "surrogateescape")
            gitlink = fields[0] == "160000" or fields[1] == "160000"
            result.append((fields[4][:1], path, gitlink))
            i += 2
        return result

    # ── restore ──────────────────────────────────────────────────────────

    def restore(self, target: Snapshot,
                databases: list[DatabaseDump] | None = None) -> RestoreResult:
        """Make protected files match ``target``.

        A snapshot of the current state is taken first, so every restore can
        itself be undone. Only paths the store tracks are touched: ignored
        dependency trees, files over the size limit and anything outside the
        project root are left alone. ``databases`` are safety dumps of the
        current databases, attached to that "before" snapshot.
        """
        self.ensure()
        with self.lock():
            before = self._snapshot_locked("before-restore", {"restoring_to": target.seq},
                                           databases=databases)
            result = RestoreResult(target=target, before=before)
            to_write: list[str] = []
            to_delete: list[str] = []
            modified: set[str] = set()
            for status, rel, gitlink in self._raw_diff(before, target):
                if not _safe_rel(rel):
                    result.failed.append(rel)
                elif gitlink:
                    result.kept.append(rel)  # nested repo: contents were never snapshotted
                elif status == "D":
                    to_delete.append(rel)
                elif status == "A" and os.path.lexists(self.root / rel):
                    # On disk now but not in the "before" snapshot => unprotected
                    # (grew over the size limit, became ignored...). Never clobber it.
                    result.kept.append(rel)
                else:
                    to_write.append(rel)
                    if status != "A":
                        modified.add(rel)
            for rel in to_delete:
                full = self.root / rel
                try:
                    if full.is_dir() and not full.is_symlink():
                        full.rmdir()  # gitlink / empty dir only; never recursive
                    else:
                        full.unlink()
                    result.deleted.append(rel)
                    _prune_empty_dirs(full.parent, self.root)
                except FileNotFoundError:
                    result.deleted.append(rel)
                except OSError:
                    result.failed.append(rel)
            previous_modes: dict[str, int] = {}
            for rel in to_write:
                with contextlib.suppress(OSError):
                    st = os.lstat(self.root / rel)
                    if stat.S_ISREG(st.st_mode):
                        previous_modes[rel] = stat.S_IMODE(st.st_mode)
            if to_write:
                # A directory may now stand where the snapshot had a file. Its
                # protected contents were deleted above; only remove it if it
                # is now empty — never wipe ignored data (node_modules, ...).
                for rel in to_write:
                    full = self.root / rel
                    if full.is_dir() and not full.is_symlink():
                        with contextlib.suppress(OSError):
                            full.rmdir()
                proc = self._git(
                    "restore", f"--source={target.commit}", "--worktree", "--staged",
                    "--pathspec-from-file=-", "--pathspec-file-nul",
                    input_bytes=b"\0".join(_literal(p) for p in to_write), check=False,
                )
                if proc.returncode != 0:
                    # Fall back to writing blobs one by one.
                    for rel in to_write:
                        if not self._write_blob(target, rel):
                            result.failed.append(rel)
                            continue
                        result.written.append(rel)
                else:
                    result.written.extend(to_write)
                private = self._private_modes()
                for rel in to_write:
                    _drop_sqlite_sidecars(self.root / rel)
                    mode = previous_modes.get(rel, private.get(rel))
                    if mode is not None:
                        with contextlib.suppress(OSError):
                            full = self.root / rel
                            if full.is_file() and not full.is_symlink():
                                os.chmod(full, mode)
            result.modified = [rel for rel in result.written if rel in modified]
            git_note = self._restore_git_dir()
            if git_note:
                result.written.append(git_note)
            # Make the shadow index describe the restored tree.
            self._git("read-tree", target.tree)
            self._git("update-index", "-q", "--refresh", check=False)
            self._append_log({"kind": "restore", "time": time.time(), "to": target.seq,
                              "before": before.seq, "written": len(result.written),
                              "deleted": len(result.deleted), "failed": result.failed[:50]})
            return result

    def _write_blob(self, target: Snapshot, rel: str) -> bool:
        proc = self._git("cat-file", "blob", f"{target.commit}:{rel}", check=False)
        if proc.returncode != 0:
            return False
        full = self.root / rel
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            tmp = full.with_name(full.name + ".airbag-tmp")
            tmp.write_bytes(proc.stdout)
            _platform.replace(tmp, full)
        except OSError:
            return False
        return True

    # ── read-only checkouts (vcr bisect) ─────────────────────────────────

    def materialize(self, target: Snapshot, dest: str | Path) -> None:
        """Write the files of ``target`` into ``dest``, an empty directory outside the project.

        Uses a throwaway index, so the project, the shadow index and the refs
        are not touched.
        """
        dest = Path(dest).resolve()
        if dest == self.root or self.root in dest.parents:
            raise AirbagError(f"refusing to check out a snapshot inside the project: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.iterdir()):
            raise AirbagError(f"checkout directory is not empty: {dest}")
        with tempfile.TemporaryDirectory(prefix="vcr-index-") as tmp:
            env = {"GIT_WORK_TREE": str(dest), "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            self._git("read-tree", target.tree, env_extra=env, cwd=dest)
            self._git("checkout-index", "-a", "-f", env_extra=env, cwd=dest)

    def ignored_top_level_dirs(self) -> list[str]:
        """Top-level project directories that snapshots skip (``node_modules``, ``.venv``...)."""
        if not (self.git_dir / "HEAD").exists() or not self.root.is_dir():
            return []
        out = self._git(
            "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory",
            check=False,
        ).stdout
        dirs: list[str] = []
        for raw in out.split(b"\0"):
            rel = raw.decode("utf-8", "surrogateescape")
            if not rel.endswith("/") or "/" in rel[:-1]:
                continue
            name = rel[:-1]
            if name and name != ".git" and (self.root / name).is_dir():
                dirs.append(name)
        return sorted(dirs)

    # ── action log / undo state ──────────────────────────────────────────

    def _append_log(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        fd = os.open(str(self.log_file), os.O_WRONLY | os.O_APPEND | os.O_CREAT | _O_BINARY,
                     0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def log_event(self, record: dict[str, Any]) -> None:
        """Append one action/decision record (single O_APPEND write)."""
        self._append_log({"time": time.time(), **record})

    def recent_events(self, max_bytes: int = 16384) -> list[dict[str, Any]]:
        """The last few log records, read from the file tail (cheap on long logs)."""
        try:
            with open(self.log_file, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes))
                chunk = fh.read()
        except OSError:
            return []
        lines = chunk.split(b"\n")
        if size > max_bytes:
            lines = lines[1:]  # first line may be cut
        out: list[dict[str, Any]] = []
        for line in lines:
            with contextlib.suppress(ValueError):
                if line.strip():
                    out.append(json.loads(line))
        return out

    def events(self) -> list[dict[str, Any]]:
        if not self.log_file.exists():
            return []
        events: list[dict[str, Any]] = []
        with open(self.log_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    events.append(json.loads(line))
        return events

    def read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, update: dict[str, Any]) -> None:
        state = {**self.read_state(), **update}
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
            _platform.replace(tmp, self.state_file)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def update_state(self, update: dict[str, Any]) -> None:
        with self.lock():
            self._write_state(update)

    # ── maintenance ──────────────────────────────────────────────────────

    def gc(self, keep_days: float = 7.0, keep_last: int = 50) -> int:
        """Delete snapshots older than ``keep_days`` (always keeping the newest)."""
        snaps = self.snapshots()
        cutoff = time.time() - keep_days * 86400
        doomed = [s for s in snaps[:-keep_last] if s.time < cutoff] if keep_last else [
            s for s in snaps if s.time < cutoff
        ]
        if not doomed:
            return 0
        with self.lock():
            self._git(
                "update-ref", "--stdin",
                input_bytes="".join(f"delete refs/airbag/{s.seq:08d}\n" for s in doomed).encode(),
            )
            newest_gitdir = self.latest_git_dir_backup()
            for snap in doomed:
                backup = self.gitdirs / f"{snap.seq:08d}"
                if backup != newest_gitdir:
                    _platform.rmtree(backup, ignore_errors=True)
                for entry in snap.meta.get("databases") or []:
                    dump = self.dump_file(entry) if isinstance(entry, dict) else None
                    if dump is not None:
                        with contextlib.suppress(OSError):
                            dump.unlink()
            self._remove_stale_dump_tmp()
            self._git("reflog", "expire", "--expire=now", "--all", check=False)
            self._git("gc", "--prune=now", "--quiet", check=False)
        return len(doomed)

    def _remove_stale_dump_tmp(self, max_age: float = 86400.0) -> None:
        """Dumps left half-written by a killed hook process."""
        if not self.db_dir.is_dir():
            return
        cutoff = time.time() - max_age
        for path in self.db_dir.glob(".tmp-*"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()

    def disk_usage(self) -> int:
        total = 0
        for path in self.dir.rglob("*"):
            with contextlib.suppress(OSError):
                if path.is_file():
                    total += path.stat().st_size
        return total

    def info(self) -> dict[str, Any]:
        snaps = self.snapshots()
        return {
            "root": str(self.root),
            "store": str(self.dir),
            "snapshots": len(snaps),
            "latest": asdict(snaps[-1]) if snaps else None,
            "notes": list(snaps[-1].meta.get("notes") or []) if snaps else [],
        }


def _maybe_unreadable(st: os.stat_result, uid: int) -> bool:
    """Cheap pre-check before ``os.access``: could this regular file be unreadable?"""
    if st.st_mode & stat.S_IROTH:
        return False
    return not (st.st_uid == uid and st.st_mode & stat.S_IRUSR)


def _sqlite_key(full: str) -> list[list[int] | None] | None:
    """Stat identity of a SQLite file and its sidecars (``None`` when it is missing)."""
    key: list[list[int] | None] = []
    for suffix in ("", *_SQLITE_SIDECARS):
        try:
            st = os.stat(full + suffix)
        except OSError:
            if not suffix:
                return None
            key.append(None)
            continue
        # The change time catches an in-place write that put the mtime back (on Windows
        # st_ctime is the creation time, so it is read from NTFS instead).
        ctime = _platform.change_time_ns(full + suffix, st)
        key.append([st.st_size, st.st_mtime_ns, ctime, st.st_ino, st.st_dev])
    return key


def _sqlite_unchanged(full: str, entry: dict[str, Any]) -> bool:
    """Is the database exactly as it was when ``entry``'s backup was taken?

    Same size, mtime, ctime, inode and device for the file and each sidecar,
    none of them modified within ``_SQLITE_RACY_NS`` before that backup
    (otherwise a later write could carry the same coarse timestamps).
    """
    key = _sqlite_key(full)
    if key is None or key != entry.get("key"):
        return False
    limit = int(entry.get("checked_ns") or 0) - _SQLITE_RACY_NS
    return all(part is None or (part[1] < limit and part[2] < limit) for part in key)


def _behind_symlink(prefix: str, rel: str, cache: dict[str, bool]) -> bool:
    """Is one of ``rel``'s parent directories a symlink? (``lstat`` once per directory)"""
    parts = rel.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        parent = "/".join(parts[:i])
        hit = cache.get(parent)
        if hit is None:
            try:
                hit = stat.S_ISLNK(os.lstat(prefix + parent).st_mode)
            except OSError:
                hit = False
            cache[parent] = hit
        if hit:
            return True
    return False


def _private_modes_of(prefix: str, paths: list[str]) -> dict[str, int]:
    """Modes of the regular files among ``paths`` that have no group/other bits."""
    private: dict[str, int] = {}
    for rel in paths:
        with contextlib.suppress(OSError):
            st = os.lstat(prefix + rel)
            if stat.S_ISREG(st.st_mode) and not st.st_mode & 0o077:
                private[rel] = stat.S_IMODE(st.st_mode)
    return private


def _literal(rel: str) -> bytes:
    """Pathspec that matches ``rel`` literally (no glob expansion)."""
    return (":(literal)" + rel).encode("utf-8", "surrogateescape")


def _safe_rel(rel: str) -> bool:
    # A backslash is an ordinary filename character on POSIX.
    parts = (rel.replace("\\", "/") if os.name == "nt" else rel).split("/")
    return bool(rel) and not rel.startswith("/") and ".." not in parts and parts[0] != ".git"


def _glob_escape(path: str) -> str:
    """Escape glob metacharacters so a literal directory name is safe in a :(glob) pathspec."""
    return re.sub(r"([*?\[\]\\])", r"\\\1", path)


def _matches_exclude(rel: str, pattern: str) -> bool:
    if pattern.endswith("/"):
        name = pattern.rstrip("/")
        return name in rel.split("/")[:-1]
    return fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)


def _prune_empty_dirs(directory: Path, root: Path) -> None:
    while directory != root and root in directory.parents:
        try:
            directory.rmdir()
        except OSError:
            return
        directory = directory.parent


def _drop_sqlite_sidecars(path: Path) -> None:
    """A restored SQLite file must not be paired with a newer WAL/SHM."""
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".db3"}:
        for suffix in ("-wal", "-shm", "-journal"):
            with contextlib.suppress(OSError):
                path.with_name(path.name + suffix).unlink()
