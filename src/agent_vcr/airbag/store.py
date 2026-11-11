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
class RestoreResult:
    """What a restore changed on disk."""

    target: Snapshot
    before: Snapshot
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: Present on disk but not protected (too large, ignored, nested repo): left untouched.
    kept: list[str] = field(default_factory=list)

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
                text = marker.read_text().strip()
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
            env=_clean_env(), stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).resolve()
    return _nearest_known_root(path) or path


def _clean_env() -> dict[str, str]:
    """Environment without GIT_* variables that could redirect git elsewhere."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _is_unsafe_root(root: Path) -> str | None:
    home = Path.home().resolve()
    if root == Path(root.anchor):
        return "the filesystem root"
    if root == home:
        return "your home directory"
    if root in {home / "Desktop", home / "Documents", home / "Downloads"}:
        return f"{root.name} (too broad to protect)"
    return None


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
        """Create the shadow git dir on first use (idempotent)."""
        if (self.git_dir / "HEAD").exists():
            self._ensure_attributes()
            return
        if shutil.which("git") is None:
            raise AirbagError("git is required for the airbag but was not found on PATH")
        self.dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.dir, 0o700)  # snapshots can contain .env files
        for cmd in (
            ["git", "init", "-q", "--bare", str(self.git_dir)],
            # Not really bare: GIT_WORK_TREE always points at the project.
            ["git", "--git-dir", str(self.git_dir), "config", "core.bare", "false"],
        ):
            subprocess.run(
                cmd, check=True, capture_output=True, timeout=self.git_timeout,
                env=_clean_env(), stdin=subprocess.DEVNULL,
            )
        info = self.git_dir / "info"
        info.mkdir(exist_ok=True)
        (info / "exclude").write_text(
            "# agent-vcr airbag defaults\n" + "\n".join(self.excludes) + "\n"
        )
        (self.dir / "root.txt").write_text(str(self.root) + "\n")
        self._ensure_attributes()

    def _ensure_attributes(self) -> None:
        """Store bytes exactly: no eol conversion, no LFS/clean filters, no encoding.

        ``$GIT_DIR/info/attributes`` overrides in-tree ``.gitattributes``, so a
        project's ``* text=auto`` cannot rewrite line endings on restore.
        """
        attrs = self.git_dir / "info" / "attributes"
        if not attrs.exists():
            attrs.parent.mkdir(parents=True, exist_ok=True)
            attrs.write_text("* -text -eol -filter -ident -working-tree-encoding\n")

    @contextlib.contextmanager
    def lock(self, timeout: float = 30.0) -> Iterator[None]:
        """Serialize store operations across concurrent hook processes."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "lock", "a+") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows
                yield
                return
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise AirbagError("timed out waiting for the airbag lock") from None
                    time.sleep(0.02)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _git(
        self, *args: str, input_bytes: bytes | None = None, check: bool = True,
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
        cmd = [
            "git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
            "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
            "-c", "core.quotepath=false", "-c", "core.fsmonitor=false",
            "-c", "gc.auto=0", "-c", "advice.addIgnoredFile=false", *args,
        ]
        result = subprocess.run(
            cmd, cwd=str(self.root) if self.root.exists() else str(self.dir),
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
                 *, backup_git_dir: bool = False) -> Snapshot:
        """Record the current state of the project. Cheap when nothing changed.

        ``backup_git_dir`` also preserves the project's own ``.git`` (as
        hardlinks — near-instant, no extra space) for commands that would
        delete it, e.g. ``rm -rf`` of the project.
        """
        self.ensure()
        with self.lock():
            snap = self._snapshot_locked(label, meta or {})
            if backup_git_dir:
                self._backup_git_dir(snap.seq)
            return snap

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
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)

        def link_or_copy(a: str, b: str) -> object:
            # Git replaces files via rename and never edits objects in place,
            # so a hardlink is a faithful point-in-time copy.
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
            return b

        shutil.copytree(src, tmp, symlinks=True, copy_function=link_or_copy)
        os.replace(tmp, dst)
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
                                  env=_clean_env(), stdin=subprocess.DEVNULL)
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

        def link_or_copy(a: str, b: str) -> object:
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
            return b

        shutil.copytree(backup, target, symlinks=True, copy_function=link_or_copy)
        return note

    def _snapshot_locked(self, label: str, meta: dict[str, Any]) -> Snapshot:
        if not self.root.exists():
            self.root.mkdir(parents=True)
        skipped: list[str] = []
        changed = self._changed_paths()
        if len(changed) > self.max_files:
            raise AirbagError(
                f"{len(changed)} files would be snapshotted (limit {self.max_files}); "
                "add excludes or raise AGENT_VCR_AIRBAG_MAX_FILES"
            )
        to_add: list[str] = []
        # Untracked nested repositories are listed as "dir/": git cannot add their
        # contents (only a gitlink, and not even that before their first commit).
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
            # Files over the limit must not stay in the index with an old, small
            # blob: a later restore would write that stale copy over the real file.
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
        self._remember_private_modes(to_add + precious)

        tree = self._git("write-tree").stdout.decode().strip()
        last = self.latest()
        if last is not None and last.tree == tree:
            return last  # nothing changed since the previous snapshot
        files = int(self._git("ls-files", "-z").stdout.count(b"\0"))
        seq = (last.seq + 1) if last is not None else 1
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

    def _remember_private_modes(self, paths: list[str]) -> None:
        """Remember permission bits of private files (e.g. ``.env`` at 0600).

        Git only stores the executable bit, so a deleted-then-restored secret
        would otherwise come back group/world-readable.
        """
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
        tmp = self.dir / "modes.json.tmp"
        tmp.write_text(json.dumps(modes))
        os.replace(tmp, self.dir / "modes.json")

    def _private_modes(self) -> dict[str, int]:
        try:
            data = json.loads((self.dir / "modes.json").read_text())
        except (OSError, ValueError):
            return {}
        return {str(k): int(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _precious_ignored_files(self) -> tuple[list[str], list[str]]:
        """Ignored files matching the precious patterns (e.g. ``.env``): (small, too big)."""
        if not self.precious:
            return [], []
        out = self._git(
            "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory",
        ).stdout
        result: list[str] = []
        too_big: list[str] = []
        for raw in out.split(b"\0"):
            if not raw or raw.endswith(b"/"):
                continue  # collapsed ignored directory (node_modules/, ...)
            rel = raw.decode("utf-8", "surrogateescape")
            name = rel.rsplit("/", 1)[-1]
            if not any(fnmatch.fnmatch(name, pat) for pat in self.precious):
                continue
            if any(_matches_exclude(rel, pat) for pat in self.excludes):
                continue
            try:
                if (self.root / rel).stat().st_size <= self.max_file_bytes:
                    result.append(rel)
                else:
                    too_big.append(rel)
            except OSError:
                continue
        return result, too_big

    def _stage_sqlite_consistently(self) -> list[str]:
        """Replace staged SQLite files with a consistent online-backup copy.

        Copying a live SQLite file byte-for-byte can capture a torn state
        (especially in WAL mode). The backup API produces a consistent image.
        Returns the nested git repositories (gitlinks) found in the index: their
        contents are *not* in the snapshot.
        """
        out = self._git("ls-files", "-z", "--stage").stdout
        nested: list[str] = []
        for raw in out.split(b"\0"):
            if not raw:
                continue
            info, _, path_b = raw.partition(b"\t")
            rel = path_b.decode("utf-8", "surrogateescape")
            if info.startswith(b"160000"):
                nested.append(rel)
                continue
            if not rel.lower().endswith((".db", ".sqlite", ".sqlite3", ".db3")):
                continue
            full = self.root / rel
            try:
                with open(full, "rb") as fh:
                    if fh.read(16) != _SQLITE_MAGIC:
                        continue
            except OSError:
                continue
            mode = info.split(b" ")[0].decode()
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

    def restore(self, target: Snapshot) -> RestoreResult:
        """Make protected files match ``target``.

        A snapshot of the current state is taken first, so every restore can
        itself be undone. Only paths the store tracks are touched: ignored
        dependency trees, files over the size limit and anything outside the
        project root are left alone.
        """
        self.ensure()
        with self.lock():
            before = self._snapshot_locked("before-restore", {"restoring_to": target.seq})
            result = RestoreResult(target=target, before=before)
            to_write: list[str] = []
            to_delete: list[str] = []
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
            os.replace(tmp, full)
        except OSError:
            return False
        return True

    # ── action log / undo state ──────────────────────────────────────────

    def _append_log(self, record: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        fd = os.open(str(self.log_file), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
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
            data = json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, update: dict[str, Any]) -> None:
        state = {**self.read_state(), **update}
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix="state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(state, fh)
            os.replace(tmp, self.state_file)
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
                    shutil.rmtree(backup, ignore_errors=True)
            self._git("reflog", "expire", "--expire=now", "--all", check=False)
            self._git("gc", "--prune=now", "--quiet", check=False)
        return len(doomed)

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
        }


def _literal(rel: str) -> bytes:
    """Pathspec that matches ``rel`` literally (no glob expansion)."""
    return (":(literal)" + rel).encode("utf-8", "surrogateescape")


def _safe_rel(rel: str) -> bool:
    # A backslash is an ordinary filename character on POSIX.
    parts = (rel.replace("\\", "/") if os.name == "nt" else rel).split("/")
    return bool(rel) and not rel.startswith("/") and ".." not in parts and parts[0] != ".git"


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
