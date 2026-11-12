"""Which directories the airbag protects (snapshots), and what it does elsewhere.

User-level hooks fire in *every* directory an agent runs in: a git
repository, but also ``~``, ``/`` or a downloads folder. Snapshotting those
would be slow, huge and pointless, so the rule is:

* **protected** (snapshot before every change, undo with ``vcr undo``):
  a git repository; a directory opted in with ``vcr airbag protect`` (it
  writes ``.vcr/airbag.json``; a bare ``.vcr/`` folder, which the recorder
  creates, does not count); a directory that already has an airbag store
  (so a project whose ``.git`` the agent deleted stays protected). The
  nearest of these wins, so a dotfiles repository at ``~`` does not hide a
  protected project below it;
* **not protected** (everything else, and always the home directory, the
  filesystem root, Desktop/Documents/Downloads): nothing is snapshotted.
  Irreversible commands are stopped exactly as everywhere else, and so are
  commands that delete a directory tree or wipe many files (``rm -r``,
  ``git clean``, ``find -delete``, ``rsync --delete``). Everything else goes
  ahead, with a notice that this folder is not protected.

Standard library only: this runs inside the hook on every tool call.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_vcr.airbag.store import (
    AirbagError,
    ShadowStore,
    Snapshot,
    _clean_env,
    _is_unsafe_root,
    _nearest_known_root,
    state_home,
)

#: ``<dir>/.vcr/airbag.json`` opts a non-git directory in (``vcr airbag protect``).
OPT_IN_DIR = ".vcr"
OPT_IN_FILE = "airbag.json"
PROTECT_HINT = "`vcr airbag protect` to enable snapshots"
#: Shared state (pending approvals, recent decisions) for unprotected directories.
GUARD_DIR_NAME = "_unprotected"
#: The guard log only needs recent decisions; trim it past this size.
GUARD_LOG_MAX_BYTES = 256 * 1024

KIND_TEXT = {
    "git": "git repository",
    "opt-in": "opted in with `vcr airbag protect`",
    "store": "already has airbag snapshots",
    "unsafe": "never snapshotted",
    "plain": "not a git repository, not opted in",
    "missing": "directory does not exist",
}


@dataclass
class Scope:
    """Protection status of one directory."""

    path: Path
    #: The project the directory belongs to (or the directory itself).
    root: Path
    protected: bool
    #: "git" | "opt-in" | "store" (protected); "unsafe" | "plain" | "missing" (not).
    kind: str
    #: For "unsafe": what it is ("your home directory", ...).
    unsafe: str | None = None

    @property
    def why(self) -> str:
        if self.kind == "unsafe":
            return f"{self.unsafe}: never snapshotted"
        return KIND_TEXT.get(self.kind, self.kind)

    @property
    def short_why(self) -> str:
        """A few words for notices and the status line."""
        if self.kind == "unsafe":
            return self.unsafe or "never snapshotted"
        return {"plain": "not a git repo", "missing": "folder is gone"}.get(self.kind, self.kind)

    @property
    def can_opt_in(self) -> bool:
        """``vcr airbag protect`` would work for the directory itself."""
        return self.kind != "missing" and _is_unsafe_root(self.path) is None

    @property
    def unprotected_reason(self) -> str:
        """Why a tree-deleting command here is stopped (fed to the human and the agent)."""
        hint = f" ({PROTECT_HINT})" if self.can_opt_in else ""
        return (f"this folder is not protected by the airbag ({self.short_why}), so nothing was "
                f"snapshotted and this cannot be undone{hint}")

    @property
    def notice(self) -> str:
        """Shown when a change goes ahead without a snapshot."""
        hint = f" — {PROTECT_HINT}" if self.can_opt_in else ""
        return f"🛟 Agent VCR airbag: not protected here ({self.short_why}){hint}"

    @property
    def opt_in_hint(self) -> str:
        if not self.can_opt_in:
            return ("work inside a project folder (a git repository, or one you opt in with "
                    "`vcr airbag protect`) to get snapshots")
        return f"run `vcr airbag protect` (or `git init`) in {self.path}"

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "root": str(self.root), "protected": self.protected,
                "kind": self.kind, "why": self.why}


def _git_toplevel(path: Path) -> Path | None:
    """The git top-level of ``path`` (read-only ``rev-parse`` on the user's repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(path), capture_output=True, text=True, timeout=10,
            env=_clean_env(), stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).resolve()
    return None


def _never_walk_into(path: Path) -> bool:
    """Home and the filesystem root never count as an opt-in parent."""
    with contextlib.suppress(RuntimeError, OSError):
        if path == Path.home().resolve():
            return True
    return path == Path(path.anchor)


def marker_path(directory: str | os.PathLike[str]) -> Path:
    return Path(directory) / OPT_IN_DIR / OPT_IN_FILE


def nearest_opt_in(path: Path) -> Path | None:
    """The nearest directory at or above ``path`` opted in with ``vcr airbag protect``."""
    for candidate in (path, *path.parents):
        if _never_walk_into(candidate):
            return None
        with contextlib.suppress(OSError):
            if marker_path(candidate).is_file():
                return candidate
    return None


def resolve(start: str | os.PathLike[str]) -> Scope:
    """Protection status of ``start`` (see the module docstring for the rule)."""
    path = Path(os.path.abspath(Path(start).expanduser()))
    if not path.exists():
        # The agent may have deleted the project: its store still knows the root.
        known = _nearest_known_root(path)
        if known is not None and _is_unsafe_root(known) is None:
            return Scope(path, known, True, "store")
        return Scope(path, path, False, "missing")
    path = path.resolve()
    top = _git_toplevel(path)
    top_unsafe = _is_unsafe_root(top) if top is not None else None
    if top is not None and top_unsafe is None:
        return Scope(path, top, True, "git")
    # Not in a usable git repository (none, or a dotfiles repo at ~): the nearest opt-in
    # or existing store below it wins. Never the unsafe directory itself.
    found: list[tuple[Path, str]] = []
    known = _nearest_known_root(path)
    if known is not None:
        found.append((known, "store"))
    opt_in = nearest_opt_in(path)
    if opt_in is not None:
        found.append((opt_in, "opt-in"))
    found = [(root, kind) for root, kind in found if _is_unsafe_root(root) is None and
             (top is None or root != top and top in root.parents)]
    if found:
        root, kind = max(found, key=lambda item: len(item[0].parts))
        return Scope(path, root, True, kind)
    if top is not None:
        return Scope(path, top, False, "unsafe", top_unsafe)
    unsafe = _is_unsafe_root(path)
    if unsafe:
        return Scope(path, path, False, "unsafe", unsafe)
    return Scope(path, path, False, "plain")


def protect(directory: str | os.PathLike[str]) -> tuple[Path, bool]:
    """Opt ``directory`` in (``.vcr/airbag.json``). Returns (marker, created)."""
    path = Path(os.path.abspath(Path(directory).expanduser()))
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    unsafe = _is_unsafe_root(path.resolve())
    if unsafe:
        raise ValueError(f"refusing to protect {unsafe} ({path}): it is too broad to snapshot")
    marker = marker_path(path)
    if marker.is_file():
        return marker, False
    marker.parent.mkdir(exist_ok=True)
    tmp = marker.with_name(f"{OPT_IN_FILE}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"protect": True, "created_by": "vcr airbag protect"}) + "\n",
                       encoding="utf-8")
        os.replace(tmp, marker)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return marker, True


def unprotect(directory: str | os.PathLike[str]) -> Path | None:
    """Remove the opt-in that covers ``directory``; returns the marker removed (or None)."""
    path = Path(os.path.abspath(Path(directory).expanduser()))
    root = nearest_opt_in(path.resolve()) if path.exists() else None
    if root is None:
        return None
    marker = marker_path(root)
    marker.unlink()
    with contextlib.suppress(OSError):
        marker.parent.rmdir()  # only when nothing else is in .vcr/
    return marker




def resolve_first(*candidates: str | None) -> Scope:
    """The first protected scope among ``candidates`` (else the first one's scope).

    Hooks pass the agent's project directory, then the tool call's cwd: an
    agent started in ``~`` that works inside a repository is still protected.
    """
    first: Scope | None = None
    seen: set[str] = set()
    for raw in candidates:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        scope = resolve(raw)
        if scope.protected:
            return scope
        if first is None:
            first = scope
    return first if first is not None else resolve(os.getcwd())


class GuardStore(ShadowStore):
    """State for unprotected directories: approvals and recent decisions, no snapshots.

    One shared store (``<airbag home>/_unprotected``) so ``vcr airbag allow``
    works from any unprotected directory. It never has a ``root.txt``, so it
    is never mistaken for a protected project.
    """

    def __init__(self, root: str | os.PathLike[str], *, home: str | Path | None = None) -> None:
        base = Path(home).expanduser() if home is not None else state_home()
        super().__init__(base / GUARD_DIR_NAME, home=base)
        self.root = Path(os.path.abspath(Path(root).expanduser()))
        self.dir = base / GUARD_DIR_NAME
        self.git_dir = self.dir / "no-snapshots"
        self.index_file = self.dir / "no-index"
        self.log_file = self.dir / "log.jsonl"
        self.state_file = self.dir / "state.json"

    def ensure(self) -> None:
        raise AirbagError(f"{self.root} is not protected by the airbag (no snapshots)")

    def snapshot(self, label: str = "snapshot", meta: dict[str, Any] | None = None,
                 **_: Any) -> Snapshot:
        raise AirbagError(f"{self.root} is not protected by the airbag (no snapshots)")

    def snapshots(self) -> list[Snapshot]:
        return []

    def log_event(self, record: dict[str, Any]) -> None:
        super().log_event(record)
        with contextlib.suppress(OSError):
            if self.log_file.stat().st_size > GUARD_LOG_MAX_BYTES:
                self._trim_log()

    def _trim_log(self) -> None:
        with self.lock():
            data = self.log_file.read_bytes()
            keep = data[-GUARD_LOG_MAX_BYTES // 4:]
            keep = keep[keep.find(b"\n") + 1:]
            tmp = self.log_file.with_name(self.log_file.name + f".{os.getpid()}.tmp")
            try:
                tmp.write_bytes(keep)
                os.replace(tmp, self.log_file)
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
