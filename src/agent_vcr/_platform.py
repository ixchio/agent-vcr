"""Cross-platform helpers for the few places POSIX and Windows really differ.

Standard library only: the airbag hook imports this on every agent tool call.
Every helper keeps the exact POSIX behavior on Linux/macOS and only branches
on ``sys.platform == "win32"`` where Windows needs something else:

* file locks — ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows;
* ``fsync`` of a path — Windows can only flush a handle opened for writing;
* ``os.replace`` — Windows refuses to replace a file another process has open
  for a moment (a reader, an antivirus scan), so it is retried briefly;
* ``rmtree`` — Windows refuses to delete read-only files (git objects);
* killing a command and everything it started — process groups on POSIX,
  ``taskkill /T`` on Windows;
* UTF-8 standard streams — Windows consoles and pipes default to a legacy
  code page that cannot encode the emoji and arrows the CLI prints.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import IO, Any, Callable

WINDOWS = sys.platform == "win32"

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt as _msvcrt
except ImportError:  # POSIX
    _msvcrt = None  # type: ignore[assignment]


# ── file locks ──────────────────────────────────────────────────────────


def try_lock(handle: IO[Any]) -> bool:
    """Take an exclusive lock on an open file without waiting; False when it is held.

    Returns True when no OS lock exists on this platform (callers then rely on
    their own in-process serialization).
    """
    try:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        elif _msvcrt is not None:  # pragma: no cover - Windows
            # msvcrt locks byte ranges from the current position: always lock byte 0.
            handle.seek(0)
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_NBLCK, 1)
    except (BlockingIOError, PermissionError):
        return False
    except OSError as err:
        # msvcrt reports a held lock as EACCES / EDEADLOCK.
        if _msvcrt is not None and err.errno in (13, 36):  # pragma: no cover - Windows
            return False
        raise
    return True


def unlock(handle: IO[Any]) -> None:
    """Release a lock taken with :func:`try_lock` (never raises)."""
    with contextlib.suppress(OSError):
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
        elif _msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0)
            _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def locked(handle: IO[Any], timeout: float | None = None,
           poll: float = 0.02) -> Iterator[None]:
    """Hold an exclusive lock on ``handle`` for the ``with`` body.

    ``timeout=None`` waits forever; otherwise :class:`TimeoutError` is raised
    once ``timeout`` seconds have passed without getting the lock.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while not try_lock(handle):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("timed out waiting for a file lock")
        time.sleep(poll)
    try:
        yield
    finally:
        unlock(handle)


# ── durable, atomic file operations ─────────────────────────────────────


def fsync_path(path: str | os.PathLike[str]) -> None:
    """Flush a file that was written through another handle to disk.

    POSIX can fsync a read-only descriptor; Windows (``_commit``) needs one
    opened for writing, so the file is opened read-write there (never truncated).
    """
    flags = os.O_RDWR if WINDOWS else os.O_RDONLY
    fd = os.open(os.fspath(path), flags | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str],
            timeout: float = 5.0) -> None:
    """``os.replace`` that survives Windows' transient sharing violations.

    On Windows a rename onto a file fails with ``PermissionError`` while any
    other process (a concurrent reader, a virus scanner, the search indexer)
    has it open; the open is short, so the rename is retried until ``timeout``.
    On POSIX this is exactly ``os.replace``.
    """
    if not WINDOWS:
        os.replace(src, dst)
        return
    deadline = time.monotonic() + timeout  # pragma: no cover - Windows
    delay = 0.005  # pragma: no cover - Windows
    while True:  # pragma: no cover - Windows
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.1)


def _make_writable_and_retry(func: Callable[..., Any], path: str) -> None:
    """``rmtree`` error handler: clear the read-only bit (git objects) and retry once."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return  # already gone
    if not stat.S_ISLNK(mode):
        os.chmod(path, stat.S_IMODE(mode) | stat.S_IWRITE)
    func(path)


def rmtree(path: str | os.PathLike[str], ignore_errors: bool = False) -> None:
    """``shutil.rmtree`` that also removes read-only files, like ``rm -rf``.

    Git marks its object files read-only; POSIX deletes them anyway (only the
    directory's permissions matter) but Windows refuses, so on Windows the
    read-only attribute is cleared and the delete retried. Symlinks are never
    followed (``shutil.rmtree`` semantics).
    """
    if not WINDOWS:
        shutil.rmtree(path, ignore_errors=ignore_errors)
        return

    def onerror(func: Callable[..., Any], p: str, _exc: Any) -> None:  # pragma: no cover
        try:
            _make_writable_and_retry(func, p)
        except OSError:
            if not ignore_errors:
                raise

    if sys.version_info >= (3, 12):  # pragma: no cover - Windows
        shutil.rmtree(path, onexc=onerror)
    else:  # pragma: no cover - Windows
        shutil.rmtree(path, onerror=onerror)


# ── processes ───────────────────────────────────────────────────────────


def new_process_group_kwargs() -> dict[str, Any]:
    """``Popen`` kwargs that start a command in its own process group/session.

    Lets :func:`kill_process_tree` stop the command *and* everything it
    spawned (a shell's children, a test runner's workers).
    """
    if WINDOWS:  # pragma: no cover - Windows
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """Forcefully stop ``proc`` and its descendants (never raises).

    POSIX: SIGKILL to the process group started by
    :func:`new_process_group_kwargs`. Windows: ``taskkill /T /F``, which walks
    the child-process tree, falling back to ``TerminateProcess``.
    """
    if WINDOWS:  # pragma: no cover - Windows
        if proc.poll() is None:
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=10,
                )
        with contextlib.suppress(OSError):
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined,unused-ignore]
    except OSError:
        with contextlib.suppress(OSError):
            proc.kill()


# ── file change time ────────────────────────────────────────────────────

#: 100 ns ticks between 1601-01-01 (FILETIME) and 1970-01-01 (Unix epoch).
_FILETIME_EPOCH = 116_444_736_000_000_000

if sys.platform == "win32":  # pragma: no cover - Windows
    import ctypes
    from ctypes import wintypes

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = [("CreationTime", ctypes.c_longlong),
                    ("LastAccessTime", ctypes.c_longlong),
                    ("LastWriteTime", ctypes.c_longlong),
                    ("ChangeTime", ctypes.c_longlong),
                    ("FileAttributes", wintypes.DWORD)]

    def _windows_change_time_ns(path: str) -> int | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel32.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        # FILE_READ_ATTRIBUTES only, every share mode: never blocks a writer (SQLite).
        handle = create(path, 0x80, 0x1 | 0x2 | 0x4, None, 3, 0x02000000, None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            return None
        try:
            info = _FileBasicInfo()
            if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):  # FileBasicInfo
                return None
            return (int(info.ChangeTime) - _FILETIME_EPOCH) * 100
        finally:
            kernel32.CloseHandle(handle)


def change_time_ns(path: str, st: os.stat_result) -> int:
    """When ``path``'s data or metadata last changed (``st`` is its ``os.stat``).

    POSIX: ``st_ctime_ns``, which every write moves (even when the mtime is put
    back). On Windows ``st_ctime`` is the *creation* time, so the NTFS change
    time is read instead; when it cannot be read the result is "now", which
    makes a caller comparing change times assume the file changed.
    """
    if sys.platform == "win32":  # pragma: no cover - Windows
        try:
            found = _windows_change_time_ns(path)
        except (OSError, AttributeError, ValueError):
            found = None
        return found if found is not None else time.time_ns()
    return st.st_ctime_ns


# ── text I/O ────────────────────────────────────────────────────────────


def utf8_stdio() -> None:
    """Make stdout/stderr encode as UTF-8 when they would use a legacy code page.

    A Windows console or pipe defaults to e.g. cp1252, which cannot encode the
    emoji and arrows the CLI prints, so every print raised UnicodeEncodeError.
    Streams already using UTF-8 (every POSIX locale in practice, or
    ``PYTHONIOENCODING`` / ``PYTHONUTF8``) are left alone; anything still
    unencodable is replaced instead of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None or encoding in ("utf8", "utf8sig"):
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")
