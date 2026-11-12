"""Local isolated process sandbox with filesystem diffing (Phase 7)."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_vcr._platform import kill_process_tree, new_process_group_kwargs
from agent_vcr.sandbox.base import BaseSandbox
from agent_vcr.sandbox.models import (
    ExecutionResult,
    FileMetadata,
    FilesystemSnapshot,
    SandboxConfig,
)

logger = logging.getLogger(__name__)


#: Host variables passed through to sandboxed commands by default. Everything
#: else (API keys, cloud credentials, tokens) is withheld unless the caller sets
#: ``SandboxConfig.inherit_env=True`` or passes it explicitly via ``env``.
_SAFE_ENV_KEYS = ("PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
                  "SYSTEMROOT", "COMSPEC", "PATHEXT")


class LocalSandbox(BaseSandbox):
    """Run commands in a scratch directory on the local machine.

    Commands run through the platform shell, like ``subprocess`` with
    ``shell=True``: ``/bin/sh`` on POSIX, ``cmd.exe`` (``%COMSPEC%``) on
    Windows — so write ``$VAR`` on POSIX and ``%VAR%`` on Windows. The
    command's home and temp directory are the sandbox root (``HOME`` and
    ``TMPDIR``; on Windows also ``USERPROFILE``, ``TEMP`` and ``TMP``).

    .. warning::
        This is **not a security boundary**. Commands run as the current user
        through a shell and can read or write anything that user can (absolute
        paths, ``..``), and can use the network. Only ``read_file`` /
        ``write_file`` are confined to the sandbox root. Host secrets are not
        inherited by default (see ``_SAFE_ENV_KEYS``), but ``network_enabled``,
        ``memory_limit_mb`` and ``cpu_count`` are **not enforced**. Use
        :class:`~agent_vcr.sandbox.E2BSandbox` (a microVM) for untrusted code.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        workdir: str | Path | None = None,
        cleanup: bool = True,
    ) -> None:
        """Create a sandbox.

        ``cleanup=False`` keeps an auto-created scratch directory after
        :meth:`stop` (and after garbage collection / interpreter exit);
        ``root_path`` stays readable so its contents can be inspected.
        A caller-supplied ``workdir`` is never deleted.
        """
        super().__init__(config=config)
        self._custom_workdir = Path(workdir).resolve() if workdir is not None else None
        self._cleanup = cleanup
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._root_path: Path | None = None
        self._alive = False

    @property
    def root_path(self) -> Path:
        """Root workspace path for this sandbox."""
        if self._root_path is None:
            raise RuntimeError("Sandbox has not been started yet. Call start() first.")
        return self._root_path

    @property
    def is_alive(self) -> bool:
        return self._alive and self._root_path is not None and self._root_path.exists()

    def start(self) -> None:
        """Initialize the isolated workspace directory."""
        if self._alive:
            return

        if self._custom_workdir is not None:
            self._root_path = self._custom_workdir
            self._root_path.mkdir(parents=True, exist_ok=True)
        elif self._cleanup:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="agent_vcr_sandbox_")
            self._root_path = Path(self._temp_dir.name).resolve()
        else:
            # mkdtemp has no finalizer: the directory outlives this object.
            self._root_path = Path(tempfile.mkdtemp(prefix="agent_vcr_sandbox_")).resolve()

        self._alive = True
        if not self.config.network_enabled:
            logger.warning(
                "LocalSandbox cannot disable network access (network_enabled=False is "
                "not enforced); use E2BSandbox for isolation"
            )
        logger.debug("LocalSandbox started at %s", self._root_path)

    def stop(self) -> None:
        """Clean up the sandbox workspace."""
        if not self._alive:
            return

        self._alive = False
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up sandbox temporary directory: %s", exc)
            self._temp_dir = None
            self._root_path = None
        # cleanup=False / custom workdir: keep the directory and root_path.

        logger.debug("LocalSandbox stopped")

    def _resolve_safe_path(self, rel_path: str | Path) -> Path:
        """Resolve a path and ensure it stays within the sandbox root directory."""
        root = self.root_path
        target = (root / rel_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as err:
            raise PermissionError(
                f"Path traversal detected: {rel_path} escapes sandbox root {root}"
            ) from err
        return target

    def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute command in sandbox with timeout and stdout/stderr capture."""
        if not self.is_alive:
            self.start()

        effective_timeout = timeout if timeout is not None else self.config.timeout_seconds
        effective_cwd = self._resolve_safe_path(workdir) if workdir else self.root_path

        # Minimal environment: never leak host secrets (API keys, tokens) into
        # agent-run commands unless explicitly requested.
        if self.config.inherit_env:
            cmd_env = dict(os.environ)
        else:
            cmd_env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
            cmd_env["HOME"] = str(self.root_path)
            cmd_env["TMPDIR"] = str(self.root_path)
            if sys.platform == "win32":  # where Windows programs look instead
                for key in ("USERPROFILE", "TEMP", "TMP"):
                    cmd_env[key] = str(self.root_path)
        cmd_env.update(self.config.env)
        if env:
            cmd_env.update(env)

        start_time = time.perf_counter()
        timed_out = False
        exit_code = 0
        stdout = ""
        stderr = ""
        error_msg: str | None = None

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(effective_cwd),
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8", errors="replace",
                **new_process_group_kwargs(),
            )
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = -1
                error_msg = f"Command timed out after {effective_timeout:.2f}s"
                # The shell's children too (process group / Windows process tree).
                kill_process_tree(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    proc.kill()
                    stdout, stderr = "", ""
        except Exception as exc:  # noqa: BLE001
            exit_code = -1
            error_msg = str(exc)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            error=error_msg,
        )

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        """Read a file safely from within the sandbox."""
        target = self._resolve_safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        if binary:
            return target.read_bytes()
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file atomically inside the sandbox, creating parent directories."""
        target = self._resolve_safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                        dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def list_files(self, path: str = ".") -> list[str]:
        """List all relative file paths under path."""
        target = self._resolve_safe_path(path)
        if not target.exists():
            return []
        paths: list[str] = []
        for root, _, files in os.walk(target):
            for f in files:
                p = Path(root) / f
                try:
                    if p.is_symlink():
                        p.resolve().relative_to(self.root_path)
                    paths.append(p.relative_to(self.root_path).as_posix())
                except (ValueError, OSError):
                    continue
        return sorted(paths)

    def snapshot_filesystem(self, path: str = ".") -> FilesystemSnapshot:
        """Compute SHA-256 hash, size, and mtime for files under ``path``.

        The snapshot's ``root_path`` is the scanned directory, so
        ``diff_filesystem(before)`` re-scans the same subtree.
        """
        if not self.is_alive:
            self.start()

        target = self._resolve_safe_path(path)
        entries: dict[str, FileMetadata] = {}

        if target.exists():
            for root, dirs, files in os.walk(target):
                # Ignore hidden VCS / cache directories
                dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}]
                for f in files:
                    file_path = Path(root) / f
                    try:
                        if file_path.is_symlink():
                            file_path.resolve().relative_to(self.root_path)
                        rel_path = file_path.relative_to(self.root_path).as_posix()
                        stat = file_path.stat()
                        # Compute sha256
                        h = hashlib.sha256()
                        with open(file_path, "rb") as fp:
                            while chunk := fp.read(65536):
                                h.update(chunk)
                        entries[rel_path] = FileMetadata(
                            path=rel_path,
                            size=stat.st_size,
                            sha256=h.hexdigest(),
                            mtime=stat.st_mtime,
                            is_dir=False,
                        )
                    except (OSError, ValueError):
                        continue

        return FilesystemSnapshot(
            root_path=str(target),
            timestamp=time.time(),
            entries=entries,
        )
