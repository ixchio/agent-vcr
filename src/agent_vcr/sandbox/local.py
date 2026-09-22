"""Local isolated process sandbox with filesystem diffing (Phase 7)."""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from agent_vcr.sandbox.base import BaseSandbox
from agent_vcr.sandbox.models import (
    ExecutionResult,
    FileMetadata,
    FilesystemSnapshot,
    SandboxConfig,
)

logger = logging.getLogger(__name__)


class LocalSandbox(BaseSandbox):
    """Local execution sandbox operating within an isolated directory."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        workdir: str | Path | None = None,
        cleanup: bool = True,
    ) -> None:
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
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="agent_vcr_sandbox_")
            self._root_path = Path(self._temp_dir.name).resolve()

        self._alive = True
        logger.debug("LocalSandbox started at %s", self._root_path)

    def stop(self) -> None:
        """Clean up the sandbox workspace."""
        if not self._alive:
            return

        self._alive = False
        if self._temp_dir is not None and self._cleanup:
            try:
                self._temp_dir.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up sandbox temporary directory: %s", exc)
            self._temp_dir = None
            self._root_path = None
        elif self._temp_dir is not None:
            self._root_path = None

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

        # Merge environment
        cmd_env = dict(os.environ)
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
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                exit_code = -1
                error_msg = f"Command timed out after {effective_timeout:.2f}s"
                try:
                    # Kill process group to ensure child processes are terminated
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
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
        """Write a file safely inside the sandbox, creating parent directories."""
        target = self._resolve_safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")

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
                    rel = str(p.relative_to(self.root_path))
                    paths.append(rel)
                except (ValueError, OSError):
                    continue
        return sorted(paths)

    def snapshot_filesystem(self, path: str = ".") -> FilesystemSnapshot:
        """Compute SHA-256 hash, size, and mtime for all files in the sandbox."""
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
                        rel_path = str(file_path.relative_to(self.root_path))
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
            root_path=str(self.root_path),
            timestamp=time.time(),
            entries=entries,
        )
