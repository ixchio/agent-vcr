"""E2B microVM sandbox adapter (Phase 7).

Integrates with the e2b (>=2.46.0) SDK for Firecracker microVM execution.
Falls back gracefully when the SDK is absent or unconfigured.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agent_vcr.sandbox.base import BaseSandbox
from agent_vcr.sandbox.models import (
    ExecutionResult,
    FileMetadata,
    FilesystemSnapshot,
    SandboxConfig,
)

logger = logging.getLogger(__name__)

_E2B_IMPORT_ERROR: ImportError | None = None
try:
    import e2b  # type: ignore[import-not-found]
except ImportError as exc:
    _E2B_IMPORT_ERROR = exc


class E2BSandbox(BaseSandbox):
    """Firecracker microVM sandbox via E2B."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self._api_key = api_key or (self.config.api_key or os.environ.get("E2B_API_KEY"))
        self._injected_client = client
        self._sandbox_instance: Any = None
        self._alive = False

    @property
    def is_alive(self) -> bool:
        return self._alive and self._sandbox_instance is not None

    def start(self) -> None:
        """Provision the remote microVM."""
        if self._alive:
            return

        if self._injected_client is not None:
            self._sandbox_instance = self._injected_client
            self._alive = True
            return

        if _E2B_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The e2b package is not installed. "
                'Install it with: pip install "ai-agent-vcr[sandbox]" or pip install e2b>=2.46.0'
            ) from _E2B_IMPORT_ERROR

        if not self._api_key:
            raise ValueError(
                "E2B API key required. Provide api_key or set the E2B_API_KEY environment variable."
            )

        # Instantiate E2B Sandbox
        try:
            # E2B Sandbox class
            sandbox_cls = getattr(e2b, "Sandbox", None)
            if sandbox_cls is None:
                raise RuntimeError("e2b module does not export Sandbox")

            # Check for Sandbox.create or direct instantiation
            if hasattr(sandbox_cls, "create"):
                self._sandbox_instance = sandbox_cls.create(
                    template=self.config.template,
                    api_key=self._api_key,
                    timeout=int(self.config.timeout_seconds),
                )
            else:
                self._sandbox_instance = sandbox_cls(
                    template=self.config.template,
                    api_key=self._api_key,
                    timeout=int(self.config.timeout_seconds),
                )
            self._alive = True
            logger.info("E2BSandbox microVM provisioned successfully")
        except Exception as exc:
            logger.error("Failed to provision E2B sandbox: %s", exc)
            raise

    def stop(self) -> None:
        """Kill the remote microVM."""
        if not self._alive:
            return

        self._alive = False
        if self._sandbox_instance is not None:
            try:
                if hasattr(self._sandbox_instance, "kill"):
                    self._sandbox_instance.kill()
                elif hasattr(self._sandbox_instance, "close"):
                    self._sandbox_instance.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing E2B microVM: %s", exc)
            self._sandbox_instance = None

    def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute a command inside the E2B microVM."""
        if not self.is_alive:
            self.start()

        effective_timeout = timeout if timeout is not None else self.config.timeout_seconds
        start_time = time.perf_counter()
        timed_out = False
        exit_code = 0
        stdout = ""
        stderr = ""
        error_msg: str | None = None

        cmd_env = dict(self.config.env)
        if env:
            cmd_env.update(env)

        try:
            sb = self._sandbox_instance
            # Support e2b commands.run or commands.exec or direct run
            if hasattr(sb, "commands") and hasattr(sb.commands, "run"):
                res = sb.commands.run(
                    command,
                    timeout=effective_timeout,
                    envs=cmd_env,
                    cwd=workdir or self.config.workdir,
                )
            elif hasattr(sb, "process") and hasattr(sb.process, "start_and_wait"):
                res = sb.process.start_and_wait(command)
            elif hasattr(sb, "run_command"):
                res = sb.run_command(command, timeout=effective_timeout)
            else:
                raise NotImplementedError("Injected sandbox instance missing command runner")

            stdout = getattr(res, "stdout", "") or ""
            stderr = getattr(res, "stderr", "") or ""
            raw_code = getattr(res, "exit_code", None)
            cmd_err = getattr(res, "error", None)
            if raw_code is not None:
                exit_code = int(raw_code)
            elif cmd_err is not None:
                exit_code = -1
            else:
                exit_code = 0

            if cmd_err is not None:
                error_msg = str(cmd_err)
            elif exit_code != 0 and stderr:
                error_msg = stderr.strip()
        except TimeoutError:
            timed_out = True
            exit_code = -1
            error_msg = f"Command timed out after {effective_timeout:.2f}s"
        except Exception as exc:  # noqa: BLE001
            # Check if exception represents timeout
            if "timeout" in str(exc).lower():
                timed_out = True
                exit_code = -1
                error_msg = str(exc)
            else:
                exit_code = -1
                error_msg = str(exc)

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=round(duration_ms, 2),
            timed_out=timed_out,
            error=error_msg,
        )

    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        """Read a file from the microVM."""
        if not self.is_alive:
            self.start()

        sb = self._sandbox_instance
        if hasattr(sb, "files") and hasattr(sb.files, "read"):
            content = sb.files.read(path)
            if binary and isinstance(content, str):
                return content.encode("utf-8")
            if not binary and isinstance(content, bytes):
                return content.decode("utf-8")
            return content
        raise NotImplementedError("Injected sandbox instance missing files.read")

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file to the microVM."""
        if not self.is_alive:
            self.start()

        sb = self._sandbox_instance
        if hasattr(sb, "files") and hasattr(sb.files, "write"):
            sb.files.write(path, content)
        else:
            raise NotImplementedError("Injected sandbox instance missing files.write")

    def list_files(self, path: str = ".") -> list[str]:
        """List files in the microVM."""
        if not self.is_alive:
            self.start()

        sb = self._sandbox_instance
        if hasattr(sb, "files") and hasattr(sb.files, "list"):
            items = sb.files.list(path)
            names: list[str] = []
            for item in items:
                name = getattr(item, "name", str(item))
                names.append(name)
            return sorted(names)

        # Fallback: run find command
        res = self.execute_command(f"find {path} -type f")
        if res.success:
            return sorted([line.strip() for line in res.stdout.splitlines() if line.strip()])
        return []

    def snapshot_filesystem(self, path: str = ".") -> FilesystemSnapshot:
        """Snapshot files in microVM via sha256 checksums."""
        if not self.is_alive:
            self.start()

        entries: dict[str, FileMetadata] = {}
        # Run a bash one-liner in the microVM to get relative path, size, sha256
        script = (
            f"find {path} -type f -not -path '*/.*' -exec sha256sum {{}} + 2>/dev/null"
        )
        res = self.execute_command(script)
        if res.success:
            for line in res.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    h, fpath = parts[0], parts[1]
                    clean_path = fpath
                    if clean_path.startswith("./"):
                        clean_path = clean_path[2:]
                    elif clean_path.startswith("/"):
                        clean_path = clean_path.lstrip("/")
                    entries[clean_path] = FileMetadata(
                        path=clean_path,
                        sha256=h,
                        size=0,
                        mtime=0.0,
                        is_dir=False,
                    )

        return FilesystemSnapshot(
            root_path=path,
            timestamp=time.time(),
            entries=entries,
        )
