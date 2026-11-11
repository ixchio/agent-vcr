"""E2B microVM sandbox adapter (Phase 7).

Integrates with the e2b (>=2.46.0) SDK for Firecracker microVM execution.
Falls back gracefully when the SDK is absent or unconfigured.

Timeouts: E2B's ``Sandbox.create(timeout=...)`` is the microVM *lifetime*,
so it gets ``SandboxConfig.lifetime_seconds``; ``timeout_seconds`` (or the
``timeout`` argument) is applied per command via ``commands.run(timeout=...)``.
A non-zero exit raises ``CommandExitException`` in the SDK; it is converted
into an :class:`ExecutionResult` that keeps stdout, stderr and the exit code.
"""

from __future__ import annotations

import logging
import math
import os
import shlex
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


def _sdk_exception(name: str) -> type[BaseException] | None:
    """Look up an exception class exported by the e2b SDK (None if absent)."""
    if _E2B_IMPORT_ERROR is not None:
        return None
    exc = getattr(e2b, name, None)
    return exc if isinstance(exc, type) and issubclass(exc, BaseException) else None


def _is_command_exit(exc: BaseException) -> bool:
    cls = _sdk_exception("CommandExitException")
    if cls is not None and isinstance(exc, cls):
        return True
    # Duck-typing fallback (SDK moved / faked): carries an exit code + output.
    return type(exc).__name__ == "CommandExitException" or (
        isinstance(getattr(exc, "exit_code", None), int)
        and (hasattr(exc, "stdout") or hasattr(exc, "stderr"))
    )


def _is_timeout(exc: BaseException) -> bool:
    cls = _sdk_exception("TimeoutException")
    if cls is not None and isinstance(exc, cls):
        return True
    return isinstance(exc, TimeoutError) or type(exc).__name__ in (
        "TimeoutException", "TimeoutError", "ReadTimeout", "DeadlineExceeded")


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


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

            # ``timeout`` here is the microVM lifetime, not a command timeout.
            lifetime = int(math.ceil(max(self.config.lifetime_seconds,
                                         self.config.timeout_seconds)))
            create_kwargs: dict[str, Any] = {
                "template": self.config.template,
                "api_key": self._api_key,
                "timeout": lifetime,
            }
            # Check for Sandbox.create or direct instantiation
            if hasattr(sandbox_cls, "create"):
                self._sandbox_instance = sandbox_cls.create(**create_kwargs)
            else:
                self._sandbox_instance = sandbox_cls(**create_kwargs)
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

            stdout, stderr, exit_code, error_msg = self._unpack_result(res)
        except Exception as exc:  # noqa: BLE001
            if _is_command_exit(exc):
                # Non-zero exit: the SDK raises, but the output is real.
                stdout, stderr, exit_code, error_msg = self._unpack_result(exc)
                if exit_code == 0:
                    exit_code = -1
                if not error_msg:
                    error_msg = stderr.strip() or f"Command exited with code {exit_code}"
            elif _is_timeout(exc):
                timed_out = True
                exit_code = -1
                error_msg = f"Command timed out after {effective_timeout:.2f}s: {exc}"
            else:
                exit_code = -1
                error_msg = f"{type(exc).__name__}: {exc}"

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

    @staticmethod
    def _unpack_result(res: Any) -> tuple[str, str, int, str | None]:
        """(stdout, stderr, exit_code, error) from a CommandResult-like object."""
        stdout = _text(getattr(res, "stdout", ""))
        stderr = _text(getattr(res, "stderr", ""))
        raw_code = getattr(res, "exit_code", None)
        cmd_err = getattr(res, "error", None)
        cmd_err = cmd_err if isinstance(cmd_err, str) and cmd_err else None
        if isinstance(raw_code, (int, float)) and not isinstance(raw_code, bool):
            exit_code = int(raw_code)
        elif cmd_err is not None:
            exit_code = -1
        else:
            exit_code = 0
        error_msg: str | None = cmd_err
        if error_msg is None and exit_code != 0 and stderr:
            error_msg = stderr.strip()
        return stdout, stderr, exit_code, error_msg

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
        res = self.execute_command(f"find {shlex.quote(path)} -type f")
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
            f"find {shlex.quote(path)} -type f -not -path '*/.*' -exec sha256sum {{}} + 2>/dev/null"
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
