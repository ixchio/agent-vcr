"""Base abstract interface for sandboxed execution (Phase 7)."""

from __future__ import annotations

import abc
from typing import Any, TypeVar

from agent_vcr.sandbox.models import (
    ExecutionResult,
    FilesystemDiff,
    FilesystemSnapshot,
    SandboxConfig,
)

Self = TypeVar("Self", bound="BaseSandbox")


class BaseSandbox(abc.ABC):
    """Abstract base class for all sandbox execution backends."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._started = False

    @abc.abstractmethod
    def start(self) -> None:
        """Provision and start the sandbox environment."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Tear down and clean up sandbox resources."""

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    @property
    @abc.abstractmethod
    def is_alive(self) -> bool:
        """True if the sandbox is active and able to execute commands."""

    @abc.abstractmethod
    def execute_command(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> ExecutionResult:
        """Execute a shell command inside the sandbox."""

    @abc.abstractmethod
    def read_file(self, path: str, binary: bool = False) -> str | bytes:
        """Read a file from within the sandbox."""

    @abc.abstractmethod
    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file inside the sandbox."""

    @abc.abstractmethod
    def list_files(self, path: str = ".") -> list[str]:
        """List relative file paths inside the sandbox directory."""

    @abc.abstractmethod
    def snapshot_filesystem(self, path: str = ".") -> FilesystemSnapshot:
        """Capture a snapshot of the sandbox filesystem."""

    def diff_filesystem(
        self, before: FilesystemSnapshot, after: FilesystemSnapshot | None = None
    ) -> FilesystemDiff:
        """Compute differences between snapshots.

        If `after` is omitted, captures a new current snapshot.
        """
        if after is None:
            after = self.snapshot_filesystem(path=before.root_path or ".")
        return before.diff(after)

    def record_execution(
        self,
        recorder: Any,
        command: str,
        result: ExecutionResult,
        diff: FilesystemDiff | None = None,
        node_name: str = "sandbox:command",
    ) -> None:
        """Record the sandbox command execution and optional filesystem diff into VCR.

        Records a tool step with execution details, timing, exit code, and fs changes.
        """
        if recorder is None:
            return

        input_state = {
            "command": command,
            "provider": self.config.provider,
            "workdir": self.config.workdir,
        }
        output_state = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
        if diff is not None:
            output_state["filesystem_diff"] = diff.to_dict()

        # Check if recorder has record_tool_call or record_step
        err_msg = result.error if not result.success else None
        if hasattr(recorder, "record_tool_call"):
            recorder.record_tool_call(
                tool_name="sandbox_exec",
                tool_input=input_state,
                tool_output=output_state,
                latency_ms=result.duration_ms,
                error=err_msg,
            )
        elif hasattr(recorder, "record_step"):
            from agent_vcr.models import FrameMetadata

            recorder.record_step(
                node_name=node_name,
                input_state=input_state,
                output_state=output_state,
                metadata=FrameMetadata(
                    latency_ms=float(result.duration_ms),
                    error_message=err_msg,
                    error_type="SandboxError" if err_msg else None,
                    custom={"tool_name": "sandbox_exec"},
                ),
            )

    # -- Context manager support ------------------------------------------

    def __enter__(self: Self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def __aenter__(self: Self) -> Self:
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
