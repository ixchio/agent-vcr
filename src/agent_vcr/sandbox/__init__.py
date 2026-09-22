"""Sandbox runtime package for isolated execution and filesystem diffs (Phase 7)."""

from agent_vcr.sandbox.base import BaseSandbox
from agent_vcr.sandbox.e2b_adapter import E2BSandbox
from agent_vcr.sandbox.factory import create_sandbox
from agent_vcr.sandbox.local import LocalSandbox
from agent_vcr.sandbox.models import (
    ExecutionResult,
    FileMetadata,
    FilesystemDiff,
    FilesystemSnapshot,
    SandboxConfig,
)

__all__ = [
    "BaseSandbox",
    "LocalSandbox",
    "E2BSandbox",
    "create_sandbox",
    "ExecutionResult",
    "FileMetadata",
    "FilesystemDiff",
    "FilesystemSnapshot",
    "SandboxConfig",
]
