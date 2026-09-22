"""Sandbox models for isolated microVM and container execution (Phase 7)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of a command executed in a sandbox."""

    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if command exited with code 0 and did not time out."""
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "error": self.error,
            "success": self.success,
        }


class FileMetadata(BaseModel):
    """Metadata for a file within a sandbox snapshot."""

    path: str  # relative path
    size: int = 0
    sha256: str = ""
    mtime: float = 0.0
    is_dir: bool = False


class FilesystemDiff(BaseModel):
    """Diff between two filesystem snapshots."""

    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": sorted(self.added),
            "modified": sorted(self.modified),
            "deleted": sorted(self.deleted),
            "has_changes": self.has_changes,
        }


class FilesystemSnapshot(BaseModel):
    """Point-in-time snapshot of the sandbox filesystem."""

    root_path: str = ""
    timestamp: float = 0.0
    entries: dict[str, FileMetadata] = Field(default_factory=dict)

    def diff(self, other: FilesystemSnapshot) -> FilesystemDiff:
        """Compute changes from this snapshot (before) to other snapshot (after)."""
        before_paths = set(self.entries.keys())
        after_paths = set(other.entries.keys())

        added = sorted([p for p in (after_paths - before_paths) if not other.entries[p].is_dir])
        deleted = sorted([p for p in (before_paths - after_paths) if not self.entries[p].is_dir])

        modified: list[str] = []
        common = before_paths & after_paths
        for p in sorted(common):
            b = self.entries[p]
            a = other.entries[p]
            if not a.is_dir and not b.is_dir and (a.sha256 != b.sha256 or a.size != b.size):
                modified.append(p)

        return FilesystemDiff(added=added, modified=modified, deleted=deleted)


class SandboxConfig(BaseModel):
    """Configuration for provisioning an isolated sandbox."""

    provider: Literal["auto", "local", "e2b", "mock"] = "auto"
    timeout_seconds: float = 60.0
    memory_limit_mb: int = 512
    cpu_count: int = 1
    workdir: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    api_key: str | None = None
    template: str = "base"
    network_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
