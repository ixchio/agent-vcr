"""Workflow registry + durable run store (Phase 5).

- :class:`WorkflowRegistry`: versioned definitions at
  ``<dir>/<name>.v<version>.json`` + ``<name>.latest`` pointer. Saving an
  unchanged definition is a no-op (same version); any change bumps.
- :class:`RunStore`: one atomic JSON file per run
  (``<dir>/<run-id>.json``) embedding the full definition snapshot the
  run executes — the reproducibility contract. Writers persist AFTER
  every top-level step, so a new engine on the same directory resumes
  exactly where the old one died (durability granularity: top-level step).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_vcr.workflows.models import RunRecord, RunStatus, WorkflowDef

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class WorkflowRegistry:
    """Versioned workflow definition store."""

    def __init__(self, directory: str | Path = ".vcr/workflows"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, definition: WorkflowDef) -> WorkflowDef:
        """Store a definition; bumps version only when content changed."""
        latest = self.latest(definition.name)
        if latest is not None and latest.model_dump(exclude={"version"}) == \
                definition.model_dump(exclude={"version"}):
            return latest
        version = (latest.version + 1) if latest is not None else 1
        stored = definition.model_copy(update={"version": version})
        _atomic_json(self.dir / f"{definition.name}.v{version}.json", stored.model_dump())
        _atomic_json(self.dir / f"{definition.name}.latest", {"version": version})
        logger.info("Workflow %s saved at v%d", definition.name, version)
        return stored

    def get(self, name: str, version: int | None = None) -> WorkflowDef:
        if version is None:
            latest = self.latest(name)
            if latest is None:
                raise KeyError(f"Workflow {name!r} not found")
            return latest
        path = self.dir / f"{name}.v{version}.json"
        if not path.exists():
            raise KeyError(f"Workflow {name!r} v{version} not found")
        return WorkflowDef(**json.loads(path.read_text()))

    def latest(self, name: str) -> WorkflowDef | None:
        pointer = self.dir / f"{name}.latest"
        if not pointer.exists():
            return None
        try:
            version = json.loads(pointer.read_text())["version"]
        except (ValueError, KeyError, OSError):
            return None
        return self.get(name, version)

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pointer in sorted(self.dir.glob("*.latest")):
            definition = self.latest(pointer.stem)
            if definition is not None:
                out.append({"name": definition.name, "version": definition.version,
                            "description": definition.description,
                            "steps": len(definition.steps)})
        return out


class RunStore:
    """Durable per-run state (crash-recovery contract)."""

    def __init__(self, directory: str | Path = ".vcr/workflow_runs"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def create(self, definition: WorkflowDef, inputs: dict[str, Any] | None = None,
               run_id: str | None = None, session_id: str = "") -> RunRecord:
        record = RunRecord(
            run_id=run_id or uuid.uuid4().hex[:12],
            workflow=definition.name,
            workflow_version=definition.version,
            definition=definition,
            inputs=inputs or {},
            session_id=session_id,
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        self.persist(record)
        return record

    def load(self, run_id: str) -> RunRecord:
        path = self.dir / f"{run_id}.json"
        if not path.exists():
            raise KeyError(f"Run {run_id!r} not found")
        try:
            return RunRecord(**json.loads(path.read_text()))
        except ValueError as err:
            raise ValueError(f"Corrupt run file {run_id!r}: {err}") from err

    def persist(self, record: RunRecord) -> None:
        record.updated_at = _utcnow()
        _atomic_json(self.dir / f"{record.run_id}.json", record.model_dump())

    def list(self, workflow: str | None = None, status: RunStatus | None = None,
             limit: int = 100) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                record = RunRecord(**json.loads(path.read_text()))
            except ValueError:
                continue
            if workflow and record.workflow != workflow:
                continue
            if status and record.status != status:
                continue
            out.append({"run_id": record.run_id, "workflow": record.workflow,
                        "version": record.workflow_version, "status": record.status.value,
                        "updated_at": record.updated_at})
            if len(out) >= limit:
                break
        return out
