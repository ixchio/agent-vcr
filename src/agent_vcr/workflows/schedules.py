"""Schedules + triggers for workflows (Phase 5).

Schedule kinds (all pure, deterministic, timezone-naive UTC):

- ``interval``: every N seconds from ``last_run`` (or now when never run).
- ``cron``: standard 5-field expressions (minute hour dom month dow)
  supporting ``*``, ``*/n``, ``a,b``, ``a-b``. Matched with minute
  precision: due when the current minute matches and no run happened
  in this minute yet.
- ``at``: one-shot ISO-8601 datetime; due once, then spent.

:class:`Scheduler` persists schedules + last-run markers in the registry
directory and executes due workflows through a :class:`WorkflowEngine`.
: meth:`tick` is side-effect free (planning); :meth:`run_pending`
executes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _aware(moment: datetime) -> datetime:
    """Coerce naive datetimes to UTC (mixed naive/aware arithmetic crashes)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _coerce_stamp(value: str) -> datetime:
    return _aware(datetime.fromisoformat(value))


class ScheduleSpec(BaseModel):
    """One schedule attached to a workflow."""

    name: str
    workflow: str
    kind: str = "interval"  # interval | cron | at
    interval_s: float = 3600.0
    cron: str = ""
    at: str = ""  # ISO datetime, one-shot
    enabled: bool = True
    inputs: dict[str, Any] = Field(default_factory=dict)
    last_run_at: str = ""
    last_status: str = ""


def _parse_field(field: str, low: int, high: int) -> set[int]:
    """Parse one cron field into matching values (real cron subset)."""
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, _, step_s = part.partition("/")
            step = int(step_s)
        if part == "*":
            values.update(range(low, high + 1, step))
        elif "-" in part:
            start_s, _, end_s = part.partition("-")
            values.update(range(int(start_s), int(end_s) + 1, step))
        elif part:
            values.add(int(part))
    if not all(low <= v <= high for v in values):
        raise ValueError(f"Cron field {field!r} out of range {low}-{high}")
    return values


def _parse_cron(expr: str) -> tuple[frozenset[int], ...]:
    """Parse once; reuse for repeated matching (next_due_in scans)."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression needs 5 fields: {expr!r}")
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    parsed: list[frozenset[int]] = []
    for field, (low, high) in zip(fields, bounds):
        values = set(_parse_field(field, low, high))
        if low == 0 and high == 7 and 7 in values:
            values.add(0)  # cron: 0 and 7 both mean Sunday
        parsed.append(frozenset(values))
    return tuple(parsed)


def _matches_cron(parsed: tuple[frozenset[int], ...], moment: datetime) -> bool:
    minute, hour, dom, month, dow = parsed
    sunday = (moment.weekday() + 1) % 7
    return (moment.minute in minute and moment.hour in hour
            and moment.day in dom and moment.month in month and sunday in dow)


def cron_matches(expr: str, moment: datetime) -> bool:
    """True when a 5-field cron expression matches ``moment`` (minute precision)."""
    return _matches_cron(_parse_cron(expr), moment)


def schedule_due(schedule: ScheduleSpec, now: datetime) -> bool:
    """Pure due-check used by :meth:`Scheduler.tick` (testable, no I/O)."""
    moment = _aware(now)
    if not schedule.enabled:
        return False
    if schedule.kind == "interval":
        if not schedule.last_run_at:
            return True
        last = _coerce_stamp(schedule.last_run_at)
        return (moment - last).total_seconds() >= schedule.interval_s
    if schedule.kind == "cron":
        if not schedule.cron:
            return False
        if not cron_matches(schedule.cron, moment):
            return False
        if schedule.last_run_at:
            last = _coerce_stamp(schedule.last_run_at)
            # one run per matching minute at most
            if last.replace(second=0, microsecond=0) == moment.replace(second=0, microsecond=0):
                return False
        return True
    if schedule.kind == "at":
        if not schedule.at or schedule.last_run_at:
            return False
        return moment >= _coerce_stamp(schedule.at)
    raise ValueError(f"Unknown schedule kind {schedule.kind!r}")


class Scheduler:
    """Persistent schedules + due execution through a workflow engine."""

    def __init__(self, directory: str | Path = ".vcr/schedules"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.dir / "schedules.json"

    def _load(self) -> dict[str, ScheduleSpec]:
        if not self._path().exists():
            return {}
        try:
            raw = json.loads(self._path().read_text())
        except ValueError as err:
            raise ValueError(f"Corrupt schedules file: {err}") from err
        return {name: ScheduleSpec(**spec) for name, spec in raw.items()}

    def _save(self, schedules: dict[str, ScheduleSpec]) -> None:
        import contextlib
        import os
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({k: v.model_dump() for k, v in schedules.items()}, f, indent=2)
            os.replace(tmp, str(self._path()))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def put(self, schedule: ScheduleSpec) -> ScheduleSpec:
        """Create or replace a schedule (validates eagerly)."""
        if schedule.kind == "cron":
            cron_matches(schedule.cron, datetime(2026, 1, 1, 12, 0))  # validate syntax
        elif schedule.kind == "at":
            datetime.fromisoformat(schedule.at)  # validate format
        elif schedule.kind != "interval":
            raise ValueError(f"Unknown schedule kind {schedule.kind!r}")
        schedules = self._load()
        schedules[schedule.name] = schedule
        self._save(schedules)
        return schedule

    def remove(self, name: str) -> bool:
        schedules = self._load()
        if name not in schedules:
            return False
        del schedules[name]
        self._save(schedules)
        return True

    def all(self) -> list[ScheduleSpec]:
        """All schedules (named `all` — `list` would shadow the builtin)."""
        return list(self._load().values())

    def tick(self, now: datetime | None = None) -> list[ScheduleSpec]:
        """Due schedules right now (no side effects)."""
        moment = _aware(now or datetime.now(timezone.utc))
        return [s for s in self._load().values() if schedule_due(s, moment)]

    def run_pending(
        self,
        engine: Any,
        get_workflow: Callable[[str], Any],
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Execute all due schedules; records last-run markers."""
        moment = _aware(now or datetime.now(timezone.utc))
        results: list[dict[str, Any]] = []
        schedules = self._load()
        # NOTE: filter these same instances (not tick()'s fresh copies)
        # so last-run markers persist in the final _save below.
        due = [s for s in schedules.values() if schedule_due(s, moment)]
        for schedule in due:
            try:
                workflow = get_workflow(schedule.workflow)
                record = engine.start(workflow, dict(schedule.inputs), tags=["schedule"])
                results.append({"schedule": schedule.name, "run_id": record.run_id,
                                "status": record.status.value})
                schedule.last_run_at = moment.isoformat()
                schedule.last_status = record.status.value
            except Exception as err:  # noqa: BLE001 — one schedule never blocks others
                logger.warning("Scheduled workflow %s failed: %s", schedule.name, err)
                results.append({"schedule": schedule.name, "error": str(err)})
                schedule.last_run_at = moment.isoformat()
                schedule.last_status = "error"
        self._save(schedules)
        return results


def next_due_in(schedule: ScheduleSpec, now: datetime) -> float | None:
    """Seconds until the next due time (None when disabled/spent)."""
    moment = _aware(now)
    if not schedule.enabled:
        return None
    if schedule.kind == "interval":
        if not schedule.last_run_at:
            return 0.0
        last = _coerce_stamp(schedule.last_run_at)
        return max(0.0, schedule.interval_s - (moment - last).total_seconds())
    if schedule.kind == "at":
        if schedule.last_run_at:
            return None
        return max(0.0, (_coerce_stamp(schedule.at) - moment).total_seconds())
    if schedule.kind == "cron":
        parsed = _parse_cron(schedule.cron)
        probe = moment.replace(second=0, microsecond=0)
        if schedule.last_run_at:
            last = _coerce_stamp(schedule.last_run_at)
            if last.replace(second=0, microsecond=0) == probe and _matches_cron(parsed, probe):
                probe += timedelta(minutes=1)
        for _ in range(525600 + 60):  # scan up to ~1 year, minute by minute
            if _matches_cron(parsed, probe):
                return max(0.0, (probe - moment).total_seconds())
            probe += timedelta(minutes=1)
        return None
    raise ValueError(f"Unknown schedule kind {schedule.kind!r}")
