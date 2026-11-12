"""Schedules + triggers for workflows (Phase 5).

Schedule kinds (all pure, deterministic, timezone-naive UTC):

- ``interval``: every N seconds from ``last_run`` (or now when never run).
- ``cron``: standard 5-field expressions (minute hour dom month dow)
  supporting ``*``, ``*/n``, ``a/n``, ``a,b``, ``a-b``, ``a-b/n``. As in
  Vixie cron, when **both** day-of-month and day-of-week are restricted
  (neither starts with ``*``) a day matches if *either* field matches.
  Matched with minute precision. **Catch-up:** if a due minute was missed
  since the last run (or since the schedule was created), the schedule is
  due once — never a burst of repeats.
- ``at``: one-shot ISO-8601 datetime; due once, then spent.

ISO timestamps may end in ``Z`` (normalized for Python 3.9/3.10).

:class:`Scheduler` persists schedules + last-run markers in the registry
directory and executes due workflows through a :class:`WorkflowEngine`.
:meth:`Scheduler.tick` is side-effect free (planning);
:meth:`Scheduler.run_pending` executes. Every read-modify-write of the
schedules file happens under an exclusive file lock (``fcntl.flock`` on
POSIX, ``msvcrt.locking`` on Windows, in-process lock otherwise) and
re-reads the file, so concurrent ``put``/``remove`` calls survive and two
scheduler processes never both claim the same due run. Each schedule's
run is *claimed* (``last_run_at`` persisted) before the workflow executes:
a crash mid-run never causes a re-run (at-most-once per due slot).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import IO, Any, Callable, NamedTuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SEARCH_DAYS = 366 * 5 + 2  # next/previous-match horizon: ~5 years (covers leap days)
_LOCK_TIMEOUT_S = 30.0

try:  # POSIX
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    _fcntl = None  # type: ignore[assignment]
_msvcrt: Any
try:  # Windows
    _msvcrt = importlib.import_module("msvcrt")
except ImportError:
    _msvcrt = None


def _aware(moment: datetime) -> datetime:
    """Coerce naive datetimes to UTC (mixed naive/aware arithmetic crashes)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def _parse_iso(value: str) -> datetime:
    """``fromisoformat`` that also accepts a trailing ``Z`` on Python 3.9/3.10."""
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def _coerce_stamp(value: str) -> datetime:
    return _aware(_parse_iso(value))


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    created_at: str = ""  # cron catch-up baseline before the first run


class _Cron(NamedTuple):
    minute: frozenset[int]
    hour: frozenset[int]
    dom: frozenset[int]
    month: frozenset[int]
    dow: frozenset[int]
    dom_star: bool
    dow_star: bool


def _parse_field(field: str, low: int, high: int) -> set[int]:
    """Parse one cron field into matching values (real cron subset)."""
    values: set[int] = set()
    for raw in field.split(","):
        part = raw.strip()
        if not part:
            raise ValueError(f"Empty element in cron field {field!r}")
        step = 1
        stepped = "/" in part
        if stepped:
            part, _, step_s = part.partition("/")
            step = int(step_s)
            if step <= 0:
                raise ValueError(f"Cron step must be positive in {field!r}")
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_s, _, end_s = part.partition("-")
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Cron range {part!r} is reversed in {field!r}")
        else:
            start = int(part)
            end = high if stepped else start  # "a/n" = from a to the max, every n
        if not (low <= start <= high and low <= end <= high):
            raise ValueError(f"Cron field {field!r} out of range {low}-{high}")
        values.update(range(start, end + 1, step))
    return values


def _parse_cron(expr: str) -> _Cron:
    """Parse once; reuse for repeated matching."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression needs 5 fields: {expr!r}")
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    parsed: list[frozenset[int]] = []
    for field, (low, high) in zip(fields, bounds):
        values = _parse_field(field, low, high)
        if low == 0 and high == 7 and 7 in values:
            values.discard(7)
            values.add(0)  # cron: 0 and 7 both mean Sunday
        parsed.append(frozenset(values))
    return _Cron(parsed[0], parsed[1], parsed[2], parsed[3], parsed[4],
                 dom_star=fields[2].startswith("*"), dow_star=fields[4].startswith("*"))


def _day_matches(cron: _Cron, day: date) -> bool:
    if day.month not in cron.month:
        return False
    dom_ok = day.day in cron.dom
    dow_ok = (day.weekday() + 1) % 7 in cron.dow  # Sunday = 0
    if not cron.dom_star and not cron.dow_star:
        return dom_ok or dow_ok  # Vixie cron: restricted DOM *or* DOW
    return dom_ok and dow_ok


def _matches_cron(cron: _Cron, moment: datetime) -> bool:
    return (moment.minute in cron.minute and moment.hour in cron.hour
            and _day_matches(cron, moment.date()))


def cron_matches(expr: str, moment: datetime) -> bool:
    """True when a 5-field cron expression matches ``moment`` (minute precision)."""
    return _matches_cron(_parse_cron(expr), moment)


def _floor_minute(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


def _next_match(cron: _Cron, start: datetime, days: int = _SEARCH_DAYS) -> datetime | None:
    """First matching minute >= ``start`` (minute-floored) within ``days`` days."""
    start = _floor_minute(start)
    hours, minutes = sorted(cron.hour), sorted(cron.minute)
    for offset in range(days + 1):
        day = start.date() + timedelta(days=offset)
        if not _day_matches(cron, day):
            continue
        for hour in hours:
            if offset == 0 and hour < start.hour:
                continue
            for minute in minutes:
                if offset == 0 and hour == start.hour and minute < start.minute:
                    continue
                return datetime(day.year, day.month, day.day, hour, minute,
                                tzinfo=start.tzinfo)
    return None


def _prev_match(cron: _Cron, end: datetime, floor: datetime) -> datetime | None:
    """Latest matching minute in ``(floor, end]`` (both minute-floored), or None."""
    end, floor = _floor_minute(end), _floor_minute(floor)
    if end <= floor:
        return None
    hours, minutes = sorted(cron.hour, reverse=True), sorted(cron.minute, reverse=True)
    span = min((end.date() - floor.date()).days, _SEARCH_DAYS)
    for offset in range(span + 1):
        day = end.date() - timedelta(days=offset)
        if not _day_matches(cron, day):
            continue
        for hour in hours:
            if offset == 0 and hour > end.hour:
                continue
            for minute in minutes:
                if offset == 0 and hour == end.hour and minute > end.minute:
                    continue
                candidate = datetime(day.year, day.month, day.day, hour, minute,
                                     tzinfo=end.tzinfo)
                return candidate if candidate > floor else None
    return None


def _cron_due(schedule: ScheduleSpec, cron: _Cron, moment: datetime) -> bool:
    now = _floor_minute(moment)
    baseline = schedule.last_run_at or schedule.created_at
    if not baseline:
        return _matches_cron(cron, now)  # never run, unknown creation: exact minute only
    last = _floor_minute(_coerce_stamp(baseline))
    if schedule.last_run_at:
        # a run in minute M covers every due slot <= M
        return _prev_match(cron, now, last) is not None
    # created during minute M: slot M itself still counts if not yet run
    return _prev_match(cron, now, last - timedelta(minutes=1)) is not None


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
        return _cron_due(schedule, _parse_cron(schedule.cron), moment)
    if schedule.kind == "at":
        if not schedule.at or schedule.last_run_at:
            return False
        return moment >= _coerce_stamp(schedule.at)
    raise ValueError(f"Unknown schedule kind {schedule.kind!r}")


class Scheduler:
    """Persistent schedules + due execution through a workflow engine."""

    _thread_locks: dict[str, threading.Lock] = {}
    _thread_locks_guard = threading.Lock()

    def __init__(self, directory: str | Path = ".vcr/schedules",
                 lock_timeout_s: float = _LOCK_TIMEOUT_S):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock_timeout_s = lock_timeout_s

    def _path(self) -> Path:
        return self.dir / "schedules.json"

    # -- locking + atomic persistence --------------------------------------

    def _thread_lock(self) -> threading.Lock:
        key = str(self._path().resolve())
        with Scheduler._thread_locks_guard:
            return Scheduler._thread_locks.setdefault(key, threading.Lock())

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive cross-process (and cross-thread) lock on the schedules file."""
        deadline = time.monotonic() + self.lock_timeout_s
        thread_lock = self._thread_lock()
        if not thread_lock.acquire(timeout=self.lock_timeout_s):
            raise TimeoutError(f"Timed out waiting for schedule lock in {self.dir}")
        try:
            with open(self.dir / "schedules.lock", "a+", encoding="utf-8") as handle:
                self._acquire_file_lock(handle, deadline)
                try:
                    yield
                finally:
                    self._release_file_lock(handle)
        finally:
            thread_lock.release()

    def _acquire_file_lock(self, handle: IO[str], deadline: float) -> None:
        while True:
            try:
                if _fcntl is not None:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                elif _msvcrt is not None:  # pragma: no cover - Windows
                    handle.seek(0)
                    _msvcrt.locking(handle.fileno(), _msvcrt.LK_NBLCK, 1)
                return  # no OS lock available: the in-process lock is the fallback
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for schedule lock in {self.dir}") from None
                time.sleep(0.05)

    @staticmethod
    def _release_file_lock(handle: IO[str]) -> None:
        with contextlib.suppress(OSError):
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
            elif _msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)

    def _load(self) -> dict[str, ScheduleSpec]:
        if not self._path().exists():
            return {}
        try:
            raw = json.loads(self._path().read_text(encoding="utf-8"))
        except ValueError as err:
            raise ValueError(f"Corrupt schedules file: {err}") from err
        return {name: ScheduleSpec(**spec) for name, spec in raw.items()}

    def _save(self, schedules: dict[str, ScheduleSpec]) -> None:
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({k: v.model_dump() for k, v in schedules.items()}, f, indent=2)
            os.replace(tmp, str(self._path()))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- CRUD ------------------------------------------------------------------

    def put(self, schedule: ScheduleSpec) -> ScheduleSpec:
        """Create or replace a schedule (validates eagerly)."""
        if schedule.kind == "cron":
            _parse_cron(schedule.cron)  # validate syntax
        elif schedule.kind == "at":
            _parse_iso(schedule.at)  # validate format
        elif schedule.kind != "interval":
            raise ValueError(f"Unknown schedule kind {schedule.kind!r}")
        if schedule.last_run_at:
            _parse_iso(schedule.last_run_at)
        if not schedule.created_at:
            schedule = schedule.model_copy(update={"created_at": _utcnow()})
        with self._locked():
            schedules = self._load()
            schedules[schedule.name] = schedule
            self._save(schedules)
        return schedule

    def remove(self, name: str) -> bool:
        with self._locked():
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

    # -- execution ---------------------------------------------------------------

    def _claim(self, name: str, moment: datetime) -> ScheduleSpec | None:
        """Atomically re-check one schedule and mark it as running (None = not ours)."""
        with self._locked():
            schedules = self._load()
            current = schedules.get(name)
            if current is None or not schedule_due(current, moment):
                return None  # removed, disabled, or already claimed by another process
            current.last_run_at = moment.isoformat()
            current.last_status = "running"
            self._save(schedules)
            return current

    def _finish(self, name: str, claimed_at: str, status: str) -> None:
        """Record the outcome, merged into the *current* file (concurrent edits survive)."""
        with self._locked():
            schedules = self._load()
            current = schedules.get(name)
            if current is None or current.last_run_at != claimed_at:
                return  # removed or replaced while running: keep the newer state
            current.last_status = status
            self._save(schedules)

    def run_pending(
        self,
        engine: Any,
        get_workflow: Callable[[str], Any],
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Execute all due schedules; each run is claimed before it executes."""
        moment = _aware(now or datetime.now(timezone.utc))
        results: list[dict[str, Any]] = []
        for candidate in self.tick(moment):
            schedule = self._claim(candidate.name, moment)
            if schedule is None:
                continue
            try:
                workflow = get_workflow(schedule.workflow)
                record = engine.start(workflow, dict(schedule.inputs), tags=["schedule"])
                status = record.status.value
                results.append({"schedule": schedule.name, "run_id": record.run_id,
                                "status": status})
            except Exception as err:  # noqa: BLE001 — one schedule never blocks others
                logger.warning("Scheduled workflow %s failed: %s", schedule.name, err)
                results.append({"schedule": schedule.name, "error": str(err)})
                status = "error"
            self._finish(schedule.name, schedule.last_run_at, status)
        return results


def next_due_in(schedule: ScheduleSpec, now: datetime) -> float | None:
    """Seconds until the next due time (None when disabled/spent/never matching)."""
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
        if not schedule.cron:
            return None
        cron = _parse_cron(schedule.cron)
        if _cron_due(schedule, cron, moment):
            return 0.0
        nxt = _next_match(cron, _floor_minute(moment) + timedelta(minutes=1))
        return None if nxt is None else max(0.0, (nxt - moment).total_seconds())
    raise ValueError(f"Unknown schedule kind {schedule.kind!r}")
