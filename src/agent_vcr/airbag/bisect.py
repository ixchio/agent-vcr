"""``vcr bisect``: which agent action broke the tests? Zero model tokens.

The airbag already has a snapshot from just before every agent action, so the
project's history *between commits* is a list of states::

    state 0 ──action 0──► state 1 ──action 1──► … ──action n-1──► state n (now)

``state i`` is the snapshot taken before action ``i``; the last state is the
project as it is now. Each state is checked out into a temporary directory
(never the project), the test command runs there, and a binary search finds
the first action whose before-state passes and whose after-state fails.

Read-only for the project and the existing snapshots: the only write to the
store is one snapshot of the current state, exactly like ``vcr undo --dry-run``
takes. Ignored dependency trees (``node_modules``, ``.venv``) are not in
snapshots; ``copy_ignored`` symlinks the project's top-level *dependency*
directories (:data:`DEPENDENCY_DIRS`, plus any the user names) into every
checkout — never build output, which a test run could overwrite.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_vcr.airbag.core import Airbag, describe_action
from agent_vcr.airbag.store import AirbagError, Snapshot, _clean_env

DEFAULT_TIMEOUT = 300.0
OUTPUT_TAIL_LINES = 30
#: Ignored top-level directories ``copy_ignored`` links by default: dependency trees only.
#: The links are shared with the real project, so build output (``dist/``) is never linked.
DEPENDENCY_DIRS = ("node_modules", ".venv", "venv", "env", "vendor", ".tox",
                   "bower_components")


@dataclass
class Step:
    """One recorded event that may have changed files: an agent action or a restore."""

    seq: int  # snapshot taken just before it
    kind: str  # "action" | "restore"
    event: dict[str, Any]

    def describe(self) -> str:
        if self.kind == "restore":
            return f"vcr undo/rewind to snapshot #{self.event.get('to')}"
        return describe_action(self.event)


@dataclass
class TestRun:
    seq: int
    outcome: str  # "pass" | "fail" | "timeout" | "error"
    returncode: int | None
    seconds: float
    output: str = ""
    cached: bool = False

    @property
    def passed(self) -> bool:
        return self.outcome == "pass"


@dataclass
class BisectReport:
    #: "found" | "no-actions" | "good-fails" | "bad-passes" | "invalid"
    status: str
    message: str
    root: str
    command: list[str]
    good: int | None = None
    bad: int | None = None
    current: int | None = None
    runs: list[TestRun] = field(default_factory=list)
    culprit: dict[str, Any] | None = None
    culprit_kind: str | None = None
    before: int | None = None
    after: int | None = None
    after_is_current: bool = False
    changes: list[tuple[str, str]] = field(default_factory=list)
    rewind: str | None = None
    timeouts: int = 0

    @property
    def test_runs(self) -> int:
        return sum(1 for r in self.runs if not r.cached)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["test_runs"] = self.test_runs
        data["changes"] = [{"status": s, "path": p} for s, p in self.changes]
        return data


def timeline(bag: Airbag) -> list[Step]:
    """Agent actions (and undo/rewind restores) in order, each with its before-snapshot.

    Same actions as ``vcr rewind`` lists. Snapshots removed by ``gc`` are skipped.
    """
    known = {s.seq for s in bag.store.snapshots()}
    steps: list[Step] = []
    for event in bag.store.events():
        kind = event.get("kind")
        try:
            if kind == "action" and event.get("snapshot") is not None:
                seq = int(event["snapshot"])
            elif kind == "restore" and event.get("before") is not None:
                seq = int(event["before"])
            else:
                continue
        except (TypeError, ValueError):
            continue
        if seq in known:
            steps.append(Step(seq=seq, kind=str(kind), event=event))
    return steps


def _kill(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows
            proc.kill()


def run_test(argv: list[str], cwd: Path, timeout: float, *, shell: bool = False,
             env_extra: dict[str, str] | None = None) -> tuple[str, int | None, float, str]:
    """Run the test command in ``cwd``: (outcome, returncode, seconds, output tail)."""
    env = _clean_env()  # no GIT_DIR & co. pointing back at the user's repository
    env.update({"PWD": str(cwd), **(env_extra or {})})
    start = time.monotonic()
    # Output goes to a file, not a pipe: a background child the test leaves behind
    # (`server & pytest`) would keep a pipe open and make a passing run look hung.
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(
                argv[0] if shell else argv, shell=shell, cwd=str(cwd), env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
        except OSError as err:
            return "error", None, time.monotonic() - start, f"could not start {argv[0]!r}: {err}"
        try:
            proc.wait(timeout=timeout)
            outcome = "pass" if proc.returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            outcome = "timeout"
        finally:
            # The test itself on timeout, and leftover background processes (servers, watchers).
            _kill(proc)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
            if os.name != "nt":
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
        out.seek(0, os.SEEK_END)
        out.seek(max(0, out.tell() - 65536))
        text = out.read().decode("utf-8", "replace")
    tail = "\n".join(text.rstrip().splitlines()[-OUTPUT_TAIL_LINES:])
    return outcome, proc.returncode, time.monotonic() - start, tail


class _Checkouts:
    """Temporary checkouts of snapshots, one at a time, removed afterwards."""

    def __init__(self, bag: Airbag, copy_ignored: bool, extra: Sequence[str] = ()) -> None:
        self.bag = bag
        self.links: list[str] = []
        if copy_ignored or extra:
            ignored = bag.store.ignored_top_level_dirs()
            self.links = [d for d in ignored if d in DEPENDENCY_DIRS] if copy_ignored else []
            for name in extra:
                if name not in self.links and (bag.root / name).is_dir():
                    self.links.append(name)
        self.base = Path(tempfile.mkdtemp(prefix="vcr-bisect-"))

    @contextlib.contextmanager
    def checkout(self, snap: Snapshot) -> Any:
        # Keep the project's directory name: some tools derive names from it.
        dest = self.base / f"{snap.seq}" / (self.bag.root.name or "project")
        try:
            self.bag.store.materialize(snap, dest)
            for name in self.links:
                link = dest / name
                if not os.path.lexists(link):
                    os.symlink(self.bag.root / name, link, target_is_directory=True)
            yield dest
        finally:
            _remove_tree(dest.parent)

    def close(self) -> None:
        _remove_tree(self.base)


def _remove_tree(path: Path) -> None:
    """rmtree that never follows the dependency symlinks into the real project."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def bisect(
    bag: Airbag, argv: list[str], *, good: int | None = None, bad: int | None = None,
    timeout: float = DEFAULT_TIMEOUT, copy_ignored: bool = False,
    extra_links: Sequence[str] = (),
    progress: Callable[[TestRun, str], None] | None = None,
) -> BisectReport:
    """Find the first recorded action after which ``argv`` fails.

    ``copy_ignored`` links the ignored :data:`DEPENDENCY_DIRS` into each
    checkout; ``extra_links`` names more top-level directories to link.
    """
    for name in extra_links:
        if not name or name in {".", "..", ".git"} or "/" in name or "\\" in name:
            raise ValueError(f"--copy-ignored takes a top-level directory name, not {name!r}")
    shell = len(argv) == 1 and any(ch.isspace() for ch in argv[0])
    report = BisectReport(status="no-actions", message="", root=str(bag.root),
                          command=list(argv))
    steps = timeline(bag)
    if not steps:
        report.message = (f"Nothing to bisect: no recorded agent actions for {bag.root}. "
                          "Install hooks with `vcr airbag install`.")
        return report
    current = bag.store.snapshot(label="before bisect", meta={"agent": "user", "bisect": True})
    report.current = current.seq
    snaps = {s.seq: s for s in bag.store.snapshots()}
    states = [step.seq for step in steps] + [current.seq]
    n = len(steps)

    lo, hi = 0, n
    if good is not None:
        found = [i for i, seq in enumerate(states) if seq == good]
        if not found:
            report.status = "invalid"
            report.message = (f"--good #{good} is not in the timeline (snapshots "
                              f"#{states[0]}…#{states[-1]}; see `vcr rewind`)")
            return report
        lo = found[-1]
    if bad is not None:
        found = [i for i, seq in enumerate(states) if seq == bad]
        if not found:
            report.status = "invalid"
            report.message = (f"--bad #{bad} is not in the timeline (snapshots "
                              f"#{states[0]}…#{states[-1]}; see `vcr rewind`)")
            return report
        hi = found[0]
    if lo >= hi:
        report.status = "invalid"
        report.message = (f"--good #{states[lo]} must come before --bad #{states[hi]} "
                          "in the timeline")
        return report
    report.good, report.bad = states[lo], states[hi]

    cache: dict[str, TestRun] = {}
    checkouts = _Checkouts(bag, copy_ignored, extra_links)

    def test(i: int) -> TestRun:
        snap = snaps[states[i]]
        hit = cache.get(snap.tree)
        if hit is not None:
            run = TestRun(seq=snap.seq, outcome=hit.outcome, returncode=hit.returncode,
                          seconds=0.0, output=hit.output, cached=True)
        else:
            with checkouts.checkout(snap) as cwd:
                outcome, code, secs, out = run_test(
                    argv, cwd, timeout, shell=shell,
                    env_extra={"VCR_BISECT_SNAPSHOT": str(snap.seq)})
            run = TestRun(seq=snap.seq, outcome=outcome, returncode=code, seconds=secs,
                          output=out)
            cache[snap.tree] = run
            if outcome == "timeout":
                report.timeouts += 1
        report.runs.append(run)
        if progress is not None:
            progress(run, "current" if i == n else "")
        return run

    try:
        if test(hi).passed:
            report.status = "bad-passes"
            where = "now" if hi == n else f"at snapshot #{states[hi]}"
            report.message = (f"The test passes {where}, so there is nothing to find. "
                              "Pass --bad <snapshot> for a state where it fails.")
            return report
        if not test(lo).passed:
            report.status = "good-fails"
            report.message = (f"The test already fails at snapshot #{states[lo]}"
                              f"{' (the oldest recorded action)' if lo == 0 else ''}, so the "
                              "break happened before the recorded actions (or the test "
                              "cannot run in a temporary checkout). Try an older --good, "
                              "--copy-ignored, or check the output below.")
            return report
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if test(mid).passed:
                lo = mid
            else:
                hi = mid
    except AirbagError as err:
        report.status = "invalid"
        report.message = f"could not check out a snapshot: {err}"
        return report
    finally:
        checkouts.close()

    step = steps[lo]
    report.status = "found"
    report.culprit = step.event
    report.culprit_kind = step.kind
    report.before, report.after = states[lo], states[hi]
    report.after_is_current = hi == n
    report.changes = bag.store.diff(snaps[states[lo]], snaps[states[hi]])
    report.rewind = f"vcr rewind {states[lo]}"
    report.message = f"First bad {'action' if step.kind == 'action' else 'step'}: " \
                     f"{step.describe()}"
    return report
