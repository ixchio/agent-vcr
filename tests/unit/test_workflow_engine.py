"""Regression tests for workflow engine / scheduler / backend fixes (review 2026-11-11 §4).

Real in-memory tools and agents, no mocks, no network. The only real
sleeps are sub-second timeout tests.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.superagent import BudgetTracker, CallbackApprover, Policy, ToolGateway
from agent_vcr.workflows import (
    RunStatus,
    Scheduler,
    ScheduleSpec,
    WorkflowDef,
    WorkflowEngine,
    WorkflowRegistry,
    WorkflowStep,
    cron_matches,
    next_due_in,
    schedule_due,
)
from agent_vcr.workflows.backend import (
    InngestWorkflowBackend,
    LocalWorkflowBackend,
    create_workflow_backend,
)
from agent_vcr.workflows.store import RunStore


def _gateway(log: list[str] | None = None) -> ToolGateway:
    gw = ToolGateway()

    @gw.tool("echo", parameters={"type": "object"})
    def echo(value: Any = None) -> dict:
        if log is not None:
            log.append(f"echo:{value}")
        return {"value": value}

    @gw.tool("undo", parameters={"type": "object"})
    def undo(value: Any = None) -> dict:
        if log is not None:
            log.append(f"undo:{value}")
        return {"undone": value}

    @gw.tool("boom", parameters={"type": "object"})
    def boom() -> dict:
        raise RuntimeError("kaput")

    @gw.tool("metered", parameters={"type": "object"})
    def metered() -> dict:
        return {"output": {"ok": True}, "cost_usd": 0.5}

    return gw


def _engine(tmp_path: Path, gw: ToolGateway | None = None, **kwargs: Any) -> WorkflowEngine:
    kwargs.setdefault("approver", CallbackApprover(lambda req: True, name="yes"))
    return WorkflowEngine(gw or _gateway(), run_store=RunStore(tmp_path / "runs"),
                          vcr_dir=str(tmp_path / "vcr"), **kwargs)


def _def(steps: list[dict[str, Any]], name: str = "wf", **kwargs: Any) -> WorkflowDef:
    return WorkflowDef(name=name, steps=[WorkflowStep(**s) for s in steps], **kwargs)


# -- HIGH 1: compensation goes through policy / approval / timeout ----------------


class TestCompensationGuarded:
    def test_forbidden_compensation_is_skipped_not_executed(self, tmp_path: Path) -> None:
        log: list[str] = []
        eng = _engine(tmp_path, _gateway(log), policy=Policy(forbid_tools=["undo"]))
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": "A"},
                 "compensate": {"ref": "undo", "inputs": {"value": "A"}}},
                {"id": "b", "ref": "boom", "depends_on": ["a"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert log == ["echo:A"]  # undo never ran
        assert rec.compensated is False
        assert "compensation a skipped" in rec.error and "forbidden" in rec.error

    def test_compensation_needing_approval_denied_by_default_approver(
            self, tmp_path: Path) -> None:
        log: list[str] = []
        eng = WorkflowEngine(_gateway(log), run_store=RunStore(tmp_path / "runs"),
                             vcr_dir=str(tmp_path / "vcr"),
                             policy=Policy(irreversible_refs=["undo"]))  # AutoApprover()
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": 1},
                 "compensate": {"ref": "undo"}},
                {"id": "b", "ref": "boom", "depends_on": ["a"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert "undo:None" not in log
        assert "approval denied" in rec.error

    def test_hanging_agent_compensation_times_out(self, tmp_path: Path) -> None:
        release = threading.Event()
        eng = _engine(tmp_path, agents={
            "work": lambda inputs: {"ok": True},
            "hang": lambda inputs: release.wait(10) and {},
        })
        try:
            start = time.perf_counter()
            rec = eng.start(_def([
                {"id": "a", "kind": "agent", "ref": "work", "timeout_s": 0.3,
                 "compensate": {"ref": "hang", "kind": "agent"}},
                {"id": "b", "ref": "boom", "depends_on": ["a"]},
            ]))
            elapsed = time.perf_counter() - start
        finally:
            release.set()
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert elapsed < 3.0
        assert rec.compensated is False and "may still be running" in rec.error


# -- HIGH 2: $inputs ------------------------------------------------------------------


class TestRunInputs:
    def test_inputs_reference_resolves(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        wf = _def([{"id": "a", "ref": "echo", "inputs": {"value": "$inputs.x"}},
                   {"id": "b", "ref": "echo", "inputs": {"value": "$inputs.nested.k"},
                    "depends_on": ["a"]}],
                  triggers=[{"type": "webhook", "secret": "s3cr3t"}])
        try:
            rec = eng.start(wf, {"x": 1, "nested": {"k": "v"}})
            assert rec.status == RunStatus.COMMITTED, rec.error
            assert rec.outputs["a"] == {"value": 1} and rec.outputs["b"] == {"value": "v"}
            via_hook = eng.trigger(wf, {"x": 2, "nested": {"k": "w"}}, secret="s3cr3t")
            assert via_hook.outputs["a"] == {"value": 2}
            missing = eng.start(wf, {})
            assert missing.status == RunStatus.ABORTED and "$inputs.x" in missing.error
        finally:
            eng.shutdown()

    def test_schedule_inputs_reach_the_run(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        wf = _def([{"id": "a", "ref": "echo", "inputs": {"value": "$inputs.who"}}], name="s")
        try:
            sched = Scheduler(tmp_path / "sched")
            sched.put(ScheduleSpec(name="hourly", workflow="s", kind="interval",
                                   inputs={"who": "cron"}))
            results = sched.run_pending(eng, lambda name: wf)
            assert results[0]["status"] == "committed"
            assert eng.runs.load(results[0]["run_id"]).outputs["a"] == {"value": "cron"}
        finally:
            eng.shutdown()


# -- HIGH 3: timeouts are not blindly retried ---------------------------------------


class TestTimeoutRetry:
    def _slow_gateway(self, calls: list[int]) -> ToolGateway:
        gw = ToolGateway()

        @gw.tool("slow", parameters={"type": "object"})
        def slow() -> dict:
            calls.append(1)
            time.sleep(0.5)
            return {}

        return gw

    def test_non_idempotent_timeout_runs_once(self, tmp_path: Path) -> None:
        calls: list[int] = []
        eng = _engine(tmp_path, self._slow_gateway(calls))
        try:
            rec = eng.start(_def([{"id": "s", "ref": "slow", "timeout_s": 0.2,
                                   "max_retries": 2}]))
            time.sleep(0.6)  # let the abandoned attempt finish
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert len(calls) == 1
        assert "may still be running" in rec.error and "not declared idempotent" in rec.error

    def test_idempotent_timeout_is_retried(self, tmp_path: Path) -> None:
        calls: list[int] = []
        eng = _engine(tmp_path, self._slow_gateway(calls))
        try:
            rec = eng.start(_def([{"id": "s", "ref": "slow", "timeout_s": 0.2,
                                   "max_retries": 1, "idempotent": True}]))
            time.sleep(0.6)
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert len(calls) == 2


# -- HIGH 4: parallel branches do not starve the leaf pool --------------------------


class TestParallelPools:
    def test_branches_with_agents_on_tiny_pool(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path, agents={"quick": lambda inputs: {"ok": True}}, max_workers=1)
        try:
            rec = eng.start(_def([{"id": "p", "kind": "parallel", "branches": {
                "l": [{"id": "x", "kind": "agent", "ref": "quick", "timeout_s": 1.0}],
                "r": [{"id": "y", "kind": "agent", "ref": "quick", "timeout_s": 1.0}],
                "m": [{"id": "z", "kind": "agent", "ref": "quick", "timeout_s": 1.0}],
            }}]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED, rec.error
        assert set(rec.outputs["p"]) == {"l.x", "r.y", "m.z"}

    def test_nested_parallel_does_not_deadlock(self, tmp_path: Path) -> None:
        inner = {"id": "inner", "kind": "parallel", "branches": {
            "a": [{"id": "e1", "ref": "echo", "inputs": {"value": 1}}],
            "b": [{"id": "e2", "kind": "agent", "ref": "quick"}],
        }}
        eng = _engine(tmp_path, agents={"quick": lambda inputs: {"ok": True}}, max_workers=1)
        try:
            rec = eng.start(_def([{"id": "outer", "kind": "parallel", "branches": {
                "x": [inner], "y": [dict(inner)],
            }}]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED, rec.error
        assert rec.outputs["outer.x.inner"] == {"a.e1": {"value": 1}, "b.e2": {"ok": True}}


# -- MEDIUM 5: per-run budget ---------------------------------------------------------


class TestPerRunBudget:
    def test_workflow_cap_does_not_leak(self, tmp_path: Path) -> None:
        template = BudgetTracker()
        eng = _engine(tmp_path, budget=template)
        two = [{"id": "a", "ref": "metered"}, {"id": "b", "ref": "metered", "depends_on": ["a"]}]
        try:
            capped = eng.start(_def(two, name="capped", max_total_cost_usd=0.4))
            free = eng.start(_def(two, name="free"))
        finally:
            eng.shutdown()
        assert capped.status == RunStatus.ABORTED and "budget" in capped.error
        assert free.status == RunStatus.COMMITTED, free.error
        assert template.max_total_cost_usd is None and template.cost_usd == 0.0
        assert free.usage["cost_usd"] == pytest.approx(1.0)

    def test_concurrent_runs_do_not_reset_each_other(self, tmp_path: Path) -> None:
        gate = threading.Barrier(2)
        gw = ToolGateway()

        @gw.tool("pricey", parameters={"type": "object"})
        def pricey() -> dict:
            gate.wait(timeout=5)
            return {"output": {"ok": True}, "cost_usd": 0.6}

        eng = _engine(tmp_path, gw, budget=BudgetTracker(max_total_cost_usd=1.0))
        wf = _def([{"id": "a", "ref": "pricey"},
                   {"id": "b", "ref": "pricey", "depends_on": ["a"]}])
        results: list[Any] = []
        threads = [threading.Thread(target=lambda: results.append(eng.start(wf)))
                   for _ in range(2)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
        finally:
            eng.shutdown()
        # each run is independently at 0.6 after step a (< 1.0 cap) → b runs → 1.2 total
        assert [r.status for r in results] == [RunStatus.COMMITTED] * 2
        assert all(r.usage["cost_usd"] == pytest.approx(1.2) for r in results)


# -- MEDIUM 6: step errors abort cleanly / definitions validated up front ------------


class TestErrorsAbortCleanly:
    def test_foreach_over_non_list_aborts_with_compensation(self, tmp_path: Path) -> None:
        log: list[str] = []
        eng = _engine(tmp_path, _gateway(log))
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": "scalar"},
                 "compensate": {"ref": "undo", "inputs": {"value": "A"}}},
                {"id": "loop", "kind": "for_each", "for_each": "$a.value",
                 "depends_on": ["a"], "body": [{"id": "x", "ref": "echo"}]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert "not a list" in rec.error
        assert rec.compensated is True and log == ["echo:scalar", "undo:A"]
        assert eng.runs.load(rec.run_id).status == RunStatus.ABORTED

    @pytest.mark.parametrize("step, match", [
        ({"id": "c", "kind": "condition", "then_steps": [{"id": "t", "ref": "echo"}]},
         "no predicate"),
        ({"id": "p", "kind": "parallel"}, "no branches"),
        ({"id": "l", "kind": "for_each", "body": [{"id": "x", "ref": "echo"}]},
         "no for_each"),
        ({"id": "t", "ref": "no_such_tool"}, "unknown tool"),
        ({"id": "g", "kind": "agent", "ref": "ghost"}, "unknown agent"),
        ({"id": "inputs", "ref": "echo"}, "reserved"),
        ({"id": "a", "ref": "echo", "compensate": {"ref": "missing"}}, "unknown tool"),
    ])
    def test_bad_definitions_rejected_before_run(self, tmp_path: Path, step: dict[str, Any],
                                                 match: str) -> None:
        eng = _engine(tmp_path)
        try:
            with pytest.raises(ValueError, match=match):
                eng.start(_def([step]))
        finally:
            eng.shutdown()
        assert eng.runs.list() == []  # nothing left RUNNING


# -- MEDIUM 7/8: loop scope + reference syntax ----------------------------------------


class TestReferences:
    def test_loop_var_visible_in_parallel_branches(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "src", "ref": "echo", "inputs": {"value": [1, 2]}},
                {"id": "loop", "kind": "for_each", "for_each": "$src.value", "item_var": "n",
                 "depends_on": ["src"], "body": [{"id": "fan", "kind": "parallel", "branches": {
                     "l": [{"id": "x", "ref": "echo", "inputs": {"value": "$n"}}],
                     "r": [{"id": "y", "ref": "echo", "inputs": {"value": "$n"}}]}}]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED, rec.error
        assert rec.outputs["loop"]["items"][1] == {"l.x": {"value": 2}, "r.y": {"value": 2}}

    def test_literal_dollar_strings(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": ["$10", "$HOME", "$", "$$a.value"]}},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED, rec.error
        assert rec.outputs["a"] == {"value": ["$10", "$HOME", "$", "$a.value"]}

    def test_known_step_reference_still_strict(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": 1}},
                {"id": "b", "ref": "echo", "inputs": {"value": "$a.missing"},
                 "depends_on": ["a"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED and "unresolvable" in rec.error


# -- MEDIUM 9/10: skip output + on_failure=retry --------------------------------------


class TestFailureModes:
    def test_skipped_leaf_records_output(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "boom", "on_failure": "skip"},
                {"id": "b", "ref": "echo", "inputs": {"value": "$a.skipped"},
                 "depends_on": ["a"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED, rec.error
        assert rec.outputs["a"]["skipped"] is True and "kaput" in rec.outputs["a"]["reason"]
        assert rec.outputs["b"] == {"value": True}

    def test_on_failure_retry_retries(self, tmp_path: Path) -> None:
        calls: list[int] = []
        gw = ToolGateway()

        @gw.tool("flaky", parameters={"type": "object"})
        def flaky() -> dict:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient")
            return {"fine": True}

        eng = _engine(tmp_path, gw)
        try:
            rec = eng.start(_def([{"id": "f", "ref": "flaky", "on_failure": "retry"}]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED and len(calls) == 2


# -- MEDIUM 11/12: webhook secret + path traversal ------------------------------------


class TestSecurity:
    def test_webhook_secret(self, tmp_path: Path) -> None:
        eng = _engine(tmp_path)
        wf = _def([{"id": "a", "ref": "echo"}], triggers=[{"type": "webhook", "secret": "abc"}])
        try:
            for wrong in ("", "ab", "abcd", "ABC", "é"):
                with pytest.raises(PermissionError):
                    eng.trigger(wf, secret=wrong)
            assert eng.trigger(wf, secret="abc").status == RunStatus.COMMITTED
        finally:
            eng.shutdown()

    @pytest.mark.parametrize("name", ["../escape", "a/b", "..", ".hidden", "a\\b", ""])
    def test_registry_rejects_unsafe_names(self, tmp_path: Path, name: str) -> None:
        reg = WorkflowRegistry(tmp_path / "reg")
        with pytest.raises(ValueError):
            reg.save(_def([{"id": "a", "ref": "echo"}], name=name))
        with pytest.raises(KeyError):
            reg.get(name)
        assert reg.latest(name) is None
        assert sorted(p.name for p in tmp_path.iterdir()) == ["reg"]

    def test_runstore_rejects_unsafe_run_ids(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs")
        (tmp_path / "secret.json").write_text("{}")
        with pytest.raises(KeyError):
            store.load("../secret")


# -- MEDIUM 13: cron semantics ---------------------------------------------------------


class TestCron:
    def test_start_slash_step(self) -> None:
        for minute in (5, 20, 35, 50):
            assert cron_matches("5/15 * * * *", datetime(2026, 5, 4, 12, minute))
        assert not cron_matches("5/15 * * * *", datetime(2026, 5, 4, 12, 6))
        assert cron_matches("10-30/10 * * * *", datetime(2026, 5, 4, 12, 30))
        with pytest.raises(ValueError):
            cron_matches("*/0 * * * *", datetime(2026, 5, 4))

    def test_dom_dow_or_when_both_restricted(self) -> None:
        monday_4th = datetime(2026, 5, 4, 0, 0)
        friday_1st = datetime(2026, 5, 1, 0, 0)
        tuesday_5th = datetime(2026, 5, 5, 0, 0)
        assert cron_matches("0 0 1 * 1", monday_4th)  # DOW matches
        assert cron_matches("0 0 1 * 1", friday_1st)  # DOM matches
        assert not cron_matches("0 0 1 * 1", tuesday_5th)
        assert not cron_matches("0 0 * * 1", tuesday_5th)  # star DOM → AND
        assert cron_matches("0 0 */2 * 2", tuesday_5th)  # "*/2" counts as star → AND

    def test_leap_day_is_not_spent_and_fast(self) -> None:
        spec = ScheduleSpec(name="leap", workflow="w", kind="cron", cron="0 0 29 2 *")
        start = time.perf_counter()
        due_in = next_due_in(spec, datetime(2026, 5, 4))
        assert time.perf_counter() - start < 0.5
        assert due_in is not None
        expected = datetime(2028, 2, 29, tzinfo=timezone.utc) - datetime(
            2026, 5, 4, tzinfo=timezone.utc)
        assert due_in == expected.total_seconds()

    def test_missed_minute_catches_up_once(self) -> None:
        spec = ScheduleSpec(name="c", workflow="w", kind="cron", cron="30 12 * * *",
                            last_run_at="2026-05-02T12:30:05+00:00")
        late = datetime(2026, 5, 4, 12, 31)  # two slots missed
        assert schedule_due(spec, late)
        assert next_due_in(spec, late) == 0.0
        ran = spec.model_copy(update={"last_run_at": late.replace(
            tzinfo=timezone.utc).isoformat()})
        assert not schedule_due(ran, datetime(2026, 5, 4, 12, 45))  # no burst
        assert schedule_due(ran, datetime(2026, 5, 5, 12, 30))

    def test_created_at_baseline_for_never_run(self) -> None:
        spec = ScheduleSpec(name="c", workflow="w", kind="cron", cron="0 12 * * *",
                            created_at="2026-05-04T11:00:00Z")
        assert not schedule_due(spec, datetime(2026, 5, 4, 11, 59))
        assert schedule_due(spec, datetime(2026, 5, 4, 12, 7))  # missed 12:00 → catch up

    def test_z_suffix_timestamps(self) -> None:
        spec = ScheduleSpec(name="z", workflow="w", kind="at", at="2026-05-04T12:00:00Z")
        assert schedule_due(spec, datetime(2026, 5, 4, 12, 1))
        assert not schedule_due(spec, datetime(2026, 5, 4, 11, 59))
        interval = ScheduleSpec(name="i", workflow="w", kind="interval", interval_s=60,
                                last_run_at="2026-05-04T12:00:00Z")
        assert next_due_in(interval, datetime(2026, 5, 4, 12, 0, 30)) == 30.0


# -- MEDIUM 14: run_pending claims each schedule individually -------------------------


class _Run:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = RunStatus.COMMITTED


class TestRunPendingConcurrency:
    def test_concurrent_put_survives_long_run(self, tmp_path: Path) -> None:
        sched = Scheduler(tmp_path / "sched")
        sched.put(ScheduleSpec(name="job", workflow="w", kind="interval", interval_s=9999))

        class Engine:
            def start(self, workflow: Any, inputs: Any, tags: Any = None) -> _Run:
                # someone edits schedules while the workflow runs
                Scheduler(tmp_path / "sched").put(
                    ScheduleSpec(name="added", workflow="w", kind="interval"))
                return _Run("r1")

        results = sched.run_pending(Engine(), lambda name: None)
        assert [r["schedule"] for r in results] == ["job"]
        names = {s.name: s for s in sched.all()}
        assert "added" in names
        assert names["job"].last_status == "committed" and names["job"].last_run_at

    def test_crash_mid_run_keeps_claim(self, tmp_path: Path) -> None:
        sched = Scheduler(tmp_path / "sched")
        for name in ("a", "b"):
            sched.put(ScheduleSpec(name=name, workflow="w", kind="interval", interval_s=9999))

        class Crashing:
            def start(self, workflow: Any, inputs: Any, tags: Any = None) -> _Run:
                raise KeyboardInterrupt  # process dies mid-run

        with pytest.raises(KeyboardInterrupt):
            sched.run_pending(Crashing(), lambda name: None)
        state = {s.name: s for s in sched.all()}
        claimed = [n for n, s in state.items() if s.last_run_at]
        assert len(claimed) == 1 and state[claimed[0]].last_status == "running"
        assert [s.name for s in sched.tick()] == [n for n in state if n not in claimed]

    def test_two_schedulers_do_not_double_run(self, tmp_path: Path) -> None:
        Scheduler(tmp_path / "sched").put(
            ScheduleSpec(name="once", workflow="w", kind="interval", interval_s=9999))
        runs: list[str] = []
        runs_lock = threading.Lock()

        class Slow:
            def start(self, workflow: Any, inputs: Any, tags: Any = None) -> _Run:
                with runs_lock:
                    runs.append(threading.current_thread().name)
                time.sleep(0.2)
                return _Run("r")

        start = threading.Barrier(2)

        def worker() -> None:
            start.wait(timeout=5)
            Scheduler(tmp_path / "sched").run_pending(Slow(), lambda name: None)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(runs) == 1


# -- MEDIUM 15: backend factory + cooperative cancel ----------------------------------


class TestBackendAndCancel:
    def test_factory_accepts_simulate_kwarg(self, tmp_path: Path) -> None:
        backend = create_workflow_backend("inngest", simulate=True, vcr_dir=str(tmp_path))
        assert isinstance(backend, InngestWorkflowBackend) and backend.simulate

    def test_cancel_live_run_compensates_and_sticks(self, tmp_path: Path) -> None:
        log: list[str] = []
        entered, release = threading.Event(), threading.Event()

        def blocker(inputs: dict[str, Any]) -> dict:
            entered.set()
            release.wait(5)
            return {"ok": True}

        eng = _engine(tmp_path, _gateway(log), agents={"block": blocker})
        backend = LocalWorkflowBackend(engine=eng)
        wf = _def([
            {"id": "a", "ref": "echo", "inputs": {"value": "A"},
             "compensate": {"ref": "undo", "inputs": {"value": "A"}}},
            {"id": "b", "kind": "agent", "ref": "block", "depends_on": ["a"]},
            {"id": "c", "ref": "echo", "inputs": {"value": "C"}, "depends_on": ["b"]},
        ])
        out: list[Any] = []
        runner = threading.Thread(target=lambda: out.append(backend.dispatch(wf)))
        try:
            runner.start()
            assert entered.wait(5)
            run_id = eng.runs.list()[0]["run_id"]
            assert backend.cancel_run(run_id) is True
            release.set()
            runner.join(timeout=10)
        finally:
            release.set()
            eng.shutdown()
        rec = out[0]
        assert rec.status == RunStatus.ABORTED and rec.cancelled
        assert "cancelled" in rec.error.lower()
        assert "echo:C" not in log and log[-1] == "undo:A"
        stored = eng.runs.load(rec.run_id)
        assert stored.status == RunStatus.ABORTED and stored.compensated
        assert backend.cancel_run(rec.run_id) is False  # terminal → idempotent no-op

    def test_cancel_idle_run_finalizes(self, tmp_path: Path) -> None:
        log: list[str] = []
        eng = _engine(tmp_path, _gateway(log))
        wf = _def([
            {"id": "a", "ref": "echo", "inputs": {"value": "A"},
             "compensate": {"ref": "undo", "inputs": {"value": "A"}}},
            {"id": "b", "ref": "echo", "depends_on": ["a"]},
        ])
        try:
            record = eng.runs.create(wf, {})
            assert eng.step_once(record.run_id) is False  # "a" done, run idle
            assert LocalWorkflowBackend(engine=eng).cancel_run(record.run_id) is True
            final = eng.runs.load(record.run_id)
            assert final.status == RunStatus.ABORTED and final.cancelled
            assert log == ["echo:A", "undo:A"]
            assert eng.resume(record.run_id).status == RunStatus.ABORTED  # not revived
        finally:
            eng.shutdown()
