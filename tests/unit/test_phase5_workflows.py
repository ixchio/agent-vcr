"""Phase 5 tests: registry, engine, compensation, resume, schedules, API/CLI.

Handlers are real in-memory tools; timing uses zero delays; the only
real sleeps are a 1.2s-interval scheduler test (~3s) and sub-second
timeout tests. No mocks, no network (server tests use TestClient ASGI).
"""

from __future__ import annotations

import time

import pytest

import agent_vcr
from agent_vcr.player import VCRPlayer
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
from agent_vcr.workflows.store import RunStore


def _gateway(log=None):
    gw = ToolGateway()

    @gw.tool("echo", parameters={"type": "object"})
    def echo(value=None) -> dict:
        if log is not None:
            log.append(f"echo:{value}")
        return {"value": value}

    @gw.tool("shout", parameters={"type": "object"})
    def shout(value=None) -> dict:
        return {"loud": str(value).upper()}

    @gw.tool("undo_echo", parameters={"type": "object"})
    def undo_echo(value=None) -> dict:
        if log is not None:
            log.append(f"undo:{value}")
        return {"undone": value}

    @gw.tool("boom", parameters={"type": "object"})
    def boom() -> dict:
        raise RuntimeError("kaput")

    return gw


def _engine(tmp_path, log=None, **kwargs):
    kwargs.setdefault("approver", CallbackApprover(lambda req: True, name="yes"))
    return WorkflowEngine(
        _gateway(log),
        run_store=RunStore(tmp_path / "runs"),
        vcr_dir=str(tmp_path / "vcr"),
        **kwargs,
    )


def _def(steps, name="wf", **kwargs):
    return WorkflowDef(name=name, steps=[WorkflowStep(**s) for s in steps], **kwargs)


class TestRegistryAndRuns:
    def test_versioning(self, tmp_path):
        reg = WorkflowRegistry(tmp_path)
        v1 = reg.save(_def([{"id": "a", "ref": "echo"}], name="w"))
        assert v1.version == 1
        same = reg.save(_def([{"id": "a", "ref": "echo"}], name="w"))
        assert same.version == 1  # unchanged = no bump
        v2 = reg.save(_def([{"id": "a", "ref": "echo"}, {"id": "b", "ref": "echo"}], name="w"))
        assert v2.version == 2
        assert reg.get("w").version == 2
        assert reg.get("w", 1).version == 1
        assert reg.list()[0]["steps"] == 2
        with pytest.raises(KeyError):
            reg.get("missing")

    def test_runstore(self, tmp_path):
        store = RunStore(tmp_path)
        rec = store.create(_def([{"id": "a", "ref": "echo"}], name="w"), {"x": 1})
        assert store.load(rec.run_id).inputs == {"x": 1}
        assert store.list("w")[0]["status"] == "running"
        with pytest.raises(KeyError):
            store.load("nope")
        (tmp_path / f"{rec.run_id}.json").write_text("{bad")
        with pytest.raises(ValueError, match="Corrupt"):
            store.load(rec.run_id)


class TestDagValidation:
    def test_rejects(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            with pytest.raises(ValueError, match="Duplicate"):
                eng.start(_def([{"id": "a", "ref": "e"}, {"id": "a", "ref": "e"}]))
            with pytest.raises(ValueError, match="unknown"):
                eng.start(_def([{"id": "a", "ref": "e", "depends_on": ["ghost"]}]))
            with pytest.raises(ValueError, match="cycle"):
                eng.start(_def([
                    {"id": "a", "ref": "e", "depends_on": ["b"]},
                    {"id": "b", "ref": "e", "depends_on": ["a"]}]))
        finally:
            eng.shutdown()


class TestLinearAndTrace:
    def test_committed_with_wiring(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "one", "ref": "echo", "inputs": {"value": "hi"}},
                {"id": "two", "ref": "shout", "inputs": {"value": "$one.value"},
                 "depends_on": ["one"], "checkpoint_after": True},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED
        assert rec.outputs["two"] == {"loud": "HI"}
        assert rec.workflow_version == 1
        player = VCRPlayer.load(tmp_path / "vcr" / f"{rec.session_id}.vcr")
        assert "workflow:start" in player.list_nodes()
        assert player.goto_checkpoint("two")
        assert player.verify_integrity()["ok"]


class TestBranchesLoopsParallel:
    def test_condition(self, tmp_path):
        eng = _engine(tmp_path)
        wf = lambda flag: _def([  # noqa: E731
            {"id": "one", "ref": "echo", "inputs": {"value": flag}},
            {"id": "pick", "kind": "condition",
             "condition": {"ref": "$one.value", "op": "==", "value": "big"},
             "then_steps": [{"id": "t", "ref": "shout", "inputs": {"value": "taken"}}],
             "else_steps": [{"id": "e", "ref": "shout", "inputs": {"value": "left"}}],
             "depends_on": ["one"]},
        ])
        try:
            assert eng.start(wf("big")).outputs["pick.t"] == {"loud": "TAKEN"}
            assert eng.start(wf("small")).outputs["pick.e"] == {"loud": "LEFT"}
        finally:
            eng.shutdown()

    def test_parallel_join(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([{"id": "fan", "kind": "parallel", "branches": {
                "left": [{"id": "l", "ref": "echo", "inputs": {"value": 1}}],
                "right": [{"id": "r", "ref": "echo", "inputs": {"value": 2}}],
            }}]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED
        assert rec.outputs["fan"] == {"left.l": {"value": 1}, "right.r": {"value": 2}}

    def test_foreach(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "src", "ref": "echo", "inputs": {"value": [1, 2, 3]}},
                {"id": "loop", "kind": "for_each", "for_each": "$src.value",
                 "item_var": "n", "depends_on": ["src"],
                 "body": [{"id": "sh", "ref": "shout", "inputs": {"value": "$n"}}]},
            ]))
            assert rec.outputs["loop"] == {"items": [{"loud": "1"}, {"loud": "2"},
                                                      {"loud": "3"}], "count": 3}
            with pytest.raises(ValueError, match="max"):
                eng.start(_def([
                    {"id": "src", "ref": "echo", "inputs": {"value": [1, 2]}},
                    {"id": "loop", "kind": "for_each", "for_each": "$src.value",
                     "max_iterations": 1, "depends_on": ["src"],
                     "body": [{"id": "x", "ref": "echo"}]},
                ]))
        finally:
            eng.shutdown()


class TestFailuresAndCompensation:
    def test_retry_then_ok(self, tmp_path):
        calls = []
        gw = ToolGateway()

        @gw.tool("flaky", parameters={"type": "object"})
        def flaky() -> dict:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient")
            return {"fine": True}

        eng = WorkflowEngine(gw, run_store=RunStore(tmp_path / "runs"),
                             vcr_dir=str(tmp_path / "vcr"),
                             approver=CallbackApprover(lambda r: True))
        try:
            rec = eng.start(_def([{"id": "f", "ref": "flaky",
                                   "max_retries": 1, "retry_delay_s": 0.0}]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.COMMITTED and len(calls) == 2

    def test_abort_compensates_in_reverse(self, tmp_path):
        log: list[str] = []
        eng = _engine(tmp_path, log)
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": "A"},
                 "compensate": {"ref": "undo_echo", "inputs": {"value": "A"}}},
                {"id": "b", "ref": "echo", "inputs": {"value": "B"},
                 "depends_on": ["a"],
                 "compensate": {"ref": "undo_echo", "inputs": {"value": "B"}}},
                {"id": "c", "ref": "boom", "depends_on": ["b"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert rec.compensated is True
        assert log == ["echo:A", "echo:B", "undo:B", "undo:A"]
        assert "kaput" in rec.error

    def test_failed_compensation_flagged(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            rec = eng.start(_def([
                {"id": "a", "ref": "echo", "inputs": {"value": 1},
                 "compensate": {"ref": "boom"}},
                {"id": "b", "ref": "boom", "depends_on": ["a"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert rec.compensated is False
        assert "compensation" in rec.error

    def test_timeout_and_denied(self, tmp_path):
        gw = ToolGateway()

        @gw.tool("slow", parameters={"type": "object"})
        def slow() -> dict:
            time.sleep(5)
            return {}

        eng = WorkflowEngine(gw, run_store=RunStore(tmp_path / "runs"),
                             vcr_dir=str(tmp_path / "vcr"),
                             approver=CallbackApprover(lambda r: False))
        try:
            rec = eng.start(_def([{"id": "s", "ref": "slow", "timeout_s": 0.2}]))
            assert rec.status == RunStatus.ABORTED and "exceeded" in rec.error
            rec2 = eng.start(_def([{"id": "g", "ref": "slow", "timeout_s": 30,
                                    "requires_approval": True, "on_failure": "skip"}]))
            assert rec2.status == RunStatus.COMMITTED
        finally:
            eng.shutdown()

    def test_policy_step_cap_applies_to_each_workflow_action(self, tmp_path):
        eng = _engine(tmp_path, policy=Policy(max_steps=1))
        try:
            rec = eng.start(_def([
                {"id": "one", "ref": "echo", "inputs": {"value": 1}},
                {"id": "two", "ref": "echo", "inputs": {"value": 2},
                 "depends_on": ["one"]},
            ]))
        finally:
            eng.shutdown()
        assert rec.status == RunStatus.ABORTED
        assert "step budget exceeded" in rec.error

    def test_reusable_engine_resets_execution_budget(self, tmp_path):
        gateway = ToolGateway()

        @gateway.tool("metered", parameters={"type": "object"})
        def metered() -> dict:
            return {"output": {"ok": True}, "cost_usd": 0.8}

        eng = WorkflowEngine(
            gateway,
            run_store=RunStore(tmp_path / "runs"),
            vcr_dir=str(tmp_path / "vcr"),
            budget=BudgetTracker(max_total_cost_usd=1.0),
            approver=CallbackApprover(lambda req: True),
        )
        workflow = _def([{"id": "run", "ref": "metered"}])
        try:
            assert eng.start(workflow).status == RunStatus.COMMITTED
            assert eng.start(workflow).status == RunStatus.COMMITTED
        finally:
            eng.shutdown()


class TestDurableResume:
    def test_step_once_then_resume_new_engine(self, tmp_path):
        eng = _engine(tmp_path)
        wf = _def([
            {"id": "one", "ref": "echo", "inputs": {"value": 1}},
            {"id": "two", "ref": "shout", "inputs": {"value": "$one.value"},
             "depends_on": ["one"]},
            {"id": "three", "ref": "echo", "inputs": {"value": "$two.loud"},
             "depends_on": ["two"]},
        ])
        try:
            store = RunStore(tmp_path / "runs")
            record = eng.runs.create(wf, {})
            from agent_vcr.recorder import VCRRecorder

            session = VCRRecorder(output_dir=str(tmp_path / "vcr")).start_session()
            record.session_id = session.session_id
            eng.runs.persist(record)
            assert eng.step_once(record.run_id) is False  # one ran, more pending
            mid = store.load(record.run_id)
            assert mid.completed_top_steps == ["one"]
        finally:
            eng.shutdown()

        # "crash": brand-new engine on the same directories resumes
        eng2 = _engine(tmp_path)
        try:
            done = eng2.resume(record.run_id)
            assert done.status == RunStatus.COMMITTED
            assert done.outputs["three"] == {"value": "1"}
            player = VCRPlayer.load(tmp_path / "vcr" / f"{done.session_id}.vcr")
            assert len(player.frames) >= 4
            assert player.verify_integrity()["ok"]
        finally:
            eng2.shutdown()

    def test_trigger_secret(self, tmp_path):
        eng = _engine(tmp_path)
        try:
            wf = _def([{"id": "a", "ref": "echo"}],
                      triggers=[{"type": "webhook", "secret": "s3cr3t"}])
            with pytest.raises(PermissionError):
                eng.trigger(wf, secret="wrong")
            assert eng.trigger(wf, secret="s3cr3t").status == RunStatus.COMMITTED
        finally:
            eng.shutdown()


class TestSchedules:
    def test_cron_parser(self):
        from datetime import datetime

        noon = datetime(2026, 5, 4, 12, 30)  # a Monday
        assert cron_matches("* * * * *", noon)
        assert cron_matches("30 12 * * *", noon)
        assert not cron_matches("31 12 * * *", noon)
        assert cron_matches("*/15 * * * *", noon)
        assert not cron_matches("*/16 * * * *", noon)
        assert cron_matches("0 0 * * 0", datetime(2026, 5, 3, 0, 0))  # Sunday
        assert cron_matches("0 0 * * 7", datetime(2026, 5, 3, 0, 0))  # 7 = Sunday too
        with pytest.raises(ValueError):
            cron_matches("* * * *", noon)
        with pytest.raises(ValueError):
            cron_matches("99 * * * *", noon)

    def test_due_logic(self):
        from datetime import datetime

        now = datetime(2026, 5, 4, 12, 30)
        assert schedule_due(ScheduleSpec(name="i", workflow="w", kind="interval",
                                         interval_s=60), now)
        assert not schedule_due(ScheduleSpec(name="d", workflow="w", kind="interval",
                                             enabled=False), now)
        assert schedule_due(ScheduleSpec(name="c", workflow="w", kind="cron",
                                         cron="30 * * * *"), now)
        assert not schedule_due(ScheduleSpec(name="c2", workflow="w", kind="cron",
                                             cron="31 * * * *"), now)
        assert schedule_due(ScheduleSpec(name="a", workflow="w", kind="at",
                                         at="2026-05-04T12:00:00"), now)
        assert not schedule_due(ScheduleSpec(name="a2", workflow="w", kind="at",
                                             at="2026-05-04T13:00:00"), now)
        assert next_due_in(ScheduleSpec(name="i", workflow="w", kind="interval",
                                        interval_s=60), now) == 0.0

    def test_run_pending_live(self, tmp_path):

        eng = _engine(tmp_path)
        try:
            wf = _def([{"id": "a", "ref": "echo", "inputs": {"value": 1}}], name="sched")
            sched = Scheduler(tmp_path / "sched")
            sched.put(ScheduleSpec(name="every", workflow="sched", kind="interval",
                                   interval_s=9999))
            results = sched.run_pending(eng, lambda name: wf)
            assert results[0]["status"] == "committed"
            assert sched.run_pending(eng, lambda name: wf) == []  # not due again
        finally:
            eng.shutdown()

    def test_exports(self):
        assert agent_vcr.__version__ == "1.5.3"
        for name in ("WorkflowEngine", "WorkflowDef", "WorkflowRegistry",
                     "RunStore", "Scheduler", "ScheduleSpec", "RunRecord"):
            assert name in agent_vcr.__all__


class TestWorkflowAPI:
    def _server(self, tmp_path, handlers=None):
        from fastapi.testclient import TestClient

        from agent_vcr.server import VCRServer

        server = VCRServer(vcr_dir=str(tmp_path / "legacy"),
                           data_dir=str(tmp_path / "cloud"),
                           workflow_handlers=handlers)
        return TestClient(server.app, raise_server_exceptions=False)

    def _handlers(self, tmp_path):
        path = tmp_path / "handlers.py"
        path.write_text("def echo(value=None):\n    return {'value': value}\n"
                        "TOOLS = {'echo': echo}\n")
        return str(path)

    def test_registry_and_trigger(self, tmp_path):
        client = self._server(tmp_path, self._handlers(tmp_path))
        body = {"name": "hello", "steps": [
            {"id": "a", "kind": "tool", "ref": "echo", "inputs": {"value": 1}}]}
        assert client.post("/api/workflows", json=body).status_code == 201
        assert client.get("/api/workflows").json()["workflows"][0]["name"] == "hello"
        trig = client.post("/api/workflows/hello/trigger", json={"inputs": {}})
        assert trig.status_code == 200, trig.text
        run_id = trig.json()["run_id"]
        detail = client.get(f"/api/workflow-runs/{run_id}").json()
        assert detail["status"] == "committed"
        assert detail["outputs"]["a"] == {"value": 1}
        # The server reuses its registered handlers; a second request must
        # not see an executor that the first short-lived engine shut down.
        assert client.post("/api/workflows/hello/trigger", json={}).status_code == 200

    def test_no_handlers_conflict_and_secret(self, tmp_path):
        client = self._server(tmp_path)
        client.post("/api/workflows", json={"name": "h", "steps": [
            {"id": "a", "kind": "tool", "ref": "echo"}]})
        assert client.post("/api/workflows/h/trigger", json={}).status_code == 409
        client.post("/api/workflows", json={"name": "s", "steps": [
            {"id": "a", "kind": "checkpoint", "ref": "cp"}],
            "triggers": [{"type": "webhook", "secret": "shh"}]})
        assert client.post("/api/workflows/s/trigger",
                           json={"secret": "wrong"}).status_code == 403
        assert client.post("/api/workflows/s/trigger",
                           json={"secret": "shh"}).status_code == 200


class TestWorkflowCLI:
    def test_validate_run_show_resume(self, tmp_path, monkeypatch, capsys):
        import json as _json

        from agent_vcr.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "wf.json").write_text(_json.dumps({
            "name": "cli", "steps": [
                {"id": "a", "kind": "tool", "ref": "echo", "inputs": {"value": 5}},
                {"id": "b", "kind": "tool", "ref": "echo",
                 "inputs": {"value": "$a.value"}, "depends_on": ["a"]},
            ]}))
        (tmp_path / "handlers.py").write_text(
            "def echo(value=None):\n    return {'value': value}\nTOOLS = {'echo': echo}\n")
        main(["workflow", "validate", "wf.json"])
        assert "Valid: cli" in capsys.readouterr().out
        main(["workflow", "run", "--file", "wf.json", "--handlers", "handlers.py"])
        out = capsys.readouterr().out
        assert "committed" in out
        main(["workflow", "runs", "--workflow", "cli"])
        assert "committed" in capsys.readouterr().out

    def test_schedule_put_tick(self, tmp_path, monkeypatch, capsys):
        import json as _json

        from agent_vcr.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "wf.json").write_text(_json.dumps({
            "name": "tock", "steps": [{"id": "a", "kind": "tool", "ref": "echo"}]}))
        (tmp_path / "handlers.py").write_text(
            "def echo(value=None):\n    return {'value': value}\nTOOLS = {'echo': echo}\n")
        main(["workflow", "run", "--file", "wf.json", "--handlers", "handlers.py"])
        capsys.readouterr()
        main(["workflow", "schedule", "put", "--name", "every",
              "--workflow", "tock", "--kind", "interval", "--interval-s", "9999"])
        main(["workflow", "schedule", "tick", "--handlers", "handlers.py"])
        assert "committed" in capsys.readouterr().out


class TestScheduleFixes:
    def test_mixed_naive_aware_datetimes(self):
        from datetime import datetime, timezone

        from agent_vcr.workflows.schedules import next_due_in, schedule_due

        naive = datetime(2026, 5, 4, 12, 30)
        aware = datetime(2026, 5, 4, 12, 30, tzinfo=timezone.utc)
        spec = ScheduleSpec(name="m", workflow="w", kind="interval", interval_s=60,
                            last_run_at="2026-05-04T12:00:00+00:00")
        # neither combination may raise; both agree (30 min > 60 s)
        assert schedule_due(spec, naive) is True
        assert schedule_due(spec, aware) is True
        assert next_due_in(spec, naive) == 0.0
        assert next_due_in(spec, aware) == 0.0

    def test_impossible_cron_returns_none_fast(self):
        import time as _time
        from datetime import datetime

        from agent_vcr.workflows.schedules import next_due_in

        spec = ScheduleSpec(name="never", workflow="w", kind="cron", cron="30 2 30 2 *")
        start = _time.perf_counter()
        assert next_due_in(spec, datetime(2026, 5, 4, 12, 30)) is None
        assert _time.perf_counter() - start < 5.0
