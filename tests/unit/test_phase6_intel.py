"""Phase 6 tests: clusters, risk, recovery, routing, costs, diagnosis.

Fixtures are real recorded sessions (errors, models, costs, denials).
No mocks, no network (server checks use TestClient ASGI).
"""

from __future__ import annotations

import pytest

import agent_vcr
from agent_vcr.intelligence import (
    ModelRegistry,
    breakdown,
    classify_player,
    cluster_sessions,
    diagnose,
    normalize_message,
    plan_recovery,
    route,
    score_agent,
    score_session,
)
from agent_vcr.intelligence.models import FailureKind
from agent_vcr.intelligence.recovery import auto_recover_workflow
from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


def _record(directory, session_id, frames):
    rec = VCRRecorder(output_dir=str(directory), auto_save=False)
    rec.start_session(session_id)
    for kind, node, out, meta in frames:
        if kind == "error":
            rec.record_error(node, {}, out, latency_ms=1.0)
        else:
            rec.record_step(node, {}, out,
                            metadata=FrameMetadata(**meta) if meta else None)
    return rec.save()


def _llm(directory, session_id, model, cost, tokens, latency=50.0, err=False):
    rec = VCRRecorder(output_dir=str(directory), auto_save=False)
    rec.start_session(session_id)
    if err:
        rec.record_error(f"llm:{model}", {}, ValueError("boom"),
                         latency_ms=latency, model=model)
    else:
        rec.record_llm_call(model, [{"role": "user", "content": "hi"}],
                            {"content": "yo"}, tokens // 2, tokens - tokens // 2,
                            latency, cost_usd=cost)
    return rec.save()


def _fleet(tmp_path):
    d = tmp_path / "fleet"
    _record(d, "s-timeout-1", [("error", "fetch", ValueError("timeout after 3000ms"), None)])
    _record(d, "s-timeout-2", [("error", "fetch", ValueError("timeout after 9123ms"), None)])
    _record(d, "s-ok", [("node", "fetch", {"ok": True},
                         {"model": "gpt-4o", "tokens_used": 100, "cost_usd": 0.01})])
    return d


class TestClustering:
    def test_normalize(self):
        assert normalize_message("timeout after 3000ms on /a/b/c") == \
            normalize_message("timeout after 45ms on /x/y")
        assert "<uuid>" in normalize_message("job 123e4567-e89b-12d3-a456-426614174000 died")
        assert normalize_message("plain failure") == "plain failure"

    def test_clusters_repeats(self, tmp_path):
        d = _fleet(tmp_path)
        clusters = cluster_sessions(d)
        assert len(clusters) == 1
        assert clusters[0].occurrences == 2
        assert sorted(clusters[0].sessions) == ["s-timeout-1", "s-timeout-2"]
        assert "<n>" in clusters[0].template


class TestRisk:
    def test_clean_session_low(self, tmp_path):
        d = _fleet(tmp_path)
        score = score_session(VCRPlayer.load(d / "s-ok.vcr"))
        assert score.level == "LOW" and score.score < 30

    def test_error_session_high(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("bad")
        for _ in range(3):
            rec.record_error("n", {}, RuntimeError("x"), latency_ms=1.0)
        path = rec.save()
        score = score_session(VCRPlayer.load(path))
        assert score.level == "HIGH"

    def test_agent_aggregation(self, tmp_path):
        d = _fleet(tmp_path)
        agent = score_agent(d)
        assert agent.sessions_analyzed == 3
        assert agent.level in ("LOW", "MEDIUM", "HIGH")
        assert score_agent(tmp_path / "empty").sessions_analyzed == 0


class TestRecovery:
    def test_classify(self, tmp_path):
        d = _fleet(tmp_path)
        kind, _ = classify_player(VCRPlayer.load(d / "s-timeout-1.vcr"))
        assert kind == FailureKind.TRANSIENT
        kind, _ = classify_player(VCRPlayer.load(d / "s-ok.vcr"))
        assert kind == FailureKind.UNKNOWN

    def test_plan_transient_auto_retry(self, tmp_path):
        d = _fleet(tmp_path)
        plan = plan_recovery(VCRPlayer.load(d / "s-timeout-1.vcr"))
        assert plan.failure_kind == FailureKind.TRANSIENT
        assert plan.actions[0].action == "retry" and plan.actions[0].auto is True
        assert plan.actions[-1].action == "escalate"

    def test_plan_policy_and_resume(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("denied")
        rec.record_step("policy:x", {}, {"decision": "deny"})
        path = rec.save()
        plan = plan_recovery(VCRPlayer.load(path), run_id="run-1")
        assert plan.failure_kind == FailureKind.POLICY
        assert any(a.action == "resume" and a.auto for a in plan.actions)

    def test_auto_recover_terminal_and_running(self, tmp_path):
        from agent_vcr.workflows import WorkflowDef, WorkflowStep
        from agent_vcr.workflows.store import RunStore

        store = RunStore(tmp_path / "runs")
        definition = WorkflowDef(name="w", steps=[WorkflowStep(id="a", ref="echo")])
        record = store.create(definition, {})
        from agent_vcr.workflows.models import RunStatus

        record.status = RunStatus.ABORTED
        store.persist(record)

        class FakeEngine:
            runs = store

            def resume(self, run_id):
                raise AssertionError("must not resume terminal runs")

        out = auto_recover_workflow(record.run_id, FakeEngine())
        assert out["recovered"] is False and out["status"] == "aborted"


class TestRouting:
    def test_cheapest_viable(self, tmp_path):
        d = tmp_path / "models"
        _llm(d, "a1", "gpt-4o", cost=0.10, tokens=1000)
        _llm(d, "a2", "gpt-4o-mini", cost=0.01, tokens=1000)
        _llm(d, "a3", "flaky", cost=0.001, tokens=10, err=True)
        registry = ModelRegistry().ingest_dir(d)
        assert set(registry.stats()) == {"gpt-4o", "gpt-4o-mini", "flaky"}
        decision = route(registry)
        assert decision.model == "gpt-4o-mini"
        assert "cheapest" in decision.reason
        prefer = route(registry, prefer=["gpt-4o", "gpt-4o-mini"])
        assert prefer.model == "gpt-4o"
        with pytest.raises(ValueError):
            route(ModelRegistry())

    def test_constraints_relax_openly(self, tmp_path):
        d = tmp_path / "models"
        _llm(d, "a1", "only", cost=0.5, tokens=10)
        decision = route(ModelRegistry().ingest_dir(d), max_avg_cost=0.0001)
        assert decision.model == "only" and "relaxed" in decision.reason


class TestCosts:
    def test_breakdown_and_duplicates(self, tmp_path):
        d = tmp_path / "spend"
        for i in range(2):
            rec = VCRRecorder(output_dir=str(d), auto_save=False)
            rec.start_session(f"s{i}")
            rec.record_tool_call("search", {"q": "same"}, {"hits": 1}, latency_ms=5.0)
            rec.record_step("llm", {}, {"done": True},
                            metadata=FrameMetadata(model="gpt-4o", tokens_used=500,
                                                   cost_usd=0.05))
            rec.save()
        report = breakdown(d)
        assert report.total_cost_usd == pytest.approx(0.10)
        assert report.by_model["gpt-4o"] == pytest.approx(0.10)
        assert report.by_step[0]["cost_usd"] == pytest.approx(0.05)
        dup = next(e for e in report.duplicate_calls if e["node"] == "tool:search")
        assert dup["count"] == 2
        assert report.estimated_ghost_savings_usd >= 0.0


class TestDiagnose:
    def test_failed_session(self, tmp_path):
        d = _fleet(tmp_path)
        report = diagnose(d / "s-timeout-1.vcr", siblings_dir=d)
        assert report.verdict == "failed"
        assert len(report.timeline) == 1
        assert "timeout" in report.likely_cause
        assert report.recovery is not None
        assert report.risk is not None

    def test_healthy_session(self, tmp_path):
        d = _fleet(tmp_path)
        report = diagnose(d / "s-ok.vcr", siblings_dir=d)
        assert report.verdict == "ok"

    def test_public_api(self):
        assert agent_vcr.__version__ == "1.5.0"
        for name in ("cluster_sessions", "score_agent", "score_session",
                     "classify_player", "plan_recovery", "auto_recover_workflow",
                     "ModelRegistry", "route", "breakdown", "diagnose",
                     "normalize_message", "FailureKind"):
            assert name in agent_vcr.__all__


class TestIntelCLI:
    def test_commands(self, tmp_path, monkeypatch, capsys):
        from agent_vcr.cli import main

        d = _fleet(tmp_path)
        monkeypatch.chdir(tmp_path)
        main(["intel", "failures", "--dir", str(d)])
        assert "timeout" in capsys.readouterr().out
        main(["intel", "risk", "--dir", str(d)])
        assert "score" in capsys.readouterr().out
        main(["intel", "costs", "--dir", str(d)])
        assert "total_cost_usd" in capsys.readouterr().out
        main(["intel", "routes", "--dir", str(d)])
        assert "gpt-4o" in capsys.readouterr().out
        main(["intel", "why", str(d / "s-ok.vcr")])
        out = capsys.readouterr().out
        assert "s-ok" in out

    def test_why_exit_code(self, tmp_path, monkeypatch):
        from agent_vcr.cli import main

        d = _fleet(tmp_path)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main(["intel", "why", str(d / "s-timeout-1.vcr")])
        assert exc.value.code == 1


class TestIntelAPI:
    def test_endpoints(self, tmp_path):
        from fastapi.testclient import TestClient

        from agent_vcr.server import VCRServer

        d = _fleet(tmp_path)
        for src in d.glob("*.vcr"):
            (tmp_path / "cloud" / "projects" / "p" / "sessions").mkdir(parents=True, exist_ok=True)
            (tmp_path / "cloud" / "projects" / "p" / "sessions" / src.name).write_bytes(
                src.read_bytes())
        from agent_vcr.cloud_store import CloudStore

        CloudStore(tmp_path / "cloud").create_project("P", project_id="p")
        server = VCRServer(vcr_dir=str(tmp_path / "legacy"),
                           data_dir=str(tmp_path / "cloud"))
        client = TestClient(server.app, raise_server_exceptions=False)
        clusters = client.get("/api/projects/p/intel/failures").json()["clusters"]
        assert len(clusters) == 1 and clusters[0]["occurrences"] == 2
        risk = client.get("/api/projects/p/intel/risk").json()
        assert risk["sessions_analyzed"] == 3
        why = client.get("/api/projects/p/intel/why/s-timeout-1").json()
        assert why["verdict"] == "failed"
        assert client.get("/api/projects/p/intel/why/missing").status_code == 404
