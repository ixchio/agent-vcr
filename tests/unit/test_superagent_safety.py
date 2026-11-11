"""SuperAgent, sandbox and intelligence safety: redaction, timeouts, budgets, routing."""

from __future__ import annotations

import gc
import json
import shutil
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from agent_vcr.intelligence import ModelRegistry, breakdown, classify_player, plan_recovery, route
from agent_vcr.intelligence.models import FailureKind
from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.sandbox import E2BSandbox, LocalSandbox, SandboxConfig, create_sandbox
from agent_vcr.superagent import (
    BudgetTracker,
    CallbackApprover,
    InMemoryVectorStore,
    MemoryStore,
    PlanBuilder,
    SuperAgent,
    ToolGateway,
    ToolSpec,
    ToolTimeoutError,
    coerce_result,
    redact_text,
    redact_value,
)
from agent_vcr.superagent.vector_memory import QdrantVectorStore


def _approve_all() -> CallbackApprover:
    return CallbackApprover(lambda req: True)


# ---------------------------------------------------------------------------
# 1. Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_sensitive_keys_redacted_case_insensitive_and_nested(self) -> None:
        out = redact_value({
            "password": "hunter2",
            "api_key": "abc123xyz",
            "headers": {"Authorization": "Bearer abcdef123", "X-Api-Key": "k1"},
            "creds": [{"clientSecret": "s3"}, {"authToken": "t0k"}],
            "PASSWORD": "Hunter3",
            "user": "bob",
            "tokens_used": 42,
            "max_tokens": 100,
        })
        blob = json.dumps(out)
        for secret in ("hunter2", "abc123xyz", "abcdef123", "k1", "s3", "t0k", "Hunter3"):
            assert secret not in blob
        assert out["user"] == "bob"
        assert out["tokens_used"] == 42 and out["max_tokens"] == 100

    def test_auth_schemes_and_token_formats(self) -> None:
        text = ("Bearer abcdef123 | Authorization: Basic dXNlcjpwYXNz | "
                "sk-proj-abcdefgh12 AKIAIOSFODNN7EXAMPLE ghp_abcdefgh1234 "
                "xoxb-1234567890-abc eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4")
        clean = redact_text(text)
        for secret in ("abcdef123", "dXNlcjpwYXNz", "sk-proj", "AKIAIOSFODNN7EXAMPLE",
                       "ghp_abcdefgh1234", "xoxb-", "eyJhbGci"):
            assert secret not in clean
        # Prose is left alone.
        assert redact_text("basic usage; the bearer of news") == "basic usage; the bearer of news"

    def test_json_pairs_and_private_key_body(self) -> None:
        assert "hunter2" not in redact_text(json.dumps({"password": "hunter2"}))
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEsecretbody\n-----END RSA PRIVATE KEY-----"
        assert "MIIEsecretbody" not in redact_text(pem)

    def test_output_state_redacted_in_recording_and_vector_index(self, tmp_path: Path) -> None:
        gateway = ToolGateway()

        @gateway.tool("login", parameters={"type": "object"})
        def login() -> dict[str, Any]:
            return {"user": "bob", "password": "hunter2", "note": "Bearer abcdef123"}

        rec_dir = tmp_path / "vcr"
        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(rec_dir)),
                           approver=_approve_all())
        try:
            report = agent.run("g", {"goal": "g", "steps": [
                {"id": "l", "ref": "login", "checkpoint_after": True}]})
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert report.outcomes[0].output["password"] == "hunter2"  # caller still gets raw
        path = rec_dir / f"{report.session_id}.vcr"
        raw = path.read_text()
        assert "hunter2" not in raw and "abcdef123" not in raw
        mem = InMemoryVectorStore()
        mem.index_vcr_session(path)
        assert all("hunter2" not in r.text for r in mem._records.values())

    def test_index_redacts_unredacted_recordings(self, tmp_path: Path) -> None:
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("raw")
        rec.record_step("n", {"api_key": "abc123xyz"}, {"password": "hunter2"})
        path = rec.save()
        mem = InMemoryVectorStore()
        mem.index_vcr_session(path)
        texts = " ".join(r.text for r in mem._records.values())
        assert "abc123xyz" not in texts and "hunter2" not in texts

    def test_memory_store_save_is_atomic(self, tmp_path: Path) -> None:
        mem = MemoryStore()
        mem.put("password", "hunter2")
        out = mem.save(tmp_path / "m.json")
        assert "hunter2" not in out.read_text()
        assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# 2. Timeouts / idempotency
# ---------------------------------------------------------------------------


class TestTimeoutRetries:
    def _run_slow(self, tmp_path: Path, idempotent: bool, recovery: bool = True) -> tuple[
            Any, list[int]]:
        gateway = ToolGateway()
        calls: list[int] = []
        lock = threading.Lock()

        @gateway.tool("slow", parameters={"type": "object"}, idempotent=idempotent)
        def slow() -> dict[str, Any]:
            with lock:
                calls.append(1)
            time.sleep(0.4)
            return {}

        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path / "v")),
                           approver=_approve_all(), recovery=recovery)
        try:
            report = agent.run("t", {"goal": "t", "steps": [
                {"id": "s", "ref": "slow", "timeout_s": 0.2, "max_retries": 1}]})
        finally:
            agent.shutdown()
        return report, calls

    def test_non_idempotent_tool_not_retried_after_timeout(self, tmp_path: Path) -> None:
        report, calls = self._run_slow(tmp_path, idempotent=False)
        assert len(calls) == 1
        assert report.status == "aborted"
        assert "not idempotent" in report.outcomes[0].reasons[0]

    def test_idempotent_tool_is_retried(self, tmp_path: Path) -> None:
        report, calls = self._run_slow(tmp_path, idempotent=True, recovery=False)
        assert len(calls) == 2
        assert report.outcomes[0].attempts == 2

    def test_gateway_raises_tool_timeout_error(self) -> None:
        gateway = ToolGateway()
        gateway.register(ToolSpec(name="s", handler=lambda: time.sleep(0.3)))
        try:
            with pytest.raises(ToolTimeoutError) as info:
                gateway.call("s", {}, timeout_s=0.05)
        finally:
            gateway.shutdown()
        err = info.value
        assert isinstance(err, TimeoutError) and isinstance(err, ToolTimeoutError)
        assert err.still_running and not err.idempotent

    def test_recovery_auto_retry_requires_idempotency(self, tmp_path: Path) -> None:
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("t")
        rec.record_error("tool:x", {}, TimeoutError("Tool 'x' exceeded 0.2s"), latency_ms=1.0)
        player = VCRPlayer.load(rec.save())
        assert plan_recovery(player).actions[0].auto is False
        assert plan_recovery(player, idempotent=False).actions[0].auto is False
        assert plan_recovery(player, idempotent=True).actions[0].auto is True

    def test_recovery_reads_recorded_idempotency(self, tmp_path: Path) -> None:
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("t")
        rec.record_step("tool:x", {}, {"error": "timed out"}, frame_type=FrameType.ERROR,
                        metadata=FrameMetadata(error_message="timed out",
                                               custom={"idempotent": True}))
        assert plan_recovery(VCRPlayer.load(rec.save())).actions[0].auto is True


# ---------------------------------------------------------------------------
# 3. Per-run budget
# ---------------------------------------------------------------------------


class TestPerRunBudget:
    def test_plan_cap_does_not_leak_into_later_runs(self, tmp_path: Path) -> None:
        gateway = ToolGateway()

        @gateway.tool("m", parameters={"type": "object"})
        def m() -> dict[str, Any]:
            return {"output": {"ok": True}, "cost_usd": 0.5}

        shared = BudgetTracker()
        agent = SuperAgent(gateway, budget=shared, approver=_approve_all(),
                           recorder=VCRRecorder(output_dir=str(tmp_path / "v")))
        two = {"goal": "g", "steps": [{"id": "x", "ref": "m"},
                                      {"id": "y", "ref": "m", "depends_on": ["x"]}]}
        try:
            capped = agent.run("g", {**two, "max_total_cost_usd": 0.1})
            free = agent.run("g", two)
        finally:
            agent.shutdown()
        assert [o.status for o in capped.outcomes] == ["ok", "blocked"]
        assert [o.status for o in free.outcomes] == ["ok", "ok"]
        assert shared.max_total_cost_usd is None
        assert shared.cost_usd == pytest.approx(1.0)  # mirrors the last run


# ---------------------------------------------------------------------------
# 4. coerce_result
# ---------------------------------------------------------------------------


class TestCoerceResult:
    def test_non_envelope_output_dicts(self) -> None:
        assert coerce_result({"output": "hello"}).output == {"output": "hello"}
        assert coerce_result({"output": [1, 2]}).output == {"output": [1, 2]}
        assert coerce_result({"output": {"a": 1}}).output == {"a": 1}
        env = coerce_result({"output": "hi", "cost_usd": 0.25})
        assert env.output == {"result": "hi"} and env.cost_usd == 0.25
        bad = coerce_result({"output": {}, "cost_usd": "lots"})
        assert bad.output == {"output": {}, "cost_usd": "lots"}
        assert coerce_result(3).output == {"result": 3}

    def test_tool_returning_output_string_runs(self, tmp_path: Path) -> None:
        gateway = ToolGateway()
        gateway.register(ToolSpec(name="h", handler=lambda: {"output": "hello"}))
        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path)),
                           approver=_approve_all())
        try:
            report = agent.run("g", {"goal": "g", "steps": [{"id": "h", "ref": "h"}]})
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert report.outcomes[0].output == {"output": "hello"}


# ---------------------------------------------------------------------------
# 5. Planner references
# ---------------------------------------------------------------------------


class TestPlannerRefs:
    def test_literal_dollar_strings_and_escape(self) -> None:
        builder = PlanBuilder()
        plan = builder.build({"goal": "g", "steps": [{"id": "a", "ref": "t", "inputs": {
            "price": "$10", "env": "$HOME", "esc": "$$a.x", "bare": "$"}}]})
        assert builder.step_inputs(plan, "a", {}) == {
            "price": "$10", "env": "$HOME", "esc": "$a.x", "bare": "$"}

    def test_ref_adds_implicit_dependency(self) -> None:
        builder = PlanBuilder()
        plan = builder.build({"goal": "g", "steps": [
            {"id": "fetch", "ref": "t"},
            {"id": "use", "ref": "t", "inputs": {"x": {"deep": ["$fetch.items.0"]}}}]})
        assert plan.steps[1].depends_on == ["fetch"]
        assert builder.order(plan) == ["fetch", "use"]
        assert builder.step_inputs(plan, "use", {"fetch": {"items": [7]}}) == {
            "x": {"deep": [7]}}

    def test_self_reference_and_cycle_rejected(self) -> None:
        with pytest.raises(ValueError, match="its own output"):
            PlanBuilder().build({"goal": "g", "steps": [
                {"id": "a", "ref": "t", "inputs": {"x": "$a.v"}}]})
        with pytest.raises(ValueError, match="cycle"):
            PlanBuilder().build({"goal": "g", "steps": [
                {"id": "a", "ref": "t", "inputs": {"x": "$b"}},
                {"id": "b", "ref": "t", "depends_on": ["a"]}]})

    def test_producer_runs_first_even_if_listed_later(self, tmp_path: Path) -> None:
        gateway = ToolGateway()
        gateway.register(ToolSpec(name="make", handler=lambda: {"v": 5}))
        gateway.register(ToolSpec(name="use", handler=lambda x: {"got": x}))
        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path)),
                           approver=_approve_all())
        try:
            report = agent.run("g", {"goal": "g", "steps": [
                {"id": "u", "ref": "use", "inputs": {"x": "$z.v"}},
                {"id": "z", "ref": "make"}]})
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert report.outcomes[-1].output == {"got": 5}


# ---------------------------------------------------------------------------
# 6. as_handler inputs
# ---------------------------------------------------------------------------


class TestAsHandler:
    def test_nested_run_receives_inputs(self, tmp_path: Path) -> None:
        sub_gateway = ToolGateway()
        seen: list[dict[str, Any]] = []

        @sub_gateway.tool("echo", parameters={"type": "object"})
        def echo(**kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs)
            return {"echoed": kwargs["v"]}

        sub = SuperAgent(sub_gateway, recorder=VCRRecorder(output_dir=str(tmp_path / "sub")),
                         approver=_approve_all())
        handler = sub.as_handler({"goal": "s", "steps": [
            {"id": "e", "ref": "echo", "inputs": {"v": "$inputs.v"}}]})
        try:
            as_agent = handler({"v": 1})
            as_tool = handler(v=2)
        finally:
            sub.shutdown()
        assert seen == [{"v": 1}, {"v": 2}]
        assert as_agent.output["echoed"] == 1 and as_tool.output["_run_status"] == "committed"
        assert as_tool.output["_sub_session_id"]


# ---------------------------------------------------------------------------
# 7-9. Sandbox
# ---------------------------------------------------------------------------


class TestSandboxFactory:
    def test_aliases_and_case(self, tmp_path: Path) -> None:
        sb = create_sandbox("fallback", workdir=tmp_path / "a")
        assert isinstance(sb, LocalSandbox) and sb.config.provider == "local"
        e2b = create_sandbox("E2B", client=object())
        assert isinstance(e2b, E2BSandbox) and e2b.config.provider == "e2b"
        assert isinstance(create_sandbox(" Local ", workdir=tmp_path / "b"), LocalSandbox)
        with pytest.raises(ValueError, match="Unknown sandbox provider"):
            create_sandbox("docker")

    def test_config_provider_wins_over_auto(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "e2b", types.ModuleType("e2b"))
        monkeypatch.setenv("E2B_API_KEY", "test-key")
        sb = create_sandbox("auto", config=SandboxConfig(provider="local"),
                            workdir=tmp_path / "c")
        assert isinstance(sb, LocalSandbox)
        # Without an explicit config provider, auto still picks E2B.
        auto = create_sandbox("auto", client=object())
        assert isinstance(auto, E2BSandbox) and auto.config.provider == "e2b"


class TestLocalSandbox:
    def test_subdir_snapshot_diff_is_scoped(self, tmp_path: Path) -> None:
        sb = LocalSandbox(workdir=tmp_path / "ws")
        sb.start()
        sb.write_file("outside.txt", "x")
        sb.write_file("sub/keep.txt", "y")
        before = sb.snapshot_filesystem("sub")
        assert set(before.entries) == {"sub/keep.txt"}
        assert sb.diff_filesystem(before).has_changes is False
        sb.write_file("sub/new.txt", "z")
        sb.write_file("other.txt", "w")
        diff = sb.diff_filesystem(before)
        assert diff.added == ["sub/new.txt"] and not diff.deleted
        sb.stop()

    def test_cleanup_false_keeps_directory(self) -> None:
        sb = LocalSandbox(cleanup=False)
        sb.start()
        root = sb.root_path
        sb.write_file("artifact.txt", "keep me")
        sb.stop()
        del sb
        gc.collect()
        try:
            assert (root / "artifact.txt").read_text() == "keep me"
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_cleanup_true_removes_directory(self) -> None:
        sb = LocalSandbox()
        sb.start()
        root = sb.root_path
        sb.stop()
        assert not root.exists()


class _FakeCommandExit(Exception):
    def __init__(self) -> None:
        super().__init__("Command exited with code 3")
        self.stdout = "partial out"
        self.stderr = "bad things"
        self.exit_code = 3
        self.error = "bad things"


class _FakeTimeout(Exception):
    pass


class TestE2BAdapter:
    def _install_fake_sdk(self, monkeypatch: pytest.MonkeyPatch,
                          run: Any) -> dict[str, Any]:
        import agent_vcr.sandbox.e2b_adapter as adapter

        created: dict[str, Any] = {}

        class _Commands:
            def run(self, command: str, **kwargs: Any) -> Any:
                created.setdefault("run_kwargs", []).append(kwargs)
                return run(command)

        class _Instance:
            commands = _Commands()

            def kill(self) -> None:
                created["killed"] = True

        class _Sandbox:
            @classmethod
            def create(cls, **kwargs: Any) -> _Instance:
                created["create_kwargs"] = kwargs
                return _Instance()

        fake = types.SimpleNamespace(Sandbox=_Sandbox, CommandExitException=_FakeCommandExit,
                                     TimeoutException=_FakeTimeout)
        monkeypatch.setattr(adapter, "e2b", fake, raising=False)
        monkeypatch.setattr(adapter, "_E2B_IMPORT_ERROR", None)
        return created

    def test_lifetime_and_per_command_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created = self._install_fake_sdk(
            monkeypatch, lambda c: types.SimpleNamespace(stdout="ok", stderr="", exit_code=0,
                                                         error=None))
        sb = E2BSandbox(config=SandboxConfig(timeout_seconds=60, lifetime_seconds=900),
                        api_key="k")
        res = sb.execute_command("echo ok", timeout=5)
        sb.stop()
        assert res.success and res.stdout == "ok"
        assert created["create_kwargs"]["timeout"] == 900  # lifetime, not 60
        assert created["run_kwargs"][0]["timeout"] == 5

    def test_lifetime_never_shorter_than_command_timeout(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        created = self._install_fake_sdk(
            monkeypatch, lambda c: types.SimpleNamespace(stdout="", stderr="", exit_code=0))
        E2BSandbox(config=SandboxConfig(timeout_seconds=7200, lifetime_seconds=60),
                   api_key="k").start()
        assert created["create_kwargs"]["timeout"] == 7200

    def test_command_exit_exception_keeps_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(command: str) -> Any:
            raise _FakeCommandExit()

        self._install_fake_sdk(monkeypatch, boom)
        sb = E2BSandbox(api_key="k")
        res = sb.execute_command("false")
        assert (res.exit_code, res.stdout, res.stderr) == (3, "partial out", "bad things")
        assert res.error == "bad things" and not res.timed_out and not res.success

    def test_sdk_timeout_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def slow(command: str) -> Any:
            raise _FakeTimeout("deadline")

        self._install_fake_sdk(monkeypatch, slow)
        res = E2BSandbox(api_key="k").execute_command("sleep 99", timeout=1)
        assert res.timed_out and res.exit_code == -1


# ---------------------------------------------------------------------------
# 10. Vector memory
# ---------------------------------------------------------------------------


def _qdrant() -> QdrantVectorStore:
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    return QdrantVectorStore(client=QdrantClient(location=":memory:"), collection_name="t")


class TestVectorMemory:
    def test_reserved_keys_win_qdrant(self) -> None:
        store = _qdrant()
        rid = store.add("real text", metadata={"namespace": "evil", "text": "overwritten",
                                               "tag": "ok"}, namespace="ns1")
        rec = store.get(rid)
        assert rec is not None
        assert rec.text == "real text" and rec.namespace == "ns1"
        assert rec.metadata == {"tag": "ok"}

    def test_qdrant_count_clear_delete_namespaced(self) -> None:
        store = _qdrant()
        a = store.add("alpha one", namespace="a")
        store.add("alpha two", namespace="a")
        store.add("beta", namespace="b")
        assert (store.count("a"), store.count("b"), store.count()) == (2, 1, 3)
        assert store.delete(a, namespace="b") is False
        assert store.delete(a) is True
        assert store.delete(a) is False  # already gone
        assert store.clear("a") == 1
        assert (store.count("a"), store.count("b")) == (0, 1)

    def test_in_memory_search_excludes_unrelated(self) -> None:
        mem = InMemoryVectorStore()
        mem.add("kubernetes clusters orchestrate containers")
        assert mem.search("zzzz qqqq") == []
        assert mem.search("") == []
        assert mem.search("kubernetes clusters")
        mem._records["odd"] = mem._records[next(iter(mem._records))].model_copy(
            update={"id": "odd", "vector": [1.0, 0.0]})
        assert all(r.id != "odd" for r in mem.search("kubernetes clusters"))

    def test_qdrant_search_excludes_unrelated(self) -> None:
        store = _qdrant()
        store.add("kubernetes clusters orchestrate containers")
        assert store.search("") == []
        assert all(hit.score > 0 for hit in store.search("zzzz qqqq"))

    def test_in_memory_delete_any_namespace(self) -> None:
        mem = InMemoryVectorStore()
        rid = mem.add("x", namespace="team")
        assert mem.delete(rid, namespace="other") is False
        assert mem.delete(rid) is True

    @pytest.mark.parametrize("backend", ["memory", "qdrant"])
    def test_reindex_is_idempotent(self, tmp_path: Path, backend: str) -> None:
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("s1")
        rec.record_step("a", {"q": 1}, {"r": 2})
        rec.record_step("b", {"q": 3}, {"r": 4})
        path = rec.save()
        mem = InMemoryVectorStore() if backend == "memory" else _qdrant()
        assert mem.index_vcr_session(path, namespace="h") == 2
        assert mem.index_vcr_session(path, namespace="h") == 0
        assert mem.count("h") == 2

    def test_export_json_atomic_roundtrip(self, tmp_path: Path) -> None:
        mem = InMemoryVectorStore()
        mem.add("hello world", metadata={"k": 1})
        out = mem.export_json(tmp_path / "sub" / "mem.json")
        assert not list(out.parent.glob("*.tmp"))
        copy = InMemoryVectorStore()
        copy.load_json(out)
        assert copy.count() == 1


# ---------------------------------------------------------------------------
# 11. Failure classification
# ---------------------------------------------------------------------------


def _error_session(tmp_path: Path, error: Exception, sid: str = "e") -> VCRPlayer:
    rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
    rec.start_session(sid)
    rec.record_error("tool:x", {}, error, latency_ms=1.0)
    return VCRPlayer.load(rec.save())


class TestClassification:
    @pytest.mark.parametrize("message", [
        "Tool 'fetch' exceeded 0.2s", "Agent 'a' exceeded 120.0s", "HTTP 429 Too Many Requests",
        "upstream returned status 503", "rate limit reached", "request timed out",
        "connection reset by peer"])
    def test_transient(self, tmp_path: Path, message: str) -> None:
        kind, _ = classify_player(_error_session(tmp_path, RuntimeError(message)))
        assert kind == FailureKind.TRANSIENT

    @pytest.mark.parametrize("message", [
        "order 4290 not found", "listening on port 5030 failed: bad config",
        "invoice id 503 is malformed", "returned 429 rows, expected 430"])
    def test_numbers_in_ids_are_not_transient(self, tmp_path: Path, message: str) -> None:
        kind, _ = classify_player(_error_session(tmp_path, ValueError(message)))
        assert kind != FailureKind.TRANSIENT

    def test_timeout_error_type_is_transient(self, tmp_path: Path) -> None:
        kind, _ = classify_player(_error_session(tmp_path, ToolTimeoutError("x", 1.0)))
        assert kind == FailureKind.TRANSIENT


# ---------------------------------------------------------------------------
# 12. Routing
# ---------------------------------------------------------------------------


def _llm(directory: Path, sid: str, model: str, cost: float, err: bool = False) -> None:
    rec = VCRRecorder(output_dir=str(directory), auto_save=False)
    rec.start_session(sid)
    if err:
        rec.record_error(f"llm:{model}", {}, ValueError("boom"), latency_ms=1.0, model=model)
    else:
        rec.record_llm_call(model, [{"role": "user", "content": "hi"}], {"content": "yo"},
                            5, 5, 10.0, cost_usd=cost)
    rec.save()


class TestRouting:
    def test_always_failing_model_never_wins(self, tmp_path: Path) -> None:
        _llm(tmp_path, "a", "pricey", cost=0.5)
        _llm(tmp_path, "b", "broken", cost=0.0, err=True)
        _llm(tmp_path, "c", "broken", cost=0.0, err=True)
        registry = ModelRegistry().ingest_dir(tmp_path)
        decision = route(registry, max_avg_cost=0.01)
        assert decision.model == "pricey"
        assert decision.constraints_met is False
        assert "constraints not met" in decision.reason
        assert "broken" not in decision.candidates
        # Even with no success requirement a zero-success model is excluded.
        assert route(registry, min_success=0.0).model == "pricey"

    def test_all_models_failing_raises(self, tmp_path: Path) -> None:
        _llm(tmp_path, "b", "broken", cost=0.0, err=True)
        with pytest.raises(ValueError, match="successful"):
            route(ModelRegistry().ingest_dir(tmp_path))

    def test_constraints_met_flag_and_fallbacks(self, tmp_path: Path) -> None:
        _llm(tmp_path, "a", "cheap", cost=0.01)
        _llm(tmp_path, "b", "pricey", cost=0.5)
        decision = route(ModelRegistry().ingest_dir(tmp_path), fallbacks=["cheap", "pricey"])
        assert decision.model == "cheap" and decision.constraints_met is True
        assert decision.fallbacks == ["pricey"]


# ---------------------------------------------------------------------------
# 13. Costs
# ---------------------------------------------------------------------------


class TestCosts:
    def test_nested_subagent_cost_not_double_counted(self, tmp_path: Path) -> None:
        shared_dir = tmp_path / "vcr"
        sub_gateway = ToolGateway()

        @sub_gateway.tool("inner", parameters={"type": "object"})
        def inner() -> dict[str, Any]:
            return {"output": {"done": True}, "cost_usd": 0.3, "tokens_used": 30}

        sub = SuperAgent(sub_gateway, recorder=VCRRecorder(output_dir=str(shared_dir)),
                         approver=_approve_all())
        gateway = ToolGateway()
        gateway.register(ToolSpec(name="worker", handler=sub.as_handler(
            {"goal": "sub", "steps": [{"id": "i", "ref": "inner"}]})))
        parent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(shared_dir)),
                            approver=_approve_all())
        try:
            report = parent.run("p", {"goal": "p", "steps": [{"id": "w", "ref": "worker"}]})
        finally:
            parent.shutdown()
            sub.shutdown()
        assert report.total_cost_usd == pytest.approx(0.3)  # parent still sees its spend
        spend = breakdown(shared_dir)
        assert spend.total_cost_usd == pytest.approx(0.3)
        assert spend.total_tokens == 30
        rolled = [s for s in spend.by_step if s.get("rolled_up_into")]
        assert rolled and rolled[0]["cost_usd"] == 0.0

    def test_parent_cost_counted_when_sub_session_absent(self, tmp_path: Path) -> None:
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("parent")
        rec.record_step("tool:worker", {}, {"_sub_session_id": "elsewhere"},
                        metadata=FrameMetadata(cost_usd=0.2))
        rec.save()
        assert breakdown(tmp_path).total_cost_usd == pytest.approx(0.2)
