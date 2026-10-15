"""Phase 4 tests: planner, gateway, memory, verifier, policy, approval, agent.

All handlers here are real functions (in-memory tools, flaky counters,
sleepers) — no mocks, no network. The LLMPlanner transport is injected
with canned *response objects* only to test validation/repair; the live
litellm path is skipped without credentials.
"""

from __future__ import annotations

import json
import time

import pytest

import agent_vcr
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.superagent import (
    AutoApprover,
    BudgetTracker,
    CallbackApprover,
    MemoryStore,
    PlanBuilder,
    Policy,
    StepResult,
    SuperAgent,
    ToolGateway,
    ToolSpec,
    Verifier,
    redact_text,
    validate_against_schema,
)
from agent_vcr.superagent.planner import LLMPlanner


def _agent(tmp_path, **kwargs):
    gateway = ToolGateway()

    @gateway.tool("classify", parameters={"type": "object"})
    def classify(text: str = "") -> dict:
        return {"label": "refund" if "refund" in text else "other"}

    @gateway.tool("lookup", parameters={"type": "object"})
    def lookup(order_id: str = "") -> dict:
        return {"order_id": order_id, "total": 42.0}

    @gateway.tool("refund", parameters={"type": "object"}, requires_approval=True)
    def refund(order_id: str = "", amount: float = 0.0) -> dict:
        return {"refunded": True, "order_id": order_id, "amount": amount}

    kwargs.setdefault("recorder", VCRRecorder(output_dir=str(tmp_path / "vcr")))
    kwargs.setdefault("approver", CallbackApprover(lambda req: True, name="test-yes"))
    return SuperAgent(gateway, **kwargs), gateway


REFUND_PLAN = {
    "goal": "refund order",
    "steps": [
        {"id": "classify", "kind": "tool", "ref": "classify",
         "inputs": {"text": "please refund my order"},
         "check": {"contains": {"label": "refund"}}},
        {"id": "lookup", "kind": "tool", "ref": "lookup",
         "inputs": {"order_id": "ord-1"}, "depends_on": ["classify"]},
        {"id": "pay", "kind": "tool", "ref": "refund",
         "inputs": {"order_id": "$lookup.order_id", "amount": "$lookup.total"},
         "depends_on": ["lookup"], "checkpoint_after": True,
         "check": {"contains": {"refunded": True}}},
    ],
}


class TestPlanBuilder:
    def test_valid_plan_and_order(self):
        builder = PlanBuilder(known_tools={"a", "b"})
        plan = builder.build({"goal": "g", "steps": [
            {"id": "one", "kind": "tool", "ref": "a"},
            {"id": "two", "kind": "tool", "ref": "b", "depends_on": ["one"]},
        ]})
        assert builder.order(plan) == ["one", "two"]

    def test_rejects(self):
        builder = PlanBuilder(known_tools={"a"})
        with pytest.raises(ValueError, match="no steps"):
            builder.build({"goal": "g", "steps": []})
        with pytest.raises(ValueError, match="Duplicate"):
            builder.build({"goal": "g", "steps": [
                {"id": "x", "kind": "tool", "ref": "a"},
                {"id": "x", "kind": "tool", "ref": "a"}]})
        with pytest.raises(ValueError, match="unknown tool"):
            builder.build({"goal": "g", "steps": [{"id": "x", "kind": "tool", "ref": "zzz"}]})
        with pytest.raises(ValueError, match="unknown"):
            builder.build({"goal": "g", "steps": [
                {"id": "x", "kind": "tool", "ref": "a", "depends_on": ["ghost"]}]})
        with pytest.raises(ValueError, match="cycle"):
            builder.build({"goal": "g", "steps": [
                {"id": "x", "kind": "tool", "ref": "a", "depends_on": ["y"]},
                {"id": "y", "kind": "tool", "ref": "a", "depends_on": ["x"]}]})

    def test_input_refs(self):
        builder = PlanBuilder()
        plan = builder.build({"goal": "g", "steps": [
            {"id": "a", "kind": "tool", "ref": "t", "inputs": {}},
            {"id": "b", "kind": "tool", "ref": "t",
             "inputs": {"x": "$a.user.name", "ys": ["$a.n", 1], "lit": 5}},
        ]})
        resolved = builder.step_inputs(plan, "b", {"a": {"user": {"name": "zo"}, "n": 7}})
        assert resolved == {"x": "zo", "ys": [7, 1], "lit": 5}
        with pytest.raises(KeyError):
            builder.step_inputs(plan, "b", {"a": {"user": {}}})


class TestLLMPlanner:
    def _resp(self, text):
        class Msg:
            content = text

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    def test_parse_json(self):
        assert LLMPlanner.parse_json('{"steps": []}') == {"steps": []}
        with pytest.raises(ValueError):
            LLMPlanner.parse_json("not json")
        with pytest.raises(ValueError):
            LLMPlanner.parse_json('{"goal": "x"}')

    def test_repair_then_success(self):
        calls = []

        def fake_complete(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return self._resp('{"goal": "g"}')  # invalid: no steps
            return self._resp(json.dumps({
                "goal": "g",
                "steps": [{"id": "s", "kind": "tool", "ref": "t"}],
            }))

        planner = LLMPlanner("test-model", complete_fn=fake_complete)
        plan = planner.plan("g")
        assert [s.id for s in plan.steps] == ["s"]
        assert len(calls) == 2
        assert calls[0]["model"] == "test-model"

    def test_gives_up(self):
        planner = LLMPlanner("m", max_repair_attempts=0,
                             complete_fn=lambda **k: self._resp('{"nope": 1}'))
        with pytest.raises(ValueError, match="no valid plan"):
            planner.plan("g")


class TestToolGateway:
    def test_call_validate_unknown_timeout(self):
        gw = ToolGateway()

        @gw.tool("add", parameters={"type": "object", "required": ["a", "b"],
                                    "properties": {"a": {"type": "integer"},
                                                   "b": {"type": "integer"}}})
        def add(a: int = 0, b: int = 0) -> dict:
            return {"sum": a + b}

        assert gw.call("add", {"a": 2, "b": 3}).output == {"sum": 5}
        with pytest.raises(ValueError, match="invalid params"):
            gw.call("add", {"a": "x", "b": 1})
        with pytest.raises(ValueError, match="required"):
            gw.call("add", {"a": 1})
        with pytest.raises(KeyError):
            gw.call("nope", {})
        with pytest.raises(ValueError, match="already registered"):
            gw.register(ToolSpec(name="add", handler=lambda: {}))

        @gw.tool("slow", parameters={"type": "object"})
        def slow() -> dict:
            time.sleep(5)
            return {}

        with pytest.raises(TimeoutError):
            gw.call("slow", {}, timeout_s=0.1)
        gw.shutdown()

    def test_schema_validator(self):
        assert validate_against_schema({"a": 1}, {"type": "object", "required": ["a"]}) == []
        assert validate_against_schema({}, {"type": "object", "required": ["a"]})
        assert validate_against_schema({"c": "z"}, {"type": "object",
                                        "properties": {"c": {"enum": ["y"]}}})
        assert validate_against_schema([1, "x"], {"type": "array",
                                                  "items": {"type": "integer"}})
        assert validate_against_schema(True, {"type": "integer"})


class TestMemory:
    def test_namespaces_search_redaction_persist(self, tmp_path):
        mem = MemoryStore()
        mem.put("goal", "refund")
        mem.put("api_key", "api_key: sk-abcdef123456", namespace="lookup")
        mem.put("note", "nothing secret", namespace="lookup")
        assert mem.get("goal") == "refund"
        assert mem.get("missing", "dflt") == "dflt"
        assert mem.snapshot("lookup")["note"] == "nothing secret"
        hits = mem.search("sk-abcdef")
        assert hits and all("[REDACTED]" in json.dumps(v) for _, v in hits)
        assert "sk-abcdef" not in json.dumps(mem.redacted())
        path = mem.save(tmp_path / "mem.json")
        assert "sk-abcdef" not in path.read_text()
        assert MemoryStore.load(path).get("goal") == "refund"

    def test_redact_text(self):
        assert redact_text("password: hunter2 unfolded") == "[REDACTED] unfolded"
        assert redact_text("clean text") == "clean text"


class TestVerifierPolicyApproval:
    def test_verifier_checks(self):
        v = Verifier()
        ok = StepResult(output={"a": 1, "b": 2})
        assert v.verify({}, ok)
        assert v.verify({"equals": {"a": 1, "b": 2}}, ok)
        assert not v.verify({"equals": {"a": 9}}, ok)
        assert v.verify({"contains": {"a": 1}}, ok)
        assert not v.verify({"contains": {"z": 1}}, ok)
        assert v.verify({"required_keys": ["a"], "forbid_keys": ["z"]}, ok)
        assert not v.verify({"required_keys": ["z"]}, ok)
        assert not v.verify({"forbid_keys": ["a"]}, ok)
        assert not v.verify({"no_error": True}, StepResult(output={}, error="x"))
        assert not v.verify({}, StepResult(output={}, error="x"))

    def test_policy_decisions(self):
        p = Policy(forbid_tools=["rm"], approval_required=["pay"],
                   irreversible_refs=["nuke"], max_steps=2)
        assert p.decide("ok", "tool", "s1", 0) == ("allow", "")
        assert p.decide("rm", "tool", "s1", 0)[0] == "deny"
        assert p.decide("pay", "tool", "s1", 0)[0] == "require_approval"
        assert p.decide("nuke", "tool", "s9", 0)[0] == "require_approval"
        assert p.decide("ok", "tool", "s1", 5)[0] == "deny"
        assert p.decide("t", "tool", "s1", 0, tool_requires_approval=True)[0] == "require_approval"

    def test_budget(self):
        b = BudgetTracker(max_total_cost_usd=1.0, max_total_tokens=100)
        assert b.check()[0] is True
        b.account(tokens=60, cost_usd=0.9)
        assert b.check()[0] is True
        assert b.check(step_max_cost=0.2)[0] is False  # would breach
        b.account(tokens=50, cost_usd=0.2)
        assert b.check()[0] is False
        assert b.summary()["tokens_used"] == 110
        b.reset_usage()
        assert b.summary() == {"tokens_used": 0, "cost_usd": 0.0, "latency_ms": 0.0}

    def test_approvers(self):
        auto = AutoApprover(allow_refs=["cheap"], max_auto_cost_usd=1.0)
        from agent_vcr.superagent.approval import ApprovalRequest

        assert auto.decide(ApprovalRequest("s", "cheap", "tool", "r")).approved
        assert not auto.decide(ApprovalRequest("s", "other", "tool", "r")).approved
        assert not auto.decide(ApprovalRequest("s", "cheap", "tool", "r",
                                               cost_estimate_usd=5.0)).approved
        human = CallbackApprover(lambda req: req.ref == "yes")
        assert human.decide(ApprovalRequest("s", "yes", "tool", "r")).approved
        assert not human.decide(ApprovalRequest("s", "no", "tool", "r")).approved
        boom = CallbackApprover(lambda req: 1 / 0)
        denied = boom.decide(ApprovalRequest("s", "x", "tool", "r"))
        assert denied.approved is False and "callback error" in denied.note


class TestSuperAgentRun:
    def test_refund_flow_commits_with_trace(self, tmp_path):
        agent, _ = _agent(tmp_path)
        try:
            report = agent.run("refund order", REFUND_PLAN)
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert [o.step_id for o in report.outcomes] == ["classify", "lookup", "pay"]
        assert all(o.status == "ok" for o in report.outcomes)
        pay_out = report.outcomes[2].output
        assert pay_out == {"refunded": True, "order_id": "ord-1", "amount": 42.0}

        player = VCRPlayer.load(tmp_path / "vcr" / f"{report.session_id}.vcr")
        nodes = player.list_nodes()
        assert "plan" in nodes and "tool:refund" in nodes
        assert "approval:pay" in nodes and "verify:pay" in nodes
        assert player.goto_checkpoint("pay")  # checkpoint_after recorded
        assert player.verify_integrity()["ok"]

    def test_policy_deny_aborts(self, tmp_path):
        agent, _ = _agent(tmp_path, policy=Policy(forbid_tools=["refund"]))
        try:
            report = agent.run("refund order", REFUND_PLAN)
        finally:
            agent.shutdown()
        assert report.status == "aborted"
        assert report.outcomes[-1].status == "blocked"

    def test_denied_approval_with_skip(self, tmp_path):
        plan = dict(REFUND_PLAN)
        plan["steps"] = [dict(s, on_failure="skip") for s in REFUND_PLAN["steps"]]
        agent, _ = _agent(tmp_path, approver=CallbackApprover(lambda req: False))
        try:
            report = agent.run("refund order", plan)
        finally:
            agent.shutdown()
        assert report.status == "committed"  # skipped, not aborted
        assert report.outcomes[2].status == "skipped"

    def test_retry_then_success(self, tmp_path):
        gateway = ToolGateway()
        calls = []

        @gateway.tool("flaky", parameters={"type": "object"})
        def flaky() -> dict:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient")
            return {"recovered": True}

        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
                           approver=CallbackApprover(lambda req: True))
        try:
            report = agent.run("flaky task", {"goal": "flaky task", "steps": [
                {"id": "f", "kind": "tool", "ref": "flaky", "max_retries": 1}]})
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert report.outcomes[0].attempts == 2
        assert len(calls) == 2

    def test_budget_abort_and_timeout(self, tmp_path):
        gateway = ToolGateway()

        @gateway.tool("pricey", parameters={"type": "object"})
        def pricey() -> dict:
            return {"output": {}, "cost_usd": 5.0}

        @gateway.tool("slow", parameters={"type": "object"})
        def slow() -> dict:
            time.sleep(5)
            return {}

        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
                           budget=BudgetTracker(max_total_cost_usd=1.0),
                           approver=CallbackApprover(lambda req: True))
        try:
            report = agent.run("t", {"goal": "t", "steps": [
                {"id": "p", "kind": "tool", "ref": "pricey"},
                {"id": "s", "kind": "tool", "ref": "slow", "timeout_s": 0.2}]})
        finally:
            agent.shutdown()
        # pricey costs $5 but budget check happens BEFORE with step cap unset;
        # run records then... first step ok ($5 over cap) — second blocked pre-step
        assert report.outcomes[0].status == "ok"
        assert report.outcomes[1].status == "blocked"
        assert "cap" in report.outcomes[1].reasons[0]

    def test_reusable_agent_resets_execution_budget(self, tmp_path):
        gateway = ToolGateway()

        @gateway.tool("metered", parameters={"type": "object"})
        def metered() -> dict:
            return {"output": {"ok": True}, "cost_usd": 0.8}

        agent = SuperAgent(
            gateway,
            recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
            budget=BudgetTracker(max_total_cost_usd=1.0),
            approver=CallbackApprover(lambda req: True),
        )
        plan = {"goal": "g", "steps": [{"id": "run", "ref": "metered"}]}
        try:
            assert agent.run("g", plan).status == "committed"
            assert agent.run("g", plan).status == "committed"
        finally:
            agent.shutdown()

    def test_nested_subagent(self, tmp_path):
        gateway = ToolGateway()
        sub_gateway = ToolGateway()

        @sub_gateway.tool("inner", parameters={"type": "object"})
        def inner() -> dict:
            return {"inner_done": True}

        sub = SuperAgent(sub_gateway,
                         recorder=VCRRecorder(output_dir=str(tmp_path / "sub")),
                         approver=CallbackApprover(lambda req: True))
        gateway.register(ToolSpec(name="worker", handler=sub.as_handler(
            {"goal": "sub", "steps": [{"id": "i", "kind": "tool", "ref": "inner"}]}),
            parameters={"type": "object"}))
        agent = SuperAgent(gateway, recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
                           approver=CallbackApprover(lambda req: True))
        try:
            report = agent.run("nested", {"goal": "nested", "steps": [
                {"id": "w", "kind": "tool", "ref": "worker"}]})
        finally:
            agent.shutdown()
            sub.shutdown()
        assert report.status == "committed"
        assert report.outcomes[0].output["inner_done"] is True

    def test_plan_model_export(self):
        assert agent_vcr.__version__ == "1.5.4"
        for name in ("SuperAgent", "Plan", "PlanBuilder", "ToolGateway",
                     "MemoryStore", "Verifier", "Policy", "BudgetTracker",
                     "AutoApprover", "CallbackApprover", "RunReport"):
            assert name in agent_vcr.__all__


class TestSuperAgentRecovery:
    def test_recovery_retries_transient_failure(self, tmp_path):
        """Recovery engine auto-retries transient failures."""
        gateway = ToolGateway()
        calls = []

        @gateway.tool("flaky_net", parameters={"type": "object"})
        def flaky_net() -> dict:
            calls.append(1)
            if len(calls) <= 1:
                raise ConnectionError("connection timed out")
            return {"done": True}

        agent = SuperAgent(
            gateway,
            recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
            approver=CallbackApprover(lambda req: True),
            recovery=True,
            max_recovery_attempts=2,
        )
        try:
            report = agent.run("net task", {"goal": "net task", "steps": [
                {"id": "f", "kind": "tool", "ref": "flaky_net"}]})
        finally:
            agent.shutdown()
        assert report.status == "committed"
        assert len(calls) == 2  # first fail + recovery retry

    def test_recovery_respects_max_attempts(self, tmp_path):
        """Recovery stops after max_recovery_attempts."""
        gateway = ToolGateway()

        @gateway.tool("always_fail", parameters={"type": "object"})
        def always_fail() -> dict:
            raise ConnectionError("always times out")

        agent = SuperAgent(
            gateway,
            recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
            approver=CallbackApprover(lambda req: True),
            recovery=True,
            max_recovery_attempts=1,
        )
        try:
            report = agent.run("fail task", {"goal": "fail", "steps": [
                {"id": "f", "kind": "tool", "ref": "always_fail"}]})
        finally:
            agent.shutdown()
        assert report.status == "aborted"

    def test_recovery_disabled(self, tmp_path):
        """When recovery=False, abort immediately without attempting recovery."""
        gateway = ToolGateway()

        @gateway.tool("fail_tool", parameters={"type": "object"})
        def fail_tool() -> dict:
            raise ConnectionError("timeout")

        agent = SuperAgent(
            gateway,
            recorder=VCRRecorder(output_dir=str(tmp_path / "vcr")),
            approver=CallbackApprover(lambda req: True),
            recovery=False,
        )
        try:
            report = agent.run("t", {"goal": "t", "steps": [
                {"id": "f", "kind": "tool", "ref": "fail_tool"}]})
        finally:
            agent.shutdown()
        assert report.status == "aborted"


class TestMCPDiscovery:
    def test_discover_mcp_registers_tools(self):
        """MCP discovery registers tools from server response."""
        import json
        from unittest.mock import MagicMock, patch

        gateway = ToolGateway()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                    {
                        "name": "calc",
                        "description": "Calculate",
                        "inputSchema": {"type": "object"},
                    },
                ],
            },
        }).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            names = gateway.discover_mcp("http://localhost:3000/mcp")

        assert names == ["search", "calc"]
        assert "search" in gateway
        assert "calc" in gateway
        spec = gateway.get("search")
        assert spec.description == "Search the web"
        assert spec.parameters["required"] == ["query"]

    def test_discover_mcp_with_prefix(self):
        """MCP discovery prefixes tool names."""
        import json
        from unittest.mock import MagicMock, patch

        gateway = ToolGateway()
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result": {"tools": [{"name": "fetch", "inputSchema": {}}]},
        }).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            names = gateway.discover_mcp(
                "http://localhost:3000/mcp", prefix="web__"
            )

        assert names == ["web__fetch"]
        assert "web__fetch" in gateway

    def test_discover_mcp_skips_duplicates(self):
        """MCP discovery skips already-registered tools."""
        import json
        from unittest.mock import MagicMock, patch

        gateway = ToolGateway()
        gateway.register(ToolSpec(name="existing", handler=lambda: {}))
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result": {"tools": [{"name": "existing", "inputSchema": {}}]},
        }).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            names = gateway.discover_mcp("http://localhost:3000/mcp")

        assert names == []  # skipped, not re-registered

    def test_mcp_handler_calls_server(self):
        """MCP handler POSTs tools/call to the server."""
        import json
        from unittest.mock import MagicMock, patch

        gateway = ToolGateway()
        # First discover
        list_resp = MagicMock()
        list_resp.read.return_value = json.dumps({
            "result": {"tools": [{"name": "add", "inputSchema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            }}]},
        }).encode()

        call_resp = MagicMock()
        call_resp.read.return_value = json.dumps(
            {"result": {"sum": 5}}
        ).encode()

        with patch("urllib.request.urlopen", side_effect=[list_resp, call_resp]):
            gateway.discover_mcp("http://localhost:3000/mcp")
            result = gateway.call("add", {"a": 2, "b": 3})

        assert result.output == {"sum": 5}


class TestSuperAgentCLI:
    def test_run_from_files(self, tmp_path, monkeypatch, capsys):
        from agent_vcr.cli import main

        monkeypatch.chdir(tmp_path)
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(REFUND_PLAN))
        handlers = tmp_path / "handlers.py"
        handlers.write_text(
            "def classify(text=''):\n"
            "    return {'label': 'refund'}\n"
            "def lookup(order_id=''):\n"
            "    return {'order_id': order_id, 'total': 10.0}\n"
            "def refund(order_id='', amount=0.0):\n"
            "    return {'refunded': True, 'order_id': order_id, 'amount': amount}\n"
            "TOOLS = {'classify': classify, 'lookup': lookup, 'refund': refund}\n"
        )
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        main(["superagent", "run", "--plan", str(plan_path),
              "--handlers", str(handlers), "--goal", "refund order"])
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "committed"
        assert len(report["outcomes"]) == 3
