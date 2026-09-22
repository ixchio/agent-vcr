"""SuperAgent runtime — policy-gated DAG execution on the VCR kernel (Phase 4).

:class:`SuperAgent` executes a validated :class:`Plan` step by step:

``policy → budget → approval → execute (timeout) → verify → checkpoint``

Every transition is recorded as a VCR frame (plan, policy decisions,
approvals, tool/agent outputs, verifications, checkpoints), so any run
can be inspected, forked, replayed, bundled, and CI-gated with the
Phase 0–2 machinery. Sub-agents are plain callables — including
``other_agent.as_handler(sub_plan)`` for real nesting.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from agent_vcr.intelligence.recovery import plan_recovery
from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.superagent.approval import ApprovalRequest, Approver, AutoApprover
from agent_vcr.superagent.memory import MemoryStore, redact_value
from agent_vcr.superagent.models import (
    Plan,
    RunReport,
    StepOutcome,
    StepResult,
    coerce_result,
)
from agent_vcr.superagent.planner import PlanBuilder
from agent_vcr.superagent.policy import BudgetTracker, Policy
from agent_vcr.superagent.router import ToolGateway
from agent_vcr.superagent.verifier import Verifier

logger = logging.getLogger(__name__)

AgentHandler = Callable[[dict[str, Any]], Any]


class SuperAgent:
    """Policy-gated executor binding a plan to tools, agents, and VCR."""

    def __init__(
        self,
        gateway: ToolGateway | None = None,
        *,
        agents: dict[str, AgentHandler] | None = None,
        memory: MemoryStore | None = None,
        verifier: Verifier | None = None,
        policy: Policy | None = None,
        budget: BudgetTracker | None = None,
        approver: Approver | None = None,
        recorder: VCRRecorder | None = None,
        builder: PlanBuilder | None = None,
        recovery: bool = True,
        max_recovery_attempts: int = 2,
        mcp_servers: list[str] | None = None,
    ):
        self.gateway = gateway or ToolGateway()
        if mcp_servers:
            for url in mcp_servers:
                self.gateway.discover_mcp(url)
        self.agents = agents or {}
        self.memory = memory or MemoryStore()
        self.verifier = verifier or Verifier()
        self.policy = policy or Policy()
        self.budget = budget or BudgetTracker()
        self.approver = approver or AutoApprover()
        self.recorder = recorder or VCRRecorder()
        self.builder = builder or PlanBuilder(
            known_tools=set(self.gateway.names()),
            known_agents=set(self.agents),
        )
        self._recovery_enabled = recovery
        self._max_recovery_attempts = max_recovery_attempts
        self._exec_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vcr-agent")

    # -- public ----------------------------------------------------------

    def run(
        self,
        goal: str,
        plan: Plan | dict[str, Any],
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> RunReport:
        """Execute a plan to completion (or abort). Always saves the session."""
        executable = self.builder.build(plan, goal=goal)
        order = self.builder.order(executable)
        self.budget.reset_usage()
        if executable.max_total_cost_usd is not None and self.budget.max_total_cost_usd is None:
            self.budget.max_total_cost_usd = executable.max_total_cost_usd
        session = self.recorder.start_session(
            session_id=session_id,
            metadata={"goal": goal, "plan_version": executable.version,
                      "superagent": "1.0", "steps": len(executable.steps)},
            tags=["superagent", *(tags or [])],
        )
        self.recorder.record_step(
            node_name="plan",
            input_state={"goal": goal},
            output_state=executable.model_dump(),
            metadata=FrameMetadata(custom={"plan_version": executable.version}),
        )
        report = RunReport(session_id=session.session_id, goal=goal)
        outputs: dict[str, dict[str, Any]] = {}
        failed_ids: set[str] = set()
        recovery_attempts = 0

        for index, step_id in enumerate(order):
            step = next(s for s in executable.steps if s.id == step_id)
            outcome = self._run_step(executable, step_id, index, outputs, failed_ids)
            report.outcomes.append(outcome)
            if outcome.status == "ok":
                outputs[step_id] = outcome.output
            elif step.on_failure == "skip":
                outputs[step_id] = {}
                failed_ids.add(step_id)
            else:  # abort — try recovery first
                failed_ids.add(step_id)
                if (
                    self._recovery_enabled
                    and recovery_attempts < self._max_recovery_attempts
                ):
                    self.recorder.save()
                    plan_result = self._attempt_recovery()
                    if plan_result and plan_result.actions and plan_result.actions[0].auto:
                        action = plan_result.actions[0]
                        recovery_attempts += 1
                        self.recorder.record_step(
                            node_name=f"recovery:{step_id}",
                            input_state={
                                "failure_kind": plan_result.failure_kind.value,
                                "attempt": recovery_attempts,
                            },
                            output_state={
                                "action": action.action,
                                "reason": action.reason,
                            },
                        )
                        if action.action == "retry":
                            failed_ids.discard(step_id)
                            outcome = self._run_step(
                                executable, step_id, index, outputs, failed_ids,
                            )
                            report.outcomes.append(outcome)
                            if outcome.status == "ok":
                                outputs[step_id] = outcome.output
                                continue
                        # Recovery didn't help or action isn't retry
                report.status = "aborted"
                break

        totals = self.recorder.get_session()
        if totals is not None:
            report.total_tokens = totals.total_tokens
            report.total_cost_usd = totals.total_cost_usd
        report.total_latency_ms = self.budget.latency_ms
        self.recorder.save()
        logger.info("SuperAgent run %s: %s (%d steps)",
                    report.session_id, report.status, len(report.outcomes))
        return report

    def as_handler(self, plan: Plan | dict[str, Any], goal: str = "") -> AgentHandler:
        """Expose this agent (with a fixed sub-plan) as a handler for nesting.

        Accepts either a single inputs dict (agent convention) or keyword
        arguments (tool convention), so the handler works registered as a
        tool or wired directly as a sub-agent.
        """

        def handle(inputs: dict[str, Any] | None = None, **kwargs: Any) -> StepResult:
            merged_inputs = dict(inputs or {})
            merged_inputs.update(kwargs)
            sub_report = self.run(goal or "sub-task", plan)
            merged: dict[str, Any] = {}
            for outcome in sub_report.outcomes:
                if outcome.status == "ok":
                    merged.update(outcome.output)
            merged["_run_status"] = sub_report.status
            if sub_report.status != "committed":
                return StepResult(output=merged, error=f"sub-agent {sub_report.status}",
                                  tokens_used=sub_report.total_tokens,
                                  cost_usd=sub_report.total_cost_usd)
            return StepResult(output=merged, tokens_used=sub_report.total_tokens,
                              cost_usd=sub_report.total_cost_usd)

        return handle

    def shutdown(self) -> None:
        self.gateway.shutdown()
        self._exec_pool.shutdown(wait=True, cancel_futures=True)

    # -- internals ---------------------------------------------------------

    def _attempt_recovery(self) -> Any:
        """Classify failure and plan recovery from the current recording."""
        try:
            player = VCRPlayer.load(self.recorder._get_session_path())
            has_cp = any(
                f.frame_type == FrameType.CHECKPOINT for f in player.frames
            )
            return plan_recovery(player, has_checkpoint=has_cp)
        except Exception:
            logger.warning("Recovery planning failed", exc_info=True)
            return None

    def _run_step(self, plan: Plan, step_id: str, index: int,
                  outputs: dict[str, dict[str, Any]],
                  failed_ids: set[str]) -> StepOutcome:
        from agent_vcr.superagent.models import PlanStep

        step: PlanStep = next(s for s in plan.steps if s.id == step_id)

        if any(dep in failed_ids for dep in step.depends_on):
            self._record("policy", step_id, {"decision": "blocked"},
                         {"reason": "upstream step failed/skipped"})
            return StepOutcome(step_id=step_id, status="blocked",
                               reasons=["upstream step failed or skipped"])

        try:
            inputs = self.builder.step_inputs(plan, step_id, outputs)
        except KeyError as err:
            self._record("policy", step_id, {"decision": "blocked"}, {"reason": str(err)})
            return StepOutcome(step_id=step_id, status="blocked", reasons=[str(err)])

        decision, reason = self.policy.decide(
            step.ref, step.kind.value, step.id, index,
            tool_requires_approval=(
                step.kind.value == "tool" and step.ref in self.gateway
                and self.gateway.get(step.ref).requires_approval
            ),
        )
        self._record("policy", step_id, {"inputs": redact_value(inputs)},
                     {"decision": decision, "reason": reason})
        if decision == "deny":
            if step.on_failure == "skip":
                return StepOutcome(step_id=step_id, status="skipped", reasons=[reason])
            return StepOutcome(step_id=step_id, status="blocked", reasons=[reason])

        ok, budget_reason = self.budget.check(step.max_cost_usd)
        if not ok:
            self._record("budget", step_id, {}, {"blocked": budget_reason})
            return StepOutcome(step_id=step_id, status="blocked", reasons=[budget_reason])

        if decision == "require_approval" or step.requires_approval:
            approved = self._approve(step_id, step, inputs, reason)
            if not approved[0]:
                self._record("approval", step_id, {"inputs": redact_value(inputs)},
                             {"approved": False, "reason": approved[1]})
                if step.on_failure == "skip":
                    return StepOutcome(step_id=step_id, status="skipped",
                                       reasons=[approved[1]])
                return StepOutcome(step_id=step_id, status="denied", reasons=[approved[1]])

        attempts = 1 + max(0, step.max_retries)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                result = self._execute(step, inputs)
            except Exception as err:  # noqa: BLE001 — handler errors become outcomes
                last_error = f"{type(err).__name__}: {err}"
                self.recorder.record_error(
                    node_name=f"{step.kind.value}:{step.ref}",
                    input_state=redact_value(inputs),
                    error=err if isinstance(err, Exception) else RuntimeError(str(err)),
                    latency_ms=0.0,
                )
                if attempt < attempts:
                    continue
                return self._fail_or_skip(step, [last_error], attempt)
            verdict = self.verifier.verify(step.check, result)
            self._record("verify", step_id, redact_value(result.output),
                         {"passed": verdict.passed, "reasons": verdict.reasons,
                          "attempt": attempt})
            if verdict.passed:
                return self._commit_step(step, inputs, result, attempt)
            last_error = "; ".join(verdict.reasons)
            if attempt < attempts:
                continue
            return self._fail_or_skip(step, [last_error], attempt)
        return self._fail_or_skip(step, [last_error or "unknown failure"], attempts)

    def _execute(self, step: Any, inputs: dict[str, Any]) -> StepResult:
        if step.kind.value == "tool":
            return self.gateway.call(step.ref, inputs, timeout_s=step.timeout_s)
        handler = self.agents.get(step.ref)
        if handler is None:
            raise KeyError(f"Unknown agent {step.ref!r}")
        future = self._exec_pool.submit(handler, inputs)
        try:
            return coerce_result(future.result(timeout=step.timeout_s))
        except concurrent.futures.TimeoutError as err:
            future.cancel()
            raise TimeoutError(f"Agent {step.ref!r} exceeded {step.timeout_s}s") from err

    def _approve(self, step_id: str, step: Any, inputs: dict[str, Any],
                 policy_reason: str) -> tuple[bool, str]:
        request = ApprovalRequest(
            step_id=step_id, ref=step.ref, kind=step.kind.value,
            reason=policy_reason or "step requires approval",
            inputs=redact_value(inputs), cost_estimate_usd=step.max_cost_usd,
        )
        decision = self.approver.decide(request)
        self._record("approval", step_id, {"inputs": redact_value(inputs)},
                     {"approved": decision.approved, "by": decision.decided_by,
                      "reason": decision.note or policy_reason})
        return decision.approved, decision.note or policy_reason

    def _commit_step(self, step: Any, inputs: dict[str, Any],
                     result: StepResult, attempt: int) -> StepOutcome:
        node = f"{step.kind.value}:{step.ref}"
        self.recorder.record_step(
            node_name=node,
            input_state=redact_value(inputs),
            output_state=result.output,
            metadata=FrameMetadata(
                latency_ms=result.latency_ms, tokens_used=result.tokens_used or None,
                cost_usd=result.cost_usd or None,
                error_type="StepError" if result.error else None,
                error_message=result.error,
                custom={"step_id": step.id, "attempt": attempt},
            ),
            frame_type=FrameType.TOOL_CALL if step.kind.value == "tool" else FrameType.NODE_EXECUTION,
        )
        if result.error:
            return self._fail_or_skip(step, [result.error], attempt)
        self.memory.put(step.id, result.output, namespace=step.id)
        self.budget.account(tokens=result.tokens_used, cost_usd=result.cost_usd,
                            latency_ms=result.latency_ms)
        if step.checkpoint_after:
            state = dict(result.output)
            self.recorder.record_checkpoint(state, label=step.id)
        return StepOutcome(step_id=step.id, status="ok", attempts=attempt,
                           output=result.output)

    def _fail_or_skip(self, step: Any, reasons: list[str], attempt: int) -> StepOutcome:
        if step.on_failure == "skip":
            return StepOutcome(step_id=step.id, status="skipped",
                               attempts=attempt, reasons=reasons)
        return StepOutcome(step_id=step.id, status="failed",
                           attempts=attempt, reasons=reasons)

    def _record(self, prefix: str, step_id: str, inputs: dict[str, Any],
                output: dict[str, Any]) -> None:
        start = time.perf_counter()
        self.recorder.record_step(
            node_name=f"{prefix}:{step_id}",
            input_state=inputs,
            output_state=output,
            metadata=FrameMetadata(latency_ms=(time.perf_counter() - start) * 1000),
        )
