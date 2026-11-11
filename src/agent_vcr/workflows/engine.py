"""Workflow execution engine — durable DAG runs on the VCR kernel (Phase 5).

Execution model
---------------
- One run = one :class:`RunRecord` (durable JSON) + one VCR session.
- After **every top-level step** the run record is persisted; a fresh
  engine on the same directories resumes via :meth:`resume` (or manual
  :meth:`step_once` loops). Durability granularity is the top-level step
  — nested bodies (branches, loops, conditions) execute atomically.
- Leaf steps reuse the Phase 4 machinery: policy → budget → approval →
  validated tool call / agent handler (timeouts) → verifier → checkpoint.
- Failures with ``on_failure=abort`` trigger saga compensation: every
  completed leaf that declared ``compensate`` is undone in reverse
  completion order, then the run is marked ABORTED. Compensations go
  through the same policy / approval / timeout path as leaves: a
  forbidden or unapproved compensation is skipped and recorded, and a
  hanging one times out instead of blocking the abort.
- Every run gets its **own** :class:`BudgetTracker` (caps copied from the
  engine's template budget, tightened by ``WorkflowDef.max_total_cost_usd``),
  so runs never leak caps or usage into each other.
- Timeouts cannot stop a running thread. A timed-out leaf is therefore
  only retried when it is declared ``idempotent=True``; otherwise the run
  fails (abort/skip) and the frame notes the attempt may still be running.
- Parallel branches run on a dedicated pool sized to the branch count;
  agent leaves run on their own daemon threads (concurrency bounded by
  ``max_workers``), so leaves never wait on their own parents.
- Cancellation is cooperative: :meth:`cancel` (or a ``<run>.cancel``
  marker in the run store) is checked between units; the run is then
  compensated and marked ABORTED with ``cancelled=True``.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hmac
import logging
import re
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, NoReturn

from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.recorder import VCRRecorder
from agent_vcr.superagent.approval import ApprovalRequest, Approver, AutoApprover
from agent_vcr.superagent.memory import redact_value
from agent_vcr.superagent.models import StepResult, coerce_result
from agent_vcr.superagent.policy import BudgetTracker, Policy
from agent_vcr.superagent.router import ToolGateway
from agent_vcr.superagent.verifier import Verifier
from agent_vcr.workflows.models import (
    CompletedCompensation,
    ConditionSpec,
    RunRecord,
    RunStatus,
    StepKind,
    WorkflowDef,
    WorkflowStep,
)
from agent_vcr.workflows.store import RunStore

logger = logging.getLogger(__name__)

AgentHandler = Callable[[dict[str, Any]], Any]
Scope = Mapping[str, Any]

_RESERVED_HEADS = frozenset({"inputs"})
_IDENT = r"[A-Za-z_][A-Za-z0-9_-]*"
_REF_HEAD = re.compile(rf"\$({_IDENT})")
_IDENT_RE = re.compile(rf"^{_IDENT}$")
_EMPTY_SCOPE: Scope = {}


class WorkflowEngine:
    """Durable executor for versioned workflow definitions."""

    def __init__(
        self,
        gateway: ToolGateway | None = None,
        *,
        agents: dict[str, AgentHandler] | None = None,
        verifier: Verifier | None = None,
        policy: Policy | None = None,
        budget: BudgetTracker | None = None,
        approver: Approver | None = None,
        run_store: RunStore | None = None,
        backend: Any | None = None,
        vcr_dir: str = ".vcr",
        max_workers: int = 8,
    ):
        self.gateway = gateway or ToolGateway()
        self.agents = agents or {}
        self.verifier = verifier or Verifier()
        self.policy = policy or Policy()
        # Template only: caps are copied into a fresh per-run tracker; never mutated.
        self.budget = budget if budget is not None else BudgetTracker()
        self.approver = approver or AutoApprover()
        self.runs = run_store or RunStore()
        self.backend = backend
        self.vcr_dir = vcr_dir
        self.max_workers = max(1, max_workers)
        self._leaf_slots = threading.BoundedSemaphore(self.max_workers)
        self._lock = threading.RLock()
        self._active: dict[str, _RunContext] = {}

    # -- lifecycle -------------------------------------------------------

    def start(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunRecord:
        """Create a run (new VCR session) and execute it to completion.

        Definition errors raise ``ValueError`` before any run is created;
        every runtime error ends the run ABORTED (with compensation).
        """
        self.validate(workflow)
        budget = self._new_budget(workflow)
        recorder = VCRRecorder(output_dir=self.vcr_dir)
        session = recorder.start_session(
            metadata={"workflow": workflow.name, "workflow_version": workflow.version},
            tags=["workflow", workflow.name, *(tags or [])],
        )
        record = self.runs.create(workflow, inputs, session_id=session.session_id)
        recorder.record_step(node_name="workflow:start",
                             input_state={"workflow": workflow.name,
                                          "inputs": redact_value(inputs or {})},
                             output_state={"run_id": record.run_id,
                                           "version": workflow.version})
        recorder.save()
        return self._drive(record, recorder, budget)

    def step_once(self, run_id: str) -> bool:
        """Execute the next pending top-level step. Returns True when complete."""
        record = self.runs.load(run_id)
        if record.status != RunStatus.RUNNING:
            return True
        recorder = self._attach(record)
        budget = self._restore_budget(record, recorder)
        self._drive(record, recorder, budget, single=True)
        return self.runs.load(run_id).status != RunStatus.RUNNING

    def resume(self, run_id: str) -> RunRecord:
        """Continue a crashed/interrupted run to completion."""
        record = self.runs.load(run_id)
        if record.status != RunStatus.RUNNING:
            return record
        recorder = self._attach(record)
        budget = self._restore_budget(record, recorder)
        return self._drive(record, recorder, budget)

    def cancel(self, run_id: str, reason: str = "Cancelled by user") -> bool:
        """Request cooperative cancellation of a RUNNING run.

        If this engine is driving the run, it stops at the next unit
        boundary, compensates and marks the run ABORTED. If no engine in
        this process is driving it, the cancellation is finalized now
        (compensate + ABORTED). A run driven by *another process* sees the
        ``.cancel`` marker at its next unit boundary. Returns False when the
        run is already terminal.
        """
        record = self.runs.load(run_id)
        if record.status != RunStatus.RUNNING:
            return False
        self.runs.request_cancel(run_id, reason)
        with self._lock:
            ctx = self._active.get(run_id)
        if ctx is not None:
            ctx.request_cancel(reason)
            return True
        recorder = self._attach(record)
        budget = self._restore_budget(record, recorder)
        self._drive(record, recorder, budget)  # sees the marker → compensate + abort
        return True

    def trigger(self, workflow: WorkflowDef, inputs: dict[str, Any] | None = None,
                secret: str = "") -> RunRecord:
        """Start via trigger contract (webhook secrets enforced when set)."""
        for trigger in workflow.triggers:
            if trigger.type == "webhook" and trigger.secret and not hmac.compare_digest(
                    trigger.secret.encode("utf-8"), (secret or "").encode("utf-8")):
                raise PermissionError("Invalid webhook secret")
        return self.start(workflow, inputs, tags=["trigger"])

    def shutdown(self, *, close_gateway: bool = True) -> None:
        """Stop engine workers and optionally retain a shared tool gateway.

        Agent leaves that timed out run on daemon threads and are abandoned
        (they cannot be interrupted); they never block shutdown.
        """
        if close_gateway:
            self.gateway.shutdown()

    # -- validation ------------------------------------------------------------

    def validate(self, workflow: WorkflowDef) -> None:
        """Reject malformed definitions up front (raises ValueError)."""
        self._check_dag(workflow)
        for step in workflow.steps:
            if step.id in _RESERVED_HEADS:
                raise ValueError(f"Step id {step.id!r} is reserved")
        self._validate_units(workflow.steps, "")

    def _validate_units(self, steps: list[WorkflowStep], prefix: str) -> None:
        seen: set[str] = set()
        for step in steps:
            path = f"{prefix}{step.id}"
            if not step.id:
                raise ValueError(f"Step under {prefix or 'workflow'!r} has an empty id")
            if step.id in seen:
                raise ValueError(f"Duplicate step id {path!r}")
            seen.add(step.id)
            if step.max_retries < 0:
                raise ValueError(f"Step {path}: max_retries must be >= 0")
            if step.timeout_s <= 0:
                raise ValueError(f"Step {path}: timeout_s must be > 0")
            if step.retry_delay_s < 0:
                raise ValueError(f"Step {path}: retry_delay_s must be >= 0")
            kind = step.kind
            if kind in (StepKind.TOOL, StepKind.AGENT):
                self._validate_target(kind.value, step.ref, f"Step {path}")
            elif step.compensate is not None:
                raise ValueError(f"Step {path}: only tool/agent steps may declare compensate")
            if step.compensate is not None:
                if step.compensate.timeout_s is not None and step.compensate.timeout_s <= 0:
                    raise ValueError(f"Step {path}: compensate.timeout_s must be > 0")
                self._validate_target(step.compensate.kind, step.compensate.ref,
                                      f"Step {path} compensation")
            if kind == StepKind.CONDITION:
                if step.condition is None:
                    raise ValueError(f"Condition step {path} has no predicate")
                self._validate_units(step.then_steps, f"{path}.")
                self._validate_units(step.else_steps, f"{path}.")
            elif kind == StepKind.PARALLEL:
                if not step.branches:
                    raise ValueError(f"Parallel step {path} has no branches")
                for name, units in step.branches.items():
                    if not name:
                        raise ValueError(f"Parallel step {path} has an unnamed branch")
                    self._validate_units(units, f"{path}.{name}.")
            elif kind == StepKind.FOR_EACH:
                if not step.for_each:
                    raise ValueError(f"Loop {path} has no for_each reference")
                if not _IDENT_RE.match(step.item_var) or step.item_var in _RESERVED_HEADS:
                    raise ValueError(f"Loop {path}: invalid item_var {step.item_var!r}")
                if step.max_iterations < 0:
                    raise ValueError(f"Loop {path}: max_iterations must be >= 0")
                self._validate_units(step.body, f"{path}[].")

    def _validate_target(self, kind: str, ref: str, label: str) -> None:
        if not ref:
            raise ValueError(f"{label}: {kind} ref is empty")
        if kind == "tool" and ref not in self.gateway:
            raise ValueError(f"{label}: unknown tool {ref!r}")
        if kind == "agent" and ref not in self.agents:
            raise ValueError(f"{label}: unknown agent {ref!r}")

    # -- driver ------------------------------------------------------------

    def _drive(self, record: RunRecord, recorder: VCRRecorder, budget: BudgetTracker,
               single: bool = False) -> RunRecord:
        order = self._topo_order(record.definition)
        ctx = _RunContext(
            run_id=record.run_id,
            outputs=dict(record.outputs),
            inputs=dict(record.inputs),
            budget=budget,
            known_steps=_collect_ids(record.definition.steps),
            loop_vars=_collect_loop_vars(record.definition.steps),
            policy_step_count=sum(
                frame.node_name.startswith(("tool:", "agent:"))
                for frame in recorder.get_frames()
            ),
        )
        with self._lock:
            if record.run_id in self._active:
                raise RuntimeError(f"Run {record.run_id!r} is already executing in this engine")
            self._active[record.run_id] = ctx
        try:
            pending_cancel = self.runs.cancel_reason(record.run_id)
            if pending_cancel is not None:
                ctx.request_cancel(pending_cancel)
                self._abort(record, recorder, ctx, f"Run cancelled: {pending_cancel}")
                return record
            for step_id in order:
                if step_id in record.completed_top_steps:
                    continue
                step = next(s for s in record.definition.steps if s.id == step_id)
                ok = self._execute_top(step, record, recorder, ctx)
                with ctx.lock:
                    record.outputs.update(ctx.outputs)
                if ok:
                    record.completed_top_steps.append(step_id)
                    record.usage = ctx.budget.summary()
                    self.runs.persist(record)
                    recorder.save()
                if single or not ok:
                    break
            if len(record.completed_top_steps) == len(record.definition.steps) \
                    and record.status == RunStatus.RUNNING:
                record.status = RunStatus.COMMITTED
                record.usage = ctx.budget.summary()
                self.runs.persist(record)
                self.runs.clear_cancel(record.run_id)
                recorder.save()
            return record
        finally:
            with self._lock:
                self._active.pop(record.run_id, None)

    def _execute_top(self, step: WorkflowStep, record: RunRecord,
                     recorder: VCRRecorder, ctx: _RunContext) -> bool:
        if record.error and step.id not in record.completed_top_steps:
            # Resume after a crash during abort handling: the leaf already
            # ran — never re-execute it, go straight to compensation.
            self._abort(record, recorder, ctx, record.error)
            return False
        try:
            self._execute_unit(step, step.id, record, recorder, ctx, _EMPTY_SCOPE)
            return True
        except _SkipStep as err:
            with ctx.lock:
                ctx.outputs[err.path] = {"skipped": True, "reason": err.reason}
            logger.info("Step %s skipped: %s", err.path, err)
            return True
        except _AbortRun as err:
            message = str(err)
        except KeyError as err:
            message = (f"Step {step.id}: unresolvable input {err} "
                       f"(upstream step skipped or empty?)")
        except Exception as err:  # noqa: BLE001 — any step error aborts the run cleanly
            logger.warning("Step %s raised %s", step.id, err, exc_info=True)
            message = f"Step {step.id}: {type(err).__name__}: {err}"
        if ctx.cancel_event.is_set():
            message = f"Run cancelled: {ctx.cancel_reason} (during step {step.id}; {message})"
        self._abort(record, recorder, ctx, message)
        return False

    def _abort(self, record: RunRecord, recorder: VCRRecorder, ctx: _RunContext,
               message: str) -> None:
        record.error = message
        if ctx.cancel_event.is_set():
            record.cancelled = True
        self.runs.persist(record)
        self._compensate(record, recorder, ctx)
        record.status = RunStatus.ABORTED
        record.usage = ctx.budget.summary()
        self.runs.persist(record)
        self.runs.clear_cancel(record.run_id)
        recorder.save()

    # -- units ---------------------------------------------------------------

    def _execute_unit(self, step: WorkflowStep, path: str, record: RunRecord,
                      recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        self._check_cancel(ctx)
        kind = step.kind
        if kind in (StepKind.TOOL, StepKind.AGENT, StepKind.APPROVAL):
            try:
                if kind == StepKind.APPROVAL:
                    self._execute_approval(step, path, recorder, ctx, scope)
                else:
                    self._execute_leaf(step, path, record, recorder, ctx, scope)
            except _SkipStep as err:
                with ctx.lock:
                    ctx.outputs[path] = {"skipped": True, "reason": err.reason}
                self._frame(recorder, f"skip:{path}", {}, {"skipped": True,
                                                            "reason": err.reason})
                logger.info("Step %s skipped: %s", path, err.reason)
        elif kind == StepKind.CHECKPOINT:
            with ctx.lock:
                state = dict(ctx.outputs)
            recorder.record_checkpoint(state, label=step.ref or path)
        elif kind == StepKind.CONDITION:
            self._execute_condition(step, path, record, recorder, ctx, scope)
        elif kind == StepKind.PARALLEL:
            self._execute_parallel(step, path, record, recorder, ctx, scope)
        elif kind == StepKind.FOR_EACH:
            self._execute_foreach(step, path, record, recorder, ctx, scope)
        else:
            raise _AbortRun(f"Step {path}: unknown step kind {kind}")

    def _execute_leaf(self, step: WorkflowStep, path: str, record: RunRecord,
                      recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        inputs = _resolve_inputs(step.inputs, ctx, scope)
        kind = step.kind.value
        policy_step_index = ctx.next_policy_step_index()
        decision, reason = self.policy.decide(
            step.ref, kind, path, policy_step_index,
            tool_requires_approval=self._tool_requires_approval(kind, step.ref),
        )
        self._frame(recorder, f"policy:{path}", {"inputs": redact_value(inputs)},
                    {"decision": decision, "reason": reason})
        if decision == "deny":
            raise _AbortRun(f"Step {path}: denied — {reason}")
        with ctx.lock:
            ok, budget_reason = ctx.budget.check()
        if not ok:
            raise _AbortRun(f"Step {path}: budget — {budget_reason}")
        if decision == "require_approval" or step.requires_approval:
            self._approve(step, path, inputs, reason or "approval required", recorder)

        retries = max(0, step.max_retries)
        if step.on_failure == "retry":
            retries = max(retries, 1)
        attempts = 1 + retries
        node = f"{kind}:{step.ref}"
        last_error = ""
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self._backoff(step, attempt - 1, ctx)
            self._check_cancel(ctx)
            try:
                result = self._invoke(kind, step.ref, inputs, step.timeout_s)
            except TimeoutError as err:
                last_error = f"{type(err).__name__}: {err}"
                self._record_error(recorder, node, inputs, err)
                idempotent = step.idempotent or self._tool_idempotent(kind, step.ref)
                retry = idempotent and attempt < attempts
                self._frame(recorder, f"timeout:{path}", {"attempt": attempt},
                            {"error": last_error,
                             "attempt_may_still_be_running": bool(
                                 getattr(err, "still_running", True)),
                             "idempotent": idempotent, "will_retry": retry})
                if retry:
                    continue
                note = "the timed-out attempt may still be running"
                if attempt < attempts:
                    note += "; not retried because the step is not declared idempotent"
                self._leaf_failed(step, path, [f"{last_error} ({note})"])
            except _AbortRun:
                raise
            except Exception as err:  # noqa: BLE001 — recorded, then retry/compensate
                last_error = f"{type(err).__name__}: {err}"
                self._record_error(recorder, node, inputs, err)
                continue
            if result.error:
                last_error = result.error
                self._record_error(recorder, node, inputs, RuntimeError(result.error))
                continue
            verdict = self.verifier.verify(step.check, result)
            self._frame(recorder, f"verify:{path}", redact_value(result.output),
                        {"passed": verdict.passed, "reasons": verdict.reasons})
            if verdict.passed:
                self._commit_leaf(step, path, inputs, result, record, recorder, ctx, scope)
                return
            last_error = "; ".join(verdict.reasons) or "verification failed"
        self._leaf_failed(step, path, [last_error])

    @staticmethod
    def _leaf_failed(step: WorkflowStep, path: str, reasons: list[str]) -> NoReturn:
        if step.on_failure == "skip":
            raise _SkipStep(path, "; ".join(reasons))
        raise _AbortRun(f"Step {path} failed: {'; '.join(reasons)}")

    @staticmethod
    def _backoff(step: WorkflowStep, retry_number: int, ctx: _RunContext) -> None:
        delay = step.retry_delay_s * (2 ** (retry_number - 1))
        if delay > 0:
            ctx.cancel_event.wait(delay)  # wakes early on cancellation

    def _commit_leaf(self, step: WorkflowStep, path: str, inputs: dict[str, Any],
                     result: StepResult, record: RunRecord,
                     recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        recorder.record_step(
            node_name=f"{step.kind.value}:{step.ref}",
            input_state=redact_value(inputs),
            output_state=redact_value(result.output),
            metadata=FrameMetadata(
                latency_ms=result.latency_ms, tokens_used=result.tokens_used or None,
                cost_usd=result.cost_usd or None,
                custom={"step_path": path},
            ),
            frame_type=FrameType.TOOL_CALL if step.kind.value == "tool"
            else FrameType.NODE_EXECUTION,
        )
        with ctx.lock:
            ctx.outputs[path] = result.output
            ctx.budget.account(tokens=result.tokens_used, cost_usd=result.cost_usd,
                               latency_ms=result.latency_ms)
        if step.compensate is not None:
            comp_inputs = _resolve_inputs(step.compensate.inputs, ctx, scope)
            timeout = step.compensate.timeout_s or step.timeout_s
            with ctx.lock:
                record.compensations.append(CompletedCompensation(
                    step_path=path, ref=step.compensate.ref, kind=step.compensate.kind,
                    inputs=comp_inputs, timeout_s=timeout))
        if step.checkpoint_after:
            recorder.record_checkpoint(dict(result.output), label=path)

    def _tool_requires_approval(self, kind: str, ref: str) -> bool:
        return kind == "tool" and ref in self.gateway and self.gateway.get(ref).requires_approval

    def _tool_idempotent(self, kind: str, ref: str) -> bool:
        """A tool registered as idempotent (``ToolSpec.idempotent``) may be retried too."""
        if kind != "tool" or ref not in self.gateway:
            return False
        return bool(getattr(self.gateway.get(ref), "idempotent", False))

    def _invoke(self, kind: str, ref: str, inputs: dict[str, Any],
                timeout_s: float) -> StepResult:
        """Run one tool/agent with a timeout (raises TimeoutError on expiry)."""
        if kind == "tool":
            return self.gateway.call(ref, inputs, timeout_s=timeout_s)
        handler = self.agents.get(ref)
        if handler is None:
            raise KeyError(f"Unknown agent {ref!r}")
        with self._leaf_slots:
            raw = _call_with_timeout(handler, dict(inputs), timeout_s, f"Agent {ref!r}")
        return coerce_result(raw)

    def _approve(self, step: WorkflowStep, path: str, inputs: dict[str, Any],
                 reason: str, recorder: VCRRecorder) -> None:
        request = ApprovalRequest(step_id=path, ref=step.ref or path,
                                  kind=step.kind.value, reason=reason,
                                  inputs=redact_value(inputs))
        decision = self.approver.decide(request)
        self._frame(recorder, f"approval:{path}", {"inputs": redact_value(inputs)},
                    {"approved": decision.approved, "by": decision.decided_by,
                     "reason": decision.note})
        if not decision.approved:
            if step.on_failure == "skip":
                raise _SkipStep(path, f"approval denied: {decision.note}")
            raise _AbortRun(f"Step {path} denied approval: {decision.note}")

    def _execute_approval(self, step: WorkflowStep, path: str,
                          recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        inputs = _resolve_inputs(step.inputs, ctx, scope)
        self._approve(step, path, inputs, step.ref or "manual approval gate", recorder)
        with ctx.lock:
            ctx.outputs[path] = {"approved": True}

    def _execute_condition(self, step: WorkflowStep, path: str, record: RunRecord,
                           recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        if step.condition is None:
            raise _AbortRun(f"Condition step {path} has no predicate")
        matched = _evaluate(step.condition, ctx, scope)
        branch = step.then_steps if matched else step.else_steps
        self._frame(recorder, f"condition:{path}", {"predicate": step.condition.model_dump()},
                    {"matched": matched, "branch": "then" if matched else "else"})
        for sub in branch:
            self._execute_unit(sub, f"{path}.{sub.id}", record, recorder, ctx, scope)

    def _execute_parallel(self, step: WorkflowStep, path: str, record: RunRecord,
                          recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        if not step.branches:
            raise _AbortRun(f"Parallel step {path} has no branches")
        errors: list[str] = []
        errors_lock = threading.Lock()

        def run_branch(name: str, units: list[WorkflowStep]) -> None:
            try:
                for sub in units:
                    self._execute_unit(sub, f"{path}.{name}.{sub.id}", record, recorder,
                                       ctx, scope)
            except Exception as err:  # noqa: BLE001 — collected, abort decided below
                with errors_lock:
                    errors.append(f"{name}: {err}")

        # Dedicated orchestration pool sized to this step's branches: branch
        # threads never compete with leaves (or nested parallels) for slots.
        with ThreadPoolExecutor(max_workers=len(step.branches),
                                thread_name_prefix="vcr-wf-branch") as pool:
            futures = [pool.submit(run_branch, name, units)
                       for name, units in step.branches.items()]
            concurrent.futures.wait(futures)
        merged: dict[str, Any] = {}
        with ctx.lock:
            for name in step.branches:
                prefix = f"{path}.{name}."
                for key, value in ctx.outputs.items():
                    if key.startswith(prefix):
                        merged[f"{name}.{key[len(prefix):]}"] = value
            ctx.outputs[path] = merged
        if errors:
            raise _AbortRun(f"Parallel step {path} failed: {'; '.join(sorted(errors))}")

    def _execute_foreach(self, step: WorkflowStep, path: str, record: RunRecord,
                         recorder: VCRRecorder, ctx: _RunContext, scope: Scope) -> None:
        items = _resolve(step.for_each, ctx, scope)
        if not isinstance(items, list):
            raise _AbortRun(f"Loop {path}: for_each target {step.for_each!r} is not a list "
                            f"(got {type(items).__name__})")
        if len(items) > step.max_iterations:
            raise _AbortRun(f"Loop {path}: {len(items)} items exceed max {step.max_iterations}")
        collected: list[Any] = []
        for i, item in enumerate(items):
            # Loop scope is passed explicitly (not thread-local) so nested
            # parallel branches see it too.
            child: dict[str, Any] = {**scope, step.item_var: item}
            for sub in step.body:
                self._execute_unit(sub, f"{path}[{i}].{sub.id}", record, recorder, ctx, child)
            with ctx.lock:
                collected.append(ctx.outputs.get(f"{path}[{i}].{step.body[-1].id}", {})
                                 if step.body else {})
        with ctx.lock:
            ctx.outputs[path] = {"items": collected, "count": len(collected)}

    # -- cancellation ------------------------------------------------------------

    def _check_cancel(self, ctx: _RunContext) -> None:
        if not ctx.cancel_event.is_set():
            reason = self.runs.cancel_reason(ctx.run_id)
            if reason is None:
                return
            ctx.request_cancel(reason)
        raise _Cancelled(f"Run cancelled: {ctx.cancel_reason}")

    # -- compensation ----------------------------------------------------------

    def _compensate(self, record: RunRecord, recorder: VCRRecorder,
                    ctx: _RunContext) -> None:
        failed = False
        while record.compensations:
            owed = record.compensations.pop()
            if not self._run_compensation(owed, record, recorder, ctx):
                failed = True
            self.runs.persist(record)
        record.compensated = not failed

    def _run_compensation(self, owed: CompletedCompensation, record: RunRecord,
                          recorder: VCRRecorder, ctx: _RunContext) -> bool:
        """Run one compensation through policy → approval → timeout. True on success."""
        node = f"compensate:{owed.ref}"
        label = f"compensate:{owed.step_path}"
        inputs = dict(owed.inputs)
        # Compensations never consume the step budget (index 0): undo must not be
        # blocked by the cap that may have caused the abort. Forbid lists still apply.
        decision, reason = self.policy.decide(
            owed.ref, owed.kind, label, 0,
            tool_requires_approval=self._tool_requires_approval(owed.kind, owed.ref))
        self._frame(recorder, f"policy:{label}", {"inputs": redact_value(inputs)},
                    {"decision": decision, "reason": reason})
        if decision == "deny":
            return self._compensation_skipped(owed, record, recorder, f"denied — {reason}")
        if decision == "require_approval":
            request = ApprovalRequest(step_id=label, ref=owed.ref, kind=owed.kind,
                                      reason=reason or "compensation requires approval",
                                      inputs=redact_value(inputs))
            verdict = self.approver.decide(request)
            self._frame(recorder, f"approval:{label}", {"inputs": redact_value(inputs)},
                        {"approved": verdict.approved, "by": verdict.decided_by,
                         "reason": verdict.note})
            if not verdict.approved:
                return self._compensation_skipped(
                    owed, record, recorder, f"approval denied: {verdict.note}")
        try:
            result = self._invoke(owed.kind, owed.ref, inputs, owed.timeout_s)
            if result.error:
                raise RuntimeError(result.error)
        except Exception as err:  # noqa: BLE001 — keep compensating the rest
            note = " (may still be running)" if isinstance(err, TimeoutError) else ""
            logger.warning("Compensation %s failed: %s%s", owed.step_path, err, note)
            self._record_error(recorder, node, inputs, err)
            record.error += f" | compensation {owed.step_path} failed: {err}{note}"
            return False
        recorder.record_step(
            node_name=node,
            input_state=redact_value(inputs),
            output_state=redact_value(result.output),
            metadata=FrameMetadata(latency_ms=result.latency_ms,
                                   tokens_used=result.tokens_used or None,
                                   cost_usd=result.cost_usd or None,
                                   custom={"step_path": owed.step_path}),
        )
        with ctx.lock:
            ctx.budget.account(tokens=result.tokens_used, cost_usd=result.cost_usd,
                               latency_ms=result.latency_ms)
        return True

    def _compensation_skipped(self, owed: CompletedCompensation, record: RunRecord,
                              recorder: VCRRecorder, why: str) -> bool:
        logger.warning("Compensation %s skipped: %s", owed.step_path, why)
        self._frame(recorder, f"compensate-skipped:{owed.step_path}",
                    {"ref": owed.ref, "kind": owed.kind}, {"skipped": True, "reason": why})
        record.error += f" | compensation {owed.step_path} skipped: {why}"
        return False

    # -- budget --------------------------------------------------------------

    def _new_budget(self, workflow: WorkflowDef) -> BudgetTracker:
        """Fresh per-run ledger: engine caps, tightened by the workflow's own cap."""
        template = self.budget
        cap = template.max_total_cost_usd
        if workflow.max_total_cost_usd is not None:
            cap = workflow.max_total_cost_usd if cap is None \
                else min(cap, workflow.max_total_cost_usd)
        return BudgetTracker(max_total_cost_usd=cap,
                             max_total_tokens=template.max_total_tokens,
                             max_total_latency_ms=template.max_total_latency_ms)

    def _restore_budget(self, record: RunRecord, recorder: VCRRecorder) -> BudgetTracker:
        """Rebuild an interrupted run's cost, token, and latency ledger."""
        budget = self._new_budget(record.definition)
        for frame in recorder.get_frames():
            metadata = frame.metadata
            budget.account(
                tokens=metadata.tokens_used or 0,
                cost_usd=metadata.cost_usd or 0.0,
                latency_ms=metadata.latency_ms,
            )
        return budget

    def _attach(self, record: RunRecord) -> VCRRecorder:
        recorder = VCRRecorder(output_dir=self.vcr_dir)
        try:
            recorder.attach_session(record.session_id)
        except (FileNotFoundError, ValueError):
            # No recording for this run (e.g. created by another backend): start one.
            session = recorder.start_session(
                metadata={"workflow": record.workflow, "run_id": record.run_id},
                tags=["workflow", record.workflow])
            record.session_id = session.session_id
        return recorder

    # -- dag ---------------------------------------------------------------------

    @staticmethod
    def _check_dag(workflow: WorkflowDef) -> None:
        if not workflow.steps:
            raise ValueError("Workflow has no steps")
        WorkflowEngine._topo_order(workflow)

    @staticmethod
    def _topo_order(workflow: WorkflowDef) -> list[str]:
        ids = [s.id for s in workflow.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate top-level step ids")
        edges = {s.id: set(s.depends_on) for s in workflow.steps}
        for step_id, deps in edges.items():
            for dep in deps:
                if dep not in edges:
                    raise ValueError(f"Step {step_id!r} depends on unknown {dep!r}")
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(node: str, stack: list[str]) -> None:
            if node in ordered:
                return
            if node in visiting:
                cycle = stack[stack.index(node):] + [node]
                raise ValueError(f"Dependency cycle: {' -> '.join(cycle)}")
            visiting.add(node)
            for dep in sorted(edges[node]):
                visit(dep, stack + [node])
            visiting.discard(node)
            ordered.append(node)

        for step_id in edges:
            visit(step_id, [])
        return ordered

    def _frame(self, recorder: VCRRecorder, node: str,
               inputs: dict[str, Any], output: dict[str, Any]) -> None:
        # Every engine frame is redacted: outputs can carry credentials too.
        recorder.record_step(node_name=node, input_state=redact_value(inputs),
                             output_state=redact_value(output),
                             metadata=FrameMetadata(latency_ms=0.0))

    @staticmethod
    def _record_error(recorder: VCRRecorder, node: str, inputs: dict[str, Any],
                      err: BaseException) -> None:
        recorder.record_error(node_name=node, input_state=redact_value(inputs),
                              error=err if isinstance(err, Exception)
                              else RuntimeError(str(err)),
                              latency_ms=0.0)


class _RunContext:
    """Ephemeral per-run execution state (rebuilt from the run record on resume)."""

    def __init__(
        self,
        *,
        run_id: str,
        outputs: dict[str, dict[str, Any]],
        inputs: dict[str, Any],
        budget: BudgetTracker,
        known_steps: frozenset[str],
        loop_vars: frozenset[str],
        policy_step_count: int = 0,
    ):
        self.run_id = run_id
        self.outputs: dict[str, dict[str, Any]] = dict(outputs)
        self.inputs = inputs
        self.budget = budget
        self.known_steps = known_steps
        self.loop_vars = loop_vars
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.cancel_reason = ""
        self._policy_step_count = policy_step_count

    def request_cancel(self, reason: str) -> None:
        with self.lock:
            if not self.cancel_reason:
                self.cancel_reason = reason
        self.cancel_event.set()

    def next_policy_step_index(self) -> int:
        """Reserve a policy slot atomically for one leaf action."""
        with self.lock:
            index = self._policy_step_count
            self._policy_step_count += 1
            return index


class _AbortRun(Exception):
    """Internal signal: stop the run and compensate."""


class _Cancelled(_AbortRun):
    """Internal signal: cooperative cancellation was requested."""


class _SkipStep(Exception):
    """Internal signal: a skippable unit failed or was declined; carries its path."""

    def __init__(self, path: str, reason: str = ""):
        super().__init__(f"Step {path} skipped: {reason}" if reason else f"Step {path} skipped")
        self.path = path
        self.reason = reason


def _call_with_timeout(fn: Callable[[dict[str, Any]], Any], arg: dict[str, Any],
                       timeout_s: float, label: str) -> Any:
    """Run ``fn(arg)`` on a daemon thread; raise TimeoutError after ``timeout_s``.

    The thread cannot be killed: on timeout it is abandoned (daemon, so it
    never blocks interpreter exit or engine shutdown) and may still finish.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def target() -> None:
        try:
            box["value"] = fn(arg)
        except BaseException as err:  # noqa: BLE001 — re-raised in the caller
            box["error"] = err
        finally:
            done.set()

    thread = threading.Thread(target=target, name="vcr-wf-agent", daemon=True)
    thread.start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"{label} exceeded {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _collect_ids(steps: list[WorkflowStep]) -> frozenset[str]:
    ids: set[str] = set()
    for step in steps:
        ids.add(step.id)
        nested = [*step.then_steps, *step.else_steps, *step.body]
        for units in step.branches.values():
            nested.extend(units)
        ids |= _collect_ids(nested)
    return frozenset(ids)


def _collect_loop_vars(steps: list[WorkflowStep]) -> frozenset[str]:
    names: set[str] = set()
    for step in steps:
        if step.kind == StepKind.FOR_EACH:
            names.add(step.item_var)
        nested = [*step.then_steps, *step.else_steps, *step.body]
        for units in step.branches.values():
            nested.extend(units)
        names |= _collect_loop_vars(nested)
    return frozenset(names)


def _walk(current: Any, rest: str, ref: str) -> Any:
    """Follow ``.a.b.0`` from ``current``; dict keys containing dots are matched greedily."""
    if not rest:
        return current
    if not rest.startswith("."):
        raise KeyError(f"Unresolvable reference {ref!r}")
    parts = rest[1:].split(".")
    i = 0
    while i < len(parts):
        if isinstance(current, dict):
            for j in range(len(parts), i, -1):
                key = ".".join(parts[i:j])
                if key in current:
                    current = current[key]
                    i = j
                    break
            else:
                raise KeyError(f"Unresolvable reference {ref!r}")
        elif isinstance(current, list) and parts[i].isdigit() and int(parts[i]) < len(current):
            current = current[int(parts[i])]
            i += 1
        else:
            raise KeyError(f"Unresolvable reference {ref!r}")
    return current


def _resolve_ref(ref: str, ctx: _RunContext, scope: Scope) -> Any:
    """Resolve one ``$`` string. Non-references (``"$10"``, ``"$HOME"``) stay literal.

    Known scopes: loop variables in ``scope``, ``inputs`` (run inputs) and
    step outputs (any step id in the definition). ``"$$x"`` → literal ``"$x"``.
    """
    if not ref.startswith("$"):
        return ref
    if ref.startswith("$$"):
        return ref[1:]
    match = _REF_HEAD.match(ref)
    if match is None:
        return ref
    head, rest = match.group(1), ref[match.end():]
    if rest and rest[0] not in ".[":
        return ref
    if head in scope:
        return copy.deepcopy(_walk(scope[head], rest, ref))
    if head == "inputs":
        return copy.deepcopy(_walk(ctx.inputs, rest, ref))
    body = ref[1:]
    with ctx.lock:
        key = max((k for k in ctx.outputs if body == k or body.startswith(f"{k}.")),
                  key=len, default=None)
        if key is not None:
            return copy.deepcopy(_walk(ctx.outputs[key], body[len(key):], ref))
    if head in ctx.known_steps:
        raise KeyError(f"Unresolvable reference {ref!r}")
    if head in ctx.loop_vars:
        raise KeyError(f"Unresolvable reference {ref!r} (loop variable used outside its loop)")
    return ref  # not a known scope → literal string


def _resolve(value: Any, ctx: _RunContext, scope: Scope) -> Any:
    if isinstance(value, str):
        return _resolve_ref(value, ctx, scope)
    if isinstance(value, dict):
        return {k: _resolve(v, ctx, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, ctx, scope) for v in value]
    return value


def _resolve_inputs(inputs: dict[str, Any], ctx: _RunContext, scope: Scope) -> dict[str, Any]:
    return {k: _resolve(v, ctx, scope) for k, v in inputs.items()}


def _evaluate(predicate: ConditionSpec, ctx: _RunContext, scope: Scope) -> bool:
    try:
        actual = _resolve_ref(predicate.ref, ctx, scope)
    except KeyError:
        actual = None
    op, value = predicate.op, predicate.value
    if op == "==":
        return bool(actual == value)
    if op == "!=":
        return bool(actual != value)
    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not actual
    if op == "in":
        return actual in (value if isinstance(value, list) else [value])
    if op == "not_in":
        return actual not in (value if isinstance(value, list) else [value])
    try:
        if op == ">":
            return bool(actual > value)  # type: ignore[operator]
        if op == ">=":
            return bool(actual >= value)  # type: ignore[operator]
        if op == "<":
            return bool(actual < value)  # type: ignore[operator]
        if op == "<=":
            return bool(actual <= value)  # type: ignore[operator]
    except TypeError:
        return False
    raise ValueError(f"Unknown condition op {op!r}")
