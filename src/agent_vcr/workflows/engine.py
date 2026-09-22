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
  completion order, then the run is marked ABORTED.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

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
    RunRecord,
    RunStatus,
    StepKind,
    WorkflowDef,
    WorkflowStep,
)
from agent_vcr.workflows.store import RunStore

logger = logging.getLogger(__name__)

AgentHandler = Callable[[dict[str, Any]], Any]


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
        self.budget = budget if budget is not None else BudgetTracker()
        self.approver = approver or AutoApprover()
        self.runs = run_store or RunStore()
        self.backend = backend
        self.vcr_dir = vcr_dir
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vcr-wf")
        self._lock = threading.RLock()

    # -- lifecycle -------------------------------------------------------

    def start(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunRecord:
        """Create a run (new VCR session) and execute it to completion."""
        self._check_dag(workflow)
        self.budget.reset_usage()
        recorder = VCRRecorder(output_dir=self.vcr_dir)
        session = recorder.start_session(
            metadata={"workflow": workflow.name, "workflow_version": workflow.version},
            tags=["workflow", workflow.name, *(tags or [])],
        )
        if workflow.max_total_cost_usd is not None and self.budget.max_total_cost_usd is None:
            self.budget.max_total_cost_usd = workflow.max_total_cost_usd
        record = self.runs.create(workflow, inputs, session_id=session.session_id)
        recorder.record_step(node_name="workflow:start",
                             input_state={"workflow": workflow.name, "inputs": inputs or {}},
                             output_state={"run_id": record.run_id,
                                           "version": workflow.version})
        recorder.save()
        return self._drive(record, recorder)

    def step_once(self, run_id: str) -> bool:
        """Execute the next pending top-level step. Returns True when complete."""
        record = self.runs.load(run_id)
        if record.status != RunStatus.RUNNING:
            return True
        recorder = VCRRecorder(output_dir=self.vcr_dir)
        recorder.attach_session(record.session_id)
        self._restore_budget(record, recorder)
        self._drive(record, recorder, single=True)
        return self.runs.load(run_id).status != RunStatus.RUNNING

    def resume(self, run_id: str) -> RunRecord:
        """Continue a crashed/interrupted run to completion."""
        record = self.runs.load(run_id)
        if record.status != RunStatus.RUNNING:
            return record
        recorder = VCRRecorder(output_dir=self.vcr_dir)
        recorder.attach_session(record.session_id)
        self._restore_budget(record, recorder)
        return self._drive(record, recorder)

    def trigger(self, workflow: WorkflowDef, inputs: dict[str, Any] | None = None,
                secret: str = "") -> RunRecord:
        """Start via trigger contract (webhook secrets enforced when set)."""
        for trigger in workflow.triggers:
            if trigger.type == "webhook" and trigger.secret and trigger.secret != secret:
                raise PermissionError("Invalid webhook secret")
        return self.start(workflow, inputs, tags=["trigger"])

    def shutdown(self, *, close_gateway: bool = True) -> None:
        """Stop engine workers and optionally retain a shared tool gateway."""
        if close_gateway:
            self.gateway.shutdown()
        self._pool.shutdown(wait=True, cancel_futures=True)

    # -- driver ------------------------------------------------------------

    def _drive(self, record: RunRecord, recorder: VCRRecorder, single: bool = False) -> RunRecord:
        order = self._topo_order(record.definition)
        ctx = _RunContext(
            outputs=dict(record.outputs),
            policy_step_count=sum(
                frame.node_name.startswith(("tool:", "agent:"))
                for frame in recorder.get_frames()
            ),
        )
        for step_id in order:
            if step_id in record.completed_top_steps:
                continue
            step = next(s for s in record.definition.steps if s.id == step_id)
            ok = self._execute_top(step, record, recorder, ctx)
            record.outputs.update(ctx.outputs)
            if ok:
                record.completed_top_steps.append(step_id)
                self.runs.persist(record)
                recorder.save()
            if single:
                break
            if not ok:
                break
        if len(record.completed_top_steps) == len(record.definition.steps) \
                and record.status == RunStatus.RUNNING:
            record.status = RunStatus.COMMITTED
            self.runs.persist(record)
            recorder.save()
        return record

    def _execute_top(self, step: WorkflowStep, record: RunRecord,
                     recorder: VCRRecorder, ctx: _RunContext) -> bool:
        if record.error and step.id not in record.completed_top_steps:
            # Resume after a crash during abort handling: the leaf already
            # ran — never re-execute it, go straight to compensation.
            self._compensate(record, recorder)
            record.status = RunStatus.ABORTED
            self.runs.persist(record)
            recorder.save()
            return False
        try:
            self._execute_unit(step, step.id, record, recorder, ctx)
            return True
        except _SkipStep as err:
            ctx.outputs[err.path] = {}
            record.outputs[err.path] = {}
            logger.info("Step %s skipped: %s", err.path, err)
            return True
        except _AbortRun as err:
            record.error = str(err)
            self.runs.persist(record)
            self._compensate(record, recorder)
            record.status = RunStatus.ABORTED
            self.runs.persist(record)
            recorder.save()
            return False
        except KeyError as err:
            record.error = (f"Step {step.id}: unresolvable input {err} "
                            f"(upstream step skipped or empty?)")
            self.runs.persist(record)
            self._compensate(record, recorder)
            record.status = RunStatus.ABORTED
            self.runs.persist(record)
            recorder.save()
            return False

    # -- units ---------------------------------------------------------------

    def _execute_unit(self, step: WorkflowStep, path: str, record: RunRecord,
                      recorder: VCRRecorder, ctx: _RunContext) -> None:
        kind = step.kind
        if kind in (StepKind.TOOL, StepKind.AGENT):
            self._execute_leaf(step, path, record, recorder, ctx)
        elif kind == StepKind.CHECKPOINT:
            state = dict(ctx.outputs)
            recorder.record_checkpoint(state, label=step.ref or path)
        elif kind == StepKind.APPROVAL:
            self._execute_approval(step, path, recorder, ctx)
        elif kind == StepKind.CONDITION:
            self._execute_condition(step, path, record, recorder, ctx)
        elif kind == StepKind.PARALLEL:
            self._execute_parallel(step, path, record, recorder, ctx)
        elif kind == StepKind.FOR_EACH:
            self._execute_foreach(step, path, record, recorder, ctx)
        else:
            raise ValueError(f"Unknown step kind {kind}")

    def _execute_leaf(self, step: WorkflowStep, path: str, record: RunRecord,
                      recorder: VCRRecorder, ctx: _RunContext) -> None:
        inputs = _resolve_inputs(step.inputs, ctx)
        policy_step_index = ctx.next_policy_step_index()
        decision, reason = self.policy.decide(
            step.ref, step.kind.value, path, policy_step_index,
            tool_requires_approval=(
                step.kind.value == "tool" and step.ref in self.gateway
                and self.gateway.get(step.ref).requires_approval
            ),
        )
        self._frame(recorder, f"policy:{path}", {"inputs": redact_value(inputs)},
                    {"decision": decision, "reason": reason})
        if decision == "deny":
            raise _AbortRun(f"Step {path}: denied — {reason}")
        ok, budget_reason = self.budget.check()
        if not ok:
            raise _AbortRun(f"Step {path}: budget — {budget_reason}")
        if decision == "require_approval" or step.requires_approval:
            self._approve(step, path, inputs, reason or "approval required", recorder)

        attempts = 1 + max(0, step.max_retries)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                result = self._call_leaf(step, inputs)
            except Exception as err:  # noqa: BLE001 — recorded, then retry/compensate
                last_error = f"{type(err).__name__}: {err}"
                recorder.record_error(node_name=f"{step.kind.value}:{step.ref}",
                                      input_state=redact_value(inputs),
                                      error=err if isinstance(err, Exception) else RuntimeError(str(err)),
                                      latency_ms=0.0)
                if attempt < attempts:
                    time.sleep(step.retry_delay_s * (2 ** (attempt - 1)))
                    continue
                return self._leaf_failed(step, path, [last_error])
            verdict = self.verifier.verify(step.check, result)
            self._frame(recorder, f"verify:{path}", redact_value(result.output),
                        {"passed": verdict.passed, "reasons": verdict.reasons})
            if verdict.passed:
                return self._commit_leaf(step, path, inputs, result, record, recorder, ctx)
            last_error = "; ".join(verdict.reasons) or "verification failed"
            if attempt < attempts:
                time.sleep(step.retry_delay_s * (2 ** (attempt - 1)))
        self._leaf_failed(step, path, [last_error])

    def _leaf_failed(self, step: WorkflowStep, path: str, reasons: list[str]) -> None:
        if step.on_failure == "skip":
            return None
        raise _AbortRun(f"Step {path} failed: {'; '.join(reasons)}")

    def _commit_leaf(self, step: WorkflowStep, path: str, inputs: dict[str, Any],
                     result: StepResult, record: RunRecord,
                     recorder: VCRRecorder, ctx: _RunContext) -> None:
        if result.error:
            self._leaf_failed(step, path, [result.error])
            return
        recorder.record_step(
            node_name=f"{step.kind.value}:{step.ref}",
            input_state=redact_value(inputs),
            output_state=result.output,
            metadata=FrameMetadata(
                latency_ms=result.latency_ms, tokens_used=result.tokens_used or None,
                cost_usd=result.cost_usd or None,
                custom={"step_path": path},
            ),
            frame_type=FrameType.TOOL_CALL if step.kind.value == "tool" else FrameType.NODE_EXECUTION,
        )
        with self._lock:
            ctx.outputs[path] = result.output
            if step.compensate is not None:
                record.compensations.append(CompletedCompensation(
                    step_path=path, ref=step.compensate.ref, kind=step.compensate.kind,
                    inputs=_resolve_inputs(step.compensate.inputs, ctx)))
        self.budget.account(tokens=result.tokens_used, cost_usd=result.cost_usd,
                            latency_ms=result.latency_ms)
        if step.checkpoint_after:
            recorder.record_checkpoint(dict(result.output), label=path)

    def _call_leaf(self, step: WorkflowStep, inputs: dict[str, Any]) -> StepResult:
        if step.kind == StepKind.TOOL:
            return self.gateway.call(step.ref, inputs, timeout_s=step.timeout_s)
        handler = self.agents.get(step.ref)
        if handler is None:
            raise KeyError(f"Unknown agent {step.ref!r}")
        future = self._pool.submit(handler, inputs)
        try:
            return coerce_result(future.result(timeout=step.timeout_s))
        except concurrent.futures.TimeoutError as err:
            future.cancel()
            raise TimeoutError(f"Agent {step.ref!r} exceeded {step.timeout_s}s") from err

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
                raise _SkipStep(path, decision.note)
            raise _AbortRun(f"Step {path} denied approval: {decision.note}")

    def _execute_approval(self, step: WorkflowStep, path: str,
                          recorder: VCRRecorder, ctx: _RunContext) -> None:
        inputs = _resolve_inputs(step.inputs, ctx)
        self._approve(step, path, inputs, step.ref or "manual approval gate", recorder)
        with self._lock:
            ctx.outputs[path] = {"approved": True}

    def _execute_condition(self, step: WorkflowStep, path: str, record: RunRecord,
                           recorder: VCRRecorder, ctx: _RunContext) -> None:
        if step.condition is None:
            raise ValueError(f"Condition step {path} has no predicate")
        matched = _evaluate(step.condition, ctx)
        branch = step.then_steps if matched else step.else_steps
        self._frame(recorder, f"condition:{path}", {"predicate": step.condition.model_dump()},
                    {"matched": matched, "branch": "then" if matched else "else"})
        for sub in branch:
            self._execute_unit(sub, f"{path}.{sub.id}", record, recorder, ctx)

    def _execute_parallel(self, step: WorkflowStep, path: str, record: RunRecord,
                          recorder: VCRRecorder, ctx: _RunContext) -> None:
        if not step.branches:
            raise ValueError(f"Parallel step {path} has no branches")
        errors: list[str] = []

        def run_branch(name: str, units: list[WorkflowStep]) -> None:
            try:
                for sub in units:
                    self._execute_unit(sub, f"{path}.{name}.{sub.id}", record, recorder, ctx)
            except Exception as err:  # noqa: BLE001 — collected, abort decided below
                with self._lock:
                    errors.append(f"{name}: {err}")

        futures = {self._pool.submit(run_branch, name, units): name
                   for name, units in step.branches.items()}
        for future in concurrent.futures.as_completed(futures):
            future.result()  # run_branch swallows into errors; re-raise nothing
        merged: dict[str, Any] = {}
        with ctx.lock:
            for name in step.branches:
                for key, value in ctx.outputs.items():
                    if key.startswith(f"{path}.{name}."):
                        merged[f"{name}.{key[len(path) + len(name) + 2:]}"] = value
        with self._lock:
            ctx.outputs[path] = merged
        if errors:
            raise _AbortRun(f"Parallel step {path} failed: {'; '.join(sorted(errors))}")

    def _execute_foreach(self, step: WorkflowStep, path: str, record: RunRecord,
                         recorder: VCRRecorder, ctx: _RunContext) -> None:
        items = _resolve_ref(step.for_each, ctx)
        if not isinstance(items, list):
            raise ValueError(f"for_each target {step.for_each!r} is not a list")
        if len(items) > step.max_iterations:
            raise ValueError(f"Loop {path}: {len(items)} items exceed max {step.max_iterations}")
        collected: list[Any] = []
        for i, item in enumerate(items):
            ctx.scope[step.item_var] = item
            try:
                for sub in step.body:
                    self._execute_unit(sub, f"{path}[{i}].{sub.id}", record, recorder, ctx)
                collected.append(ctx.outputs.get(f"{path}[{i}].{step.body[-1].id}", {})
                                 if step.body else {})
            finally:
                ctx.scope.pop(step.item_var, None)
        with self._lock:
            ctx.outputs[path] = {"items": collected, "count": len(collected)}

    # -- compensation ----------------------------------------------------------

    def _compensate(self, record: RunRecord, recorder: VCRRecorder) -> None:
        failed = False
        while record.compensations:
            owed = record.compensations.pop()
            try:
                if owed.kind == "tool":
                    result = self.gateway.call(owed.ref, owed.inputs)
                else:
                    handler = self.agents.get(owed.ref)
                    if handler is None:
                        raise KeyError(f"Unknown compensation agent {owed.ref!r}")
                    result = coerce_result(handler(dict(owed.inputs)))
                recorder.record_step(
                    node_name=f"compensate:{owed.ref}",
                    input_state=redact_value(owed.inputs),
                    output_state=result.output,
                    metadata=FrameMetadata(custom={"step_path": owed.step_path}),
                )
            except Exception as err:  # noqa: BLE001 — keep compensating the rest
                logger.warning("Compensation %s failed: %s", owed.step_path, err)
                recorder.record_error(node_name=f"compensate:{owed.ref}",
                                      input_state=redact_value(owed.inputs),
                                      error=err if isinstance(err, Exception) else RuntimeError(str(err)),
                                      latency_ms=0.0)
                record.error += f" | compensation {owed.step_path} failed: {err}"
                failed = True
            self.runs.persist(record)
        record.compensated = not failed

    def _restore_budget(self, record: RunRecord, recorder: VCRRecorder) -> None:
        """Restore an interrupted run's cost, token, and latency ledger."""
        self.budget.reset_usage()
        if record.definition.max_total_cost_usd is not None:
            self.budget.max_total_cost_usd = record.definition.max_total_cost_usd
        for frame in recorder.get_frames():
            metadata = frame.metadata
            self.budget.account(
                tokens=metadata.tokens_used or 0,
                cost_usd=metadata.cost_usd or 0.0,
                latency_ms=metadata.latency_ms,
            )

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
        import time as _time

        start = _time.perf_counter()
        with self._lock:
            recorder.record_step(
                node_name=node, input_state=inputs, output_state=output,
                metadata=FrameMetadata(latency_ms=(_time.perf_counter() - start) * 1000),
            )


class _RunContext:
    """Ephemeral execution state (rebuilt from the run record on resume)."""

    def __init__(
        self,
        outputs: dict[str, dict[str, Any]] | None = None,
        policy_step_count: int = 0,
    ):
        self.outputs: dict[str, dict[str, Any]] = dict(outputs or {})
        self.lock = threading.RLock()
        self._scope = threading.local()
        self._policy_step_count = policy_step_count

    @property
    def scope(self) -> dict[str, Any]:
        """Return the current branch's private loop scope."""
        if not hasattr(self._scope, "values"):
            self._scope.values = {}
        return self._scope.values

    def next_policy_step_index(self) -> int:
        """Reserve a policy slot atomically for one leaf action."""
        with self.lock:
            index = self._policy_step_count
            self._policy_step_count += 1
            return index


class _AbortRun(Exception):
    """Internal signal: stop the run and compensate."""


class _SkipStep(Exception):
    """Internal signal: a skippable unit declined (approval); carries its path."""

    def __init__(self, path: str, reason: str = ""):
        super().__init__(f"Step {path} skipped: {reason}" if reason else f"Step {path} skipped")
        self.path = path


def _resolve_ref(ref: str, ctx: _RunContext) -> Any:
    if not ref.startswith("$"):
        return ref
    body = ref[1:]
    head, _, path = body.partition(".")
    with ctx.lock:
        if head in ctx.scope:
            current: Any = ctx.scope[head]
        elif head in ctx.outputs:
            current = ctx.outputs[head]
        else:
            raise KeyError(f"Unresolvable reference {ref!r}")
    for part in path.split(".") if path else []:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(f"Unresolvable reference {ref!r}")
    return current


def _resolve(value: Any, ctx: _RunContext) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return _resolve_ref(value, ctx)
    if isinstance(value, dict):
        return {k: _resolve(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, ctx) for v in value]
    return value


def _resolve_inputs(inputs: dict[str, Any], ctx: _RunContext) -> dict[str, Any]:
    return {k: _resolve(v, ctx) for k, v in inputs.items()}


def _evaluate(spec: Any, ctx: _RunContext) -> bool:
    from agent_vcr.workflows.models import ConditionSpec

    predicate: ConditionSpec = spec
    try:
        actual = _resolve_ref(predicate.ref, ctx)
    except KeyError:
        actual = None
    op, value = predicate.op, predicate.value
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
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
            return actual > value  # type: ignore[operator]
        if op == ">=":
            return actual >= value  # type: ignore[operator]
        if op == "<":
            return actual < value  # type: ignore[operator]
        if op == "<=":
            return actual <= value  # type: ignore[operator]
    except TypeError:
        return False
    raise ValueError(f"Unknown condition op {op!r}")
