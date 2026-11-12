"""Pluggable distributed workflow backends (Phase 7).

Provides abstract WorkflowBackend supporting:
- LocalWorkflowBackend (in-process durable execution on the VCR kernel)
- InngestWorkflowBackend — a *test double* only (``simulate=True``); a real Inngest
  backend is not implemented and raises InngestBackendUnavailable
"""

from __future__ import annotations

import abc
import logging
import uuid
from typing import Any

from agent_vcr.workflows.models import (
    CompletedCompensation,
    RunRecord,
    RunStatus,
    WorkflowDef,
    WorkflowStep,
)
from agent_vcr.workflows.store import RunStore

logger = logging.getLogger(__name__)

class WorkflowBackend(abc.ABC):
    """Abstract pluggable workflow execution backend."""

    @abc.abstractmethod
    def dispatch(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunRecord | str:
        """Dispatch a workflow execution. Returns either a RunRecord or dispatch receipt ID."""

    @abc.abstractmethod
    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch current run state."""

    @abc.abstractmethod
    def cancel_run(self, run_id: str) -> bool:
        """Cancel an active workflow run."""

    @abc.abstractmethod
    def resume_run(self, run_id: str) -> RunRecord | None:
        """Resume a suspended or crashed run."""

    @abc.abstractmethod
    def dispatch_step(
        self,
        run_id: str,
        step: WorkflowStep,
        inputs: dict[str, Any],
        output: Any | None = None,
    ) -> Any:
        """Dispatch or execute an individual step."""

    def register_workflow(self, workflow: WorkflowDef) -> None:
        """Optional hook to register workflow definitions with the backend."""
        del workflow  # default no-op hook


class LocalWorkflowBackend(WorkflowBackend):
    """Executes workflows locally using the VCR WorkflowEngine."""

    def __init__(
        self,
        engine: Any | None = None,
        run_store: RunStore | None = None,
        vcr_dir: str = ".vcr",
    ) -> None:
        from agent_vcr.workflows.engine import WorkflowEngine

        self.runs = run_store or RunStore()
        self.engine = engine or WorkflowEngine(run_store=self.runs, vcr_dir=vcr_dir)

    def dispatch(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> RunRecord:
        return self.engine.start(workflow, inputs=inputs, tags=tags)

    def get_run(self, run_id: str) -> RunRecord | None:
        try:
            return self.engine.runs.load(run_id)
        except (KeyError, FileNotFoundError):
            return None

    def cancel_run(self, run_id: str) -> bool:
        """Cooperatively cancel a RUNNING run (compensates, then ABORTED).

        A run executing in this engine stops at its next unit boundary; an
        idle run is compensated and aborted immediately. The engine never
        overwrites the cancellation (it is a sidecar marker, not a status
        flip on the record it holds in memory).
        """
        try:
            return bool(self.engine.cancel(run_id, "Cancelled by user"))
        except (KeyError, FileNotFoundError):
            return False

    def resume_run(self, run_id: str) -> RunRecord | None:
        try:
            return self.engine.resume(run_id)
        except (KeyError, FileNotFoundError):
            return None

    def dispatch_step(
        self,
        run_id: str,
        step: WorkflowStep,
        inputs: dict[str, Any],
        output: Any | None = None,
    ) -> bool:
        """Execute the next pending step on a run."""
        del step, inputs, output
        return self.engine.step_once(run_id)


INNGEST_UNAVAILABLE = (
    "The Inngest workflow backend is not available yet. The previous adapter only sent an "
    "'agent_vcr/<name>.triggered' event: nothing executed the workflow's steps, and cancel "
    "skipped compensation. Use create_workflow_backend('local') (LocalWorkflowBackend), which "
    "runs steps durably in-process with compensation. For tests, "
    "InngestWorkflowBackend(simulate=True) is a test double that records the events it would "
    "send and never executes or sends anything."
)


class InngestBackendUnavailable(NotImplementedError):
    """Raised when a real (non-simulated) Inngest backend is requested."""

    def __init__(self, message: str = INNGEST_UNAVAILABLE) -> None:
        super().__init__(message)


class InngestWorkflowBackend(WorkflowBackend):
    """Test double for a future Inngest backend — ``simulate=True`` only.

    A real Inngest backend is not implemented: constructing one without
    ``simulate=True`` (or with a ``client``) raises
    :class:`InngestBackendUnavailable`.

    The double never executes steps and never contacts Inngest. It creates run
    records, appends the events a real backend would send to
    ``dispatched_events``, and memoizes step outputs you hand it (as if a remote
    worker had run the step). ``cancel_run`` goes through the local
    :class:`~agent_vcr.workflows.engine.WorkflowEngine`, so compensations owed by
    memoized steps run before the run is marked ABORTED.
    """

    def __init__(
        self,
        app_id: str = "agent-vcr",
        event_key: str | None = None,
        client: Any | None = None,
        run_store: RunStore | None = None,
        simulate: bool = False,
        vcr_dir: str = ".vcr",
        engine: Any | None = None,
        **kwargs: Any,
    ) -> None:
        del event_key, kwargs
        if not simulate or client is not None:
            raise InngestBackendUnavailable()
        from agent_vcr.workflows.engine import WorkflowEngine

        self.app_id = app_id
        self.vcr_dir = vcr_dir
        self.simulate = True
        self.client = None
        if engine is not None:
            self.engine = engine
            self.runs = engine.runs
        else:
            self.runs = run_store or RunStore(directory=f"{vcr_dir}/workflow_runs")
            self.engine = WorkflowEngine(run_store=self.runs, vcr_dir=vcr_dir)
        self.dispatched_events: list[dict[str, Any]] = []
        self._registered_functions: dict[str, Any] = {}
        self._memoized_steps: dict[tuple[str, str], Any] = {}

    def format_event(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Format the Inngest event payload a real backend would send."""
        rid = run_id or str(uuid.uuid4())
        return {
            "name": f"agent_vcr/{workflow.name}.triggered",
            "data": {
                "run_id": rid,
                "workflow": workflow.name,
                "version": workflow.version,
                "inputs": inputs or {},
                "steps": [s.id for s in workflow.steps],
            },
            "user": {"id": "vcr_system"},
        }

    def register_workflow(self, workflow: WorkflowDef) -> None:
        """Record the function a real backend would register (id, trigger, steps)."""
        fn_id = f"vcr-{workflow.name}-v{workflow.version}"
        self._registered_functions[fn_id] = {
            "id": fn_id,
            "trigger": f"agent_vcr/{workflow.name}.triggered",
            "steps": [step.id for step in workflow.steps],
        }

    def memoize_step(self, run_id: str, step_id: str, output: Any) -> None:
        """Mark a top-level step as done elsewhere with ``output`` (idempotent).

        If the step declares ``compensate``, the compensation becomes owed, exactly
        as when the engine commits the step itself.
        """
        record = self.get_run(run_id)
        if record is None or record.status != RunStatus.RUNNING:
            self._memoized_steps[(run_id, step_id)] = output
            return
        if step_id in record.completed_top_steps:
            return  # already memoized: first output wins
        self._memoized_steps[(run_id, step_id)] = output
        out_dict = output if isinstance(output, dict) else {"result": output}
        record.outputs[step_id] = out_dict
        record.completed_top_steps.append(step_id)
        step = next((s for s in record.definition.steps if s.id == step_id), None)
        if step is not None and step.compensate is not None:
            record.compensations.append(_owed_compensation(record, step))
        self.runs.persist(record)

    def get_memoized_step(self, run_id: str, step_id: str) -> Any | None:
        """Retrieve memoized step output if previously recorded."""
        if (run_id, step_id) in self._memoized_steps:
            return self._memoized_steps[(run_id, step_id)]
        record = self.get_run(run_id)
        if record is not None and step_id in record.outputs:
            return record.outputs[step_id]
        return None

    def dispatch(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Create the run record and record the trigger event (nothing is sent or run)."""
        del tags
        # Steps run elsewhere, so their refs need not exist in the local gateway;
        # only the DAG shape is checked.
        self.engine._check_dag(workflow)
        self.register_workflow(workflow)
        record = self.runs.create(workflow, inputs)
        self.dispatched_events.append(self.format_event(workflow, inputs, run_id=record.run_id))
        logger.info("Simulated Inngest dispatch of %s (run %s); no step was executed",
                    workflow.name, record.run_id)
        return record.run_id

    def get_run(self, run_id: str) -> RunRecord | None:
        try:
            return self.runs.load(run_id)
        except (KeyError, FileNotFoundError):
            return None

    def cancel_run(self, run_id: str) -> bool:
        """Cancel through the engine: owed compensations run, then ABORTED.

        Returns False for unknown or already-terminal runs.
        """
        try:
            return bool(self.engine.cancel(run_id, "Cancelled via Inngest backend"))
        except (KeyError, FileNotFoundError):
            return False

    def resume_run(self, run_id: str) -> RunRecord | None:
        try:
            record = self.runs.load(run_id)
        except (KeyError, FileNotFoundError):
            return None
        if record.status == RunStatus.RUNNING:
            self.dispatched_events.append(
                self.format_event(record.definition, record.inputs, run_id=run_id))
        return record

    def dispatch_step(
        self,
        run_id: str,
        step: WorkflowStep,
        inputs: dict[str, Any],
        output: Any | None = None,
    ) -> dict[str, Any]:
        """Record the memoized step call a real backend would make."""
        cached_output = self.get_memoized_step(run_id, step.id)
        if output is not None and cached_output is None:
            self.memoize_step(run_id, step.id, output)
            cached_output = output

        kind_val = step.kind.value if hasattr(step.kind, "value") else str(step.kind)
        step_payload = {
            "run_id": run_id,
            "step_id": step.id,
            "ref": step.ref,
            "kind": kind_val,
            "inputs": inputs,
            "memoized": True,
            "cached": cached_output is not None,
            "output": cached_output,
        }
        self.dispatched_events.append({
            "name": f"agent_vcr/{step.id}.step",
            "data": step_payload,
        })
        return step_payload


def _owed_compensation(record: RunRecord, step: WorkflowStep) -> CompletedCompensation:
    """The compensation a committed top-level step owes (inputs resolved like the engine)."""
    from agent_vcr.workflows import engine as eng

    assert step.compensate is not None
    ctx = eng._RunContext(
        run_id=record.run_id, outputs=dict(record.outputs), inputs=dict(record.inputs),
        budget=eng.BudgetTracker(), known_steps=eng._collect_ids(record.definition.steps),
        loop_vars=eng._collect_loop_vars(record.definition.steps))
    return CompletedCompensation(
        step_path=step.id, ref=step.compensate.ref, kind=step.compensate.kind,
        inputs=eng._resolve_inputs(step.compensate.inputs, ctx, eng._EMPTY_SCOPE),
        timeout_s=step.compensate.timeout_s or step.timeout_s)


def create_workflow_backend(
    backend: str = "auto",
    engine: Any | None = None,
    run_store: RunStore | None = None,
    inngest_client: Any | None = None,
    simulate_inngest: bool = False,
    vcr_dir: str = ".vcr",
    **kwargs: Any,
) -> WorkflowBackend:
    """Create a workflow execution backend.

    Supported backends:
    - 'auto' / 'local' / 'engine' / 'in_process': LocalWorkflowBackend.
    - 'inngest': not available yet — raises :class:`InngestBackendUnavailable`
      unless ``simulate_inngest=True`` (or ``simulate=True``), which returns the
      :class:`InngestWorkflowBackend` test double.
    """
    target = backend.lower()
    if target == "inngest":
        simulate = bool(simulate_inngest or kwargs.pop("simulate", False))
        if not simulate or inngest_client is not None:
            raise InngestBackendUnavailable()
        return InngestWorkflowBackend(
            run_store=run_store,
            simulate=True,
            vcr_dir=vcr_dir,
            engine=engine,
            **kwargs,
        )
    if target in ("auto", "local", "engine", "in_process"):
        return LocalWorkflowBackend(engine=engine, run_store=run_store, vcr_dir=vcr_dir)
    raise ValueError(f"Unknown workflow backend: {backend}")
