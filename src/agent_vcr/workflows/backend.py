"""Pluggable distributed workflow backends (Phase 7).

Provides abstract WorkflowBackend supporting:
- LocalWorkflowBackend (in-process durable execution on the VCR kernel)
- InngestWorkflowBackend (distributed event-driven dispatch with step memoization for LLMs)
"""

from __future__ import annotations

import abc
import logging
import os
import uuid
from typing import Any

from agent_vcr.workflows.models import (
    RunRecord,
    RunStatus,
    WorkflowDef,
    WorkflowStep,
)
from agent_vcr.workflows.store import RunStore

logger = logging.getLogger(__name__)

# Check for Inngest SDK
_INNGEST_IMPORT_ERROR: ImportError | None = None
try:
    import inngest  # type: ignore[import-not-found]
except ImportError as exc:
    _INNGEST_IMPORT_ERROR = exc
    inngest = None  # type: ignore[assignment]


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
        try:
            record = self.engine.runs.load(run_id)
            if record.status == RunStatus.RUNNING:
                record.status = RunStatus.ABORTED
                record.error = "Cancelled by user"
                self.engine.runs.persist(record)
                return True
            return False
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


class InngestWorkflowBackend(WorkflowBackend):
    """Distributed workflow backend using Inngest (inngest >= 0.5.0).

    Uses step memoization so non-deterministic LLM/agent responses are preserved
    across retries and distributed worker recovery.
    """

    def __init__(
        self,
        app_id: str = "agent-vcr",
        event_key: str | None = None,
        client: Any | None = None,
        run_store: RunStore | None = None,
        simulate: bool = False,
        vcr_dir: str = ".vcr",
        **kwargs: Any,
    ) -> None:
        self.app_id = app_id
        self.event_key = event_key or os.environ.get("INNGEST_EVENT_KEY")
        self.vcr_dir = vcr_dir
        self.runs = run_store or RunStore(directory=f"{vcr_dir}/workflow_runs")
        self.simulate = simulate
        self.dispatched_events: list[dict[str, Any]] = []
        self._registered_functions: dict[str, Any] = {}
        self._memoized_steps: dict[tuple[str, str], Any] = {}

        if client is not None:
            self.client = client
        elif self.simulate or _INNGEST_IMPORT_ERROR is not None:
            self.client = None
        else:
            inngest_mod = getattr(inngest, "Inngest", None)
            if inngest_mod is not None:
                self.client = inngest_mod(app_id=app_id, event_key=self.event_key)
            else:
                self.client = None

    def format_event(
        self,
        workflow: WorkflowDef,
        inputs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Format an Inngest event payload for a workflow trigger."""
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
        """Build Inngest function definition with step memoization."""
        fn_id = f"vcr-{workflow.name}-v{workflow.version}"
        trigger_event = f"agent_vcr/{workflow.name}.triggered"

        # Record step plan for memoized execution
        self._registered_functions[fn_id] = {
            "id": fn_id,
            "trigger": trigger_event,
            "steps": [step.id for step in workflow.steps],
        }
        logger.debug("Registered Inngest function %s for event %s", fn_id, trigger_event)

    def memoize_step(self, run_id: str, step_id: str, output: Any) -> None:
        """Store memoized step output for deterministic recovery and replay."""
        self._memoized_steps[(run_id, step_id)] = output
        record = self.get_run(run_id)
        if record is not None:
            out_dict = output if isinstance(output, dict) else {"result": output}
            record.outputs[step_id] = out_dict
            if step_id not in record.completed_top_steps:
                record.completed_top_steps.append(step_id)
            self.runs.persist(record)

    def get_memoized_step(self, run_id: str, step_id: str) -> Any | None:
        """Retrieve memoized step output if previously executed."""
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
        """Send event to Inngest to trigger distributed execution."""
        self.register_workflow(workflow)
        record = self.runs.create(workflow, inputs)
        event_payload = self.format_event(workflow, inputs, run_id=record.run_id)

        self.dispatched_events.append(event_payload)

        if self.client is not None and hasattr(self.client, "send"):
            try:
                event_cls = (
                    getattr(inngest, "Event", None) if inngest is not None else None
                )
                if event_cls is not None:
                    evt = event_cls(
                        name=event_payload["name"],
                        data=event_payload["data"],
                        user=event_payload.get("user"),
                    )
                    self.client.send(evt)
                else:
                    self.client.send(event_payload)
            except Exception as exc:
                logger.error("Failed to send event to Inngest: %s", exc)
                if not self.simulate:
                    raise

        logger.info(
            "Dispatched workflow %s (run %s) via Inngest event",
            workflow.name,
            record.run_id,
        )
        return record.run_id

    def get_run(self, run_id: str) -> RunRecord | None:
        try:
            return self.runs.load(run_id)
        except (KeyError, FileNotFoundError):
            return None

    def cancel_run(self, run_id: str) -> bool:
        try:
            record = self.runs.load(run_id)
            if record.status == RunStatus.RUNNING:
                record.status = RunStatus.ABORTED
                record.error = "Cancelled via Inngest backend"
                self.runs.persist(record)
                return True
            return False
        except (KeyError, FileNotFoundError):
            return False

    def resume_run(self, run_id: str) -> RunRecord | None:
        try:
            record = self.runs.load(run_id)
            if record.status == RunStatus.RUNNING:
                # Re-dispatch event to resume steps
                event_payload = self.format_event(record.definition, record.inputs, run_id=run_id)
                self.dispatched_events.append(event_payload)
            return record
        except (KeyError, FileNotFoundError):
            return None

    def dispatch_step(
        self,
        run_id: str,
        step: WorkflowStep,
        inputs: dict[str, Any],
        output: Any | None = None,
    ) -> dict[str, Any]:
        """Format and dispatch an Inngest memoized step call."""
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
    - 'auto': chooses Inngest if inngest_client or INNGEST_EVENT_KEY is provided; otherwise LocalWorkflowBackend.
    - 'local': LocalWorkflowBackend.
    - 'inngest': InngestWorkflowBackend.
    """
    target = backend.lower()
    if target == "auto":
        has_inngest_cfg = bool(inngest_client or os.environ.get("INNGEST_EVENT_KEY"))
        try:
            import inngest  # noqa: F401
            inngest_available = True
        except ImportError:
            inngest_available = False

        target = "inngest" if (inngest_available and has_inngest_cfg) else "local"

    if target == "inngest":
        return InngestWorkflowBackend(
            client=inngest_client,
            run_store=run_store,
            simulate=simulate_inngest or kwargs.get("simulate", False),
            **kwargs,
        )
    elif target in ("local", "engine", "in_process"):
        return LocalWorkflowBackend(engine=engine, run_store=run_store, vcr_dir=vcr_dir)
    else:
        raise ValueError(f"Unknown workflow backend: {backend}")
