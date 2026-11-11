"""CrewAI integration for Agent VCR.

Two recording paths are provided:

* :class:`VCRCrewAI` wraps ``crew.kickoff()`` (and ``kickoff_async`` /
  ``kickoff_for_each``).  For the duration of the call it installs VCR
  callbacks at every level CrewAI actually uses -- ``crew.step_callback`` /
  ``crew.task_callback`` *and* each ``agent.step_callback`` / ``task.callback``
  (CrewAI copies the crew-level callbacks into agents and tasks at kickoff) --
  chaining any user callbacks, and restores every one of them afterwards.
* :class:`VCRCrewCallback` listens on CrewAI's event bus
  (``crewai.events.crewai_event_bus``).  It must be attached explicitly
  (``with VCRCrewCallback(recorder): crew.kickoff()``); CrewAI's ``Crew`` has no
  ``callbacks`` field, so ``Crew(callbacks=[...])`` silently does nothing.

Recording never breaks the user's crew: failures inside VCR callbacks are
logged, never raised.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Callable

from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)

_TASK_LABEL_MAX = 60

# Attribute set on every VCR-generated callback so we can recognise our own wrappers.
_VCR_MARKER = "_agent_vcr_recorder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_state(state: Any) -> dict[str, Any]:
    """Convert any CrewAI object to a plain dict without ever raising."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    try:
        if hasattr(state, "model_dump"):
            dumped = state.model_dump()
            if isinstance(dumped, dict):
                return dumped
            return {"_raw": str(state)}
        if hasattr(state, "__dict__"):
            return dict(state.__dict__)
    except Exception:
        logger.debug("Could not convert %r to a dict", type(state), exc_info=True)
    try:
        return {"_raw": str(state)}
    except Exception:
        return {"_raw": f"<unprintable {type(state).__name__}>"}


def _task_label(obj: Any) -> str:
    """Short, human-readable label for a Task / TaskOutput (``name`` or truncated description)."""
    label = ""
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name.strip():
        label = name
    else:
        desc = (
            obj.get("description") if isinstance(obj, dict) else getattr(obj, "description", None)
        )
        if isinstance(desc, str):
            label = desc
    label = " ".join(label.split())
    if not label:
        return "task"
    return label[:_TASK_LABEL_MAX]


def _task_node_name(obj: Any) -> str:
    return f"task:{_task_label(obj)}"


def _full_description(obj: Any) -> str | None:
    desc = obj.get("description") if isinstance(obj, dict) else getattr(obj, "description", None)
    return desc if isinstance(desc, str) else None


def _agent_role(agent: Any) -> str | None:
    if agent is None:
        return None
    if isinstance(agent, str):
        return agent
    role = getattr(agent, "role", None)
    return role if isinstance(role, str) else None


def _is_pydantic_model(obj: Any) -> bool:
    cls = type(obj)
    return hasattr(cls, "model_fields") or hasattr(cls, "__fields__")


def _set_attr(obj: Any, name: str, value: Any) -> bool:
    """Best-effort setattr; returns False (and logs) if the object refuses."""
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        logger.debug("Could not set %s on %s", name, type(obj).__name__, exc_info=True)
        return False


def _is_vcr_callback(cb: Any) -> bool:
    return getattr(cb, _VCR_MARKER, None) is not None


# ---------------------------------------------------------------------------
# kickoff() wrapper
# ---------------------------------------------------------------------------


@dataclass
class _PatchState:
    """Everything VCRCrewAI changed on a crew, so it can all be restored."""

    crew: Any = None
    crew_step: tuple[bool, Any] | None = None  # (installed, original)
    crew_task: tuple[bool, Any] | None = None
    agents: list[tuple[Any, Any]] = field(default_factory=list)  # (agent, original step_cb)
    tasks: list[tuple[Any, Any]] = field(default_factory=list)  # (task, original callback)
    executes: dict[int, tuple[Any, Callable[..., Any]]] = field(default_factory=dict)


class VCRCrewAI:
    """Drop-in wrapper for CrewAI that auto-records every task execution to VCR.

    During ``kickoff`` it installs VCR callbacks on the crew, on every agent
    (``agent.step_callback``) and on every task (``task.callback``), chaining
    any callbacks you set yourself, and restores all of them when the call
    returns or raises.  Each agent step (``AgentAction`` / ``AgentFinish``) and
    each completed task becomes one frame.  On very old CrewAI versions whose
    tasks expose a patchable ``execute()``, that method is wrapped as well.

    Usage::

        from crewai import Crew, Agent, Task
        from agent_vcr import VCRRecorder
        from agent_vcr.integrations.crewai import VCRCrewAI

        recorder = VCRRecorder()
        recorder.start_session("my_crew_run")

        crew = Crew(agents=[...], tasks=[...])
        vcr_crew = VCRCrewAI(recorder)
        result = vcr_crew.kickoff(crew)

        recorder.save()
    """

    def __init__(
        self,
        recorder: VCRRecorder,
        record_tool_calls: bool = True,
    ):
        self.recorder = recorder
        self.record_tool_calls = record_tool_calls
        self._callback_frames_recorded: int = 0
        self._count_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(self, crew: Any, inputs: dict[str, Any] | None = None) -> Any:
        """Wrap ``crew.kickoff()``, recording agent steps and tasks as VCR frames.

        Args:
            crew: A CrewAI ``Crew`` instance.
            inputs: Optional inputs forwarded to ``crew.kickoff()``.

        Returns:
            The same value returned by ``crew.kickoff()``.
        """
        self._validate_crew(crew)
        state = self._begin(crew)
        start = time.perf_counter()
        try:
            result = crew.kickoff(inputs=inputs) if inputs else crew.kickoff()
            self._log_done("kickoff", start)
            return result
        except Exception as e:
            self._record_kickoff_error("crew.kickoff", {"inputs": inputs}, e, start)
            raise
        finally:
            self._restore(state)

    async def kickoff_async(
        self,
        crew: Any,
        inputs: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Async variant of :meth:`kickoff` wrapping ``await crew.kickoff_async()``.

        Args:
            crew: A CrewAI ``Crew`` instance.
            inputs: Optional inputs forwarded to ``crew.kickoff_async()``.
            timeout: Optional timeout in seconds (``asyncio.wait_for``).  Note that
                CrewAI runs the crew in a worker thread, which keeps running after
                a timeout; VCR's callbacks are removed regardless.
        """
        self._validate_crew(crew)
        state = self._begin(crew)
        start = time.perf_counter()
        try:
            coro = crew.kickoff_async(inputs=inputs) if inputs else crew.kickoff_async()
            result = await asyncio.wait_for(coro, timeout=timeout)
            self._log_done("kickoff_async", start)
            return result
        except Exception as e:
            self._record_kickoff_error("crew.kickoff_async", {"inputs": inputs}, e, start)
            raise
        finally:
            self._restore(state)

    def kickoff_for_each(self, crew: Any, inputs: list[dict[str, Any]]) -> Any:
        """Wrap ``crew.kickoff_for_each(inputs)``.

        CrewAI copies the crew (including its callbacks) once per input, so every
        copy records into this recorder; the original crew is restored afterwards.
        """
        self._validate_crew(crew)
        state = self._begin(crew)
        start = time.perf_counter()
        try:
            result = crew.kickoff_for_each(inputs=inputs)
            self._log_done("kickoff_for_each", start)
            return result
        except Exception as e:
            self._record_kickoff_error("crew.kickoff_for_each", {"inputs": inputs}, e, start)
            raise
        finally:
            self._restore(state)

    def wrap_task(self, task: Any) -> Any:
        """Wrap a single CrewAI Task so its completion is recorded.

        On modern CrewAI (pydantic ``Task``) this chains a VCR callback onto
        ``task.callback`` -- CrewAI's supported per-task hook.  On objects with a
        patchable ``execute()`` (old CrewAI, plain objects, mocks) ``execute`` is
        wrapped instead so the frame also captures latency and errors.  Calling it
        twice with the same recorder is a no-op.
        """
        self._validate_task(task)

        execute = getattr(task, "execute", None)
        if callable(execute) and not _is_pydantic_model(task):
            if getattr(execute, _VCR_MARKER, None) is self.recorder:
                return task
            wrapper = self._make_task_wrapper(
                task_description=_full_description(task) or "unknown_task",
                original_execute=execute,
                node_name=_task_node_name(task),
            )
            if _set_attr(task, "execute", wrapper):
                return task

        if hasattr(task, "callback"):
            original = getattr(task, "callback", None)
            if getattr(original, _VCR_MARKER, None) is self.recorder:
                return task
            if not _set_attr(task, "callback", self._make_task_callback(original)):
                logger.debug("wrap_task: could not install a VCR callback on %r", task)
        else:
            logger.debug("wrap_task: %r has no supported hook; not recording it", task)
        return task

    # ------------------------------------------------------------------
    # Patch / restore
    # ------------------------------------------------------------------

    def _begin(self, crew: Any) -> _PatchState:
        """Install every VCR hook; never raises (a failure just means fewer frames)."""
        with self._count_lock:
            self._callback_frames_recorded = 0
        state = _PatchState(crew=crew)
        try:
            self._install_callbacks(crew, state)
            state.executes = self._patch_tasks(crew)
        except Exception:
            logger.warning("Agent VCR could not fully instrument the crew", exc_info=True)
        return state

    def _restore(self, state: _PatchState) -> None:
        try:
            self._restore_tasks(state.crew, state.executes)
        except Exception:
            logger.warning("Agent VCR could not restore task.execute", exc_info=True)
        self._remove_callbacks(state)

    def _install_callbacks(self, crew: Any, state: _PatchState) -> None:
        """Install VCR callbacks at crew, agent and task level.

        CrewAI semantics (0.1xx - 0.2xx):

        * at kickoff each agent without its own ``step_callback`` receives
          ``crew.step_callback``; the executor only calls the agent's one.
        * at kickoff each task without its own ``callback`` receives
          ``crew.task_callback``; on completion the task calls ``task.callback``
          and then ``crew.task_callback`` unless it is the same object.

        We therefore give every agent its own recording wrapper (chaining the
        agent's callback, or the crew's when the agent had none), and make the
        crew-level wrapper the single recording point for tasks.  Tasks with
        their own user callback get a chaining wrapper too; duplicate deliveries
        of the same ``TaskOutput`` are recorded only once.
        """
        orig_step = getattr(crew, "step_callback", None)
        orig_task = getattr(crew, "task_callback", None)
        recorded_outputs: dict[int, Any] = {}  # id -> obj (kept alive so ids stay unique)
        dedupe_lock = threading.Lock()

        crew_step_cb = self._make_step_callback(orig_step)
        if hasattr(crew, "step_callback") and _set_attr(crew, "step_callback", crew_step_cb):
            state.crew_step = (True, orig_step)

        crew_task_cb = self._make_task_callback(orig_task, recorded_outputs, dedupe_lock)
        if hasattr(crew, "task_callback") and _set_attr(crew, "task_callback", crew_task_cb):
            state.crew_task = (True, orig_task)

        for agent in self._crew_agents(crew):
            if not hasattr(agent, "step_callback"):
                continue
            original = getattr(agent, "step_callback", None)
            chained = original if original else orig_step
            wrapper = self._make_step_callback(chained, agent_role=_agent_role(agent))
            if _set_attr(agent, "step_callback", wrapper):
                state.agents.append((agent, original))

        tasks = getattr(crew, "tasks", None)
        for task in tasks if isinstance(tasks, (list, tuple)) else []:
            if not hasattr(task, "callback"):
                continue
            original = getattr(task, "callback", None)
            if not original or original is orig_task or original == orig_task:
                # CrewAI would hand this task crew.task_callback; give it ours.
                new_cb: Callable[..., Any] = crew_task_cb
            else:
                new_cb = self._make_task_callback(original, recorded_outputs, dedupe_lock)
            if _set_attr(task, "callback", new_cb):
                state.tasks.append((task, original))

    @staticmethod
    def _crew_agents(crew: Any) -> list[Any]:
        agents = getattr(crew, "agents", None)
        result = list(agents) if isinstance(agents, (list, tuple)) else []
        manager = getattr(crew, "manager_agent", None)
        if manager is not None and all(manager is not a for a in result):
            result.append(manager)
        return result

    def _remove_callbacks(self, state: _PatchState) -> None:
        """Restore every callback replaced by :meth:`_install_callbacks`."""
        for task, original in reversed(state.tasks):
            _set_attr(task, "callback", original)
        for agent, original in reversed(state.agents):
            _set_attr(agent, "step_callback", original)
        crew = state.crew
        if state.crew_task is not None:
            _set_attr(crew, "task_callback", state.crew_task[1])
        if state.crew_step is not None:
            _set_attr(crew, "step_callback", state.crew_step[1])
        state.tasks.clear()
        state.agents.clear()
        state.crew_step = state.crew_task = None

    def _bump(self) -> None:
        with self._count_lock:
            self._callback_frames_recorded += 1

    def _make_step_callback(
        self,
        original_cb: Callable[..., Any] | None,
        agent_role: str | None = None,
    ) -> Callable[..., Any]:
        """Create a step_callback that records every agent thought/action."""
        vcr_crew = self

        def step_callback(step_output: Any) -> Any:
            try:
                step_data = vcr_crew._extract_step_output(step_output)
                node_name = str(step_data.pop("_node_name", "agent_step"))
                input_state = step_data.get("input", {})
                if not isinstance(input_state, dict):
                    input_state = {"_raw": str(input_state)}
                if agent_role is not None:
                    input_state = {"agent_role": agent_role, **input_state}
                vcr_crew.recorder.record_step(
                    node_name=node_name,
                    input_state=input_state,
                    output_state=step_data.get("output", step_data),
                    metadata=FrameMetadata(),
                )
                vcr_crew._bump()
            except Exception:
                logger.warning("Agent VCR failed to record a CrewAI step", exc_info=True)

            if original_cb is not None:
                return original_cb(step_output)
            return step_output

        setattr(step_callback, _VCR_MARKER, self.recorder)
        return step_callback

    def _make_task_callback(
        self,
        original_cb: Callable[..., Any] | None,
        recorded: dict[int, Any] | None = None,
        lock: threading.Lock | None = None,
    ) -> Callable[..., Any]:
        """Create a task_callback that records task completion (once per output)."""
        vcr_crew = self
        dedupe_lock = lock or threading.Lock()

        def task_callback(task_output: Any) -> Any:
            try:
                first = True
                if recorded is not None and task_output is not None:
                    with dedupe_lock:
                        first = id(task_output) not in recorded
                        recorded[id(task_output)] = task_output
                if first:
                    task_data = vcr_crew._extract_task_output(task_output)
                    node_name = str(task_data.pop("_node_name", "task_complete"))
                    vcr_crew.recorder.record_step(
                        node_name=node_name,
                        input_state=task_data.get("input", {}),
                        output_state=task_data.get("output", task_data),
                        metadata=FrameMetadata(),
                    )
                    vcr_crew._bump()
            except Exception:
                logger.warning("Agent VCR failed to record a CrewAI task", exc_info=True)

            if original_cb is not None:
                return original_cb(task_output)
            return task_output

        setattr(task_callback, _VCR_MARKER, self.recorder)
        return task_callback

    def _log_done(self, what: str, start: float) -> None:
        logger.debug(
            "Crew %s completed in %.2fms (%d frames via callbacks)",
            what,
            (time.perf_counter() - start) * 1000,
            self._callback_frames_recorded,
        )

    def _record_kickoff_error(
        self, node_name: str, input_state: dict[str, Any], error: Exception, start: float
    ) -> None:
        try:
            self.recorder.record_error(
                node_name=node_name,
                input_state=input_state,
                error=error,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception:
            logger.warning("Agent VCR failed to record a kickoff error", exc_info=True)

    # ------------------------------------------------------------------
    # Step / task output extraction
    # ------------------------------------------------------------------

    def _extract_step_output(self, step_output: Any) -> dict[str, Any]:
        """Parse CrewAI step_callback output into a dict with _node_name."""
        data: dict[str, Any] = {}

        if isinstance(step_output, dict):
            data = dict(step_output)
            data.setdefault("_node_name", data.get("agent", "agent_step"))
            return data

        # CrewAI AgentAction-like objects
        if hasattr(step_output, "tool"):
            data["_node_name"] = f"agent_action::{getattr(step_output, 'tool', 'tool')}"
            data["input"] = self._extract_state(getattr(step_output, "tool_input", None))
            data["output"] = self._extract_state(
                getattr(step_output, "result", getattr(step_output, "log", None))
            )
            return data

        # AgentFinish: LangChain shape (.return_values) or CrewAI shape (.output)
        if hasattr(step_output, "return_values"):
            data["_node_name"] = "agent_finish"
            data["output"] = self._extract_state(getattr(step_output, "return_values", None))
            return data
        if hasattr(step_output, "output") and (
            type(step_output).__name__ == "AgentFinish" or hasattr(step_output, "thought")
        ):
            data["_node_name"] = "agent_finish"
            output: dict[str, Any] = {"output": getattr(step_output, "output", None)}
            thought = getattr(step_output, "thought", None)
            if thought:
                output["thought"] = thought
            data["output"] = output
            return data

        # Generic fallback
        data["_node_name"] = "agent_step"
        data["output"] = self._extract_state(step_output)
        return data

    def _extract_task_output(self, task_output: Any) -> dict[str, Any]:
        """Parse CrewAI task_callback output into a dict with _node_name.

        The node name is short (``task:<name or first 60 chars>``); the full
        description is kept in the frame's input.
        """
        data: dict[str, Any] = {}

        if isinstance(task_output, dict):
            data = dict(task_output)
            if "description" in data or "name" in data:
                data.setdefault("_node_name", _task_node_name(data))
            else:
                data.setdefault("_node_name", "task_complete")
            return data

        data["_node_name"] = _task_node_name(task_output)
        data["output"] = self._extract_state(
            getattr(task_output, "raw", getattr(task_output, "result", task_output))
        )
        input_state: dict[str, Any] = {}
        description = _full_description(task_output)
        if description is not None:
            input_state["description"] = description
        if hasattr(task_output, "agent"):
            input_state["agent"] = str(getattr(task_output, "agent", ""))
        data["input"] = input_state
        return data

    # ------------------------------------------------------------------
    # task.execute() patching (only for old CrewAI / non-pydantic tasks)
    # ------------------------------------------------------------------

    def _validate_crew(self, crew: Any) -> None:
        if not hasattr(crew, "tasks"):
            raise ValueError(
                "crew must have a 'tasks' attribute. "
                "Make sure you are passing a crewai.Crew instance."
            )

    def _validate_task(self, task: Any) -> None:
        if not hasattr(task, "execute") and not hasattr(task, "callback"):
            raise ValueError(
                "task must have an 'execute' method or a 'callback' field. "
                "Make sure you are passing a crewai.Task instance."
            )

    def _patch_tasks(self, crew: Any) -> dict[int, tuple[Any, Callable[..., Any]]]:
        """Wrap ``task.execute`` where that is possible; return originals keyed by id(task).

        Pydantic tasks (all current CrewAI versions) are skipped: assigning a
        method on them raises, and the callbacks already record every task.
        """
        originals: dict[int, tuple[Any, Callable[..., Any]]] = {}
        tasks = getattr(crew, "tasks", None)
        for task in tasks if isinstance(tasks, (list, tuple)) else []:
            execute = getattr(task, "execute", None)
            if not callable(execute) or _is_pydantic_model(task) or id(task) in originals:
                continue
            wrapper = self._make_task_wrapper(
                task_description=_full_description(task) or f"task_{id(task)}",
                original_execute=execute,
                node_name=_task_node_name(task),
            )
            if _set_attr(task, "execute", wrapper):
                originals[id(task)] = (task, execute)
        return originals

    def _restore_tasks(
        self, crew: Any, originals: dict[int, tuple[Any, Callable[..., Any]]]
    ) -> None:
        for task, original in originals.values():
            _set_attr(task, "execute", original)
        originals.clear()

    def _make_task_wrapper(
        self,
        task_description: str,
        original_execute: Callable[..., Any],
        node_name: str | None = None,
    ) -> Callable[..., Any]:
        recorder = self.recorder
        name = node_name or f"task:{' '.join(task_description.split())[:_TASK_LABEL_MAX]}"

        def wrapped_execute(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            error: Exception | None = None
            output: Any = None

            try:
                output = original_execute(*args, **kwargs)
                return output
            except Exception as e:
                error = e
                raise
            finally:
                try:
                    latency_ms = (time.perf_counter() - start) * 1000
                    input_state = self._extract_state(
                        kwargs.get("context") or (args[0] if args else {})
                    )
                    input_state.setdefault("task_description", task_description)
                    if error is not None:
                        logger.warning("Error in CrewAI task '%s': %s", name, error)
                        recorder.record_error(
                            node_name=name,
                            input_state=input_state,
                            error=error,
                            latency_ms=latency_ms,
                        )
                    else:
                        recorder.record_step(
                            node_name=name,
                            input_state=input_state,
                            output_state=self._extract_state(output),
                            metadata=FrameMetadata(latency_ms=latency_ms),
                        )
                except Exception:
                    logger.warning("Agent VCR failed to record task '%s'", name, exc_info=True)

        setattr(wrapped_execute, _VCR_MARKER, recorder)
        return wrapped_execute

    def _extract_state(self, state: Any) -> dict[str, Any]:
        """Safely convert any CrewAI state to a plain dict."""
        return _safe_state(state)


# ---------------------------------------------------------------------------
# Event-bus listener
# ---------------------------------------------------------------------------

_EVENT_NAMES = (
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "AgentExecutionCompletedEvent",
    "AgentExecutionErrorEvent",
    "ToolUsageFinishedEvent",
    "ToolUsageErrorEvent",
    "LLMCallCompletedEvent",
    "LLMCallFailedEvent",
)


def _load_event_bus() -> tuple[Any, dict[str, type]]:
    """Return CrewAI's event bus and the event classes available in this version."""
    events_mod: Any = None
    # Prefer crewai.events: crewai.utilities.events re-exports deprecated *subclasses*
    # of the event types, which would never match isinstance() checks on newer CrewAI.
    for mod_name in ("crewai.events", "crewai.utilities.events"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                events_mod = importlib.import_module(mod_name)
            break
        except ImportError:
            continue
    if events_mod is None:
        raise ImportError(
            "VCRCrewCallback.attach() needs a CrewAI version with an event bus "
            "(crewai.events / crewai.utilities.events). Use VCRCrewAI(recorder)"
            ".kickoff(crew) instead."
        )
    bus = getattr(events_mod, "crewai_event_bus", None)
    if bus is None or isinstance(bus, type):
        # Deprecated class shim (or missing) -> look for the real singleton instance.
        with contextlib.suppress(ImportError, AttributeError):
            bus = importlib.import_module("crewai.events.event_bus").crewai_event_bus
    if bus is None:
        raise ImportError("crewai_event_bus not found in this CrewAI version")
    classes: dict[str, type] = {}
    for name in _EVENT_NAMES:
        cls = getattr(events_mod, name, None)
        if isinstance(cls, type):
            classes[name] = cls
    return bus, classes


class VCRCrewCallback:
    """Records CrewAI events to VCR via CrewAI's event bus.

    CrewAI's ``Crew`` has **no** ``callbacks`` field: ``Crew(callbacks=[cb])``
    is silently ignored by CrewAI and records nothing.  Attach the listener
    explicitly instead -- it records task start/completion/failure, agent
    executions, tool usage and LLM calls for as long as it is attached::

        with VCRCrewCallback(recorder):
            crew.kickoff()

        # or
        listener = VCRCrewCallback(recorder).attach()
        try:
            crew.kickoff()
        finally:
            listener.detach()

    The event bus is process-global: while attached, events from *every* crew
    in the process are recorded into this listener's recorder.  ``detach()``
    removes only this listener's handlers.

    The ``on_*`` methods can also be called manually.
    """

    def __init__(self, recorder: VCRRecorder, record_llm_calls: bool = True):
        self.recorder = recorder
        self.record_llm_calls = record_llm_calls
        self._task_start_times: dict[str, float] = {}
        self._lock = threading.Lock()
        self._bus: Any = None
        self._handlers: list[tuple[type, Callable[[Any, Any], None]]] = []
        self._attached = False
        self._used = False

    # ------------------------------------------------------------------
    # Event-bus attachment
    # ------------------------------------------------------------------

    @property
    def attached(self) -> bool:
        return self._attached

    def attach(self) -> VCRCrewCallback:
        """Register this listener on CrewAI's event bus (idempotent)."""
        bus, classes = _load_event_bus()
        handlers: dict[str, Callable[[Any, Any], None]] = {
            "TaskStartedEvent": self._ev_task_started,
            "TaskCompletedEvent": self._ev_task_completed,
            "TaskFailedEvent": self._ev_task_failed,
            "AgentExecutionCompletedEvent": self._ev_agent_completed,
            "AgentExecutionErrorEvent": self._ev_agent_error,
            "ToolUsageFinishedEvent": self._ev_tool_finished,
            "ToolUsageErrorEvent": self._ev_tool_error,
        }
        if self.record_llm_calls:
            handlers["LLMCallCompletedEvent"] = self._ev_llm_completed
            handlers["LLMCallFailedEvent"] = self._ev_llm_failed
        with self._lock:
            self._used = True
            if self._attached:
                return self
            self._bus = bus
            for name, handler in handlers.items():
                cls = classes.get(name)
                if cls is None:
                    logger.debug("CrewAI event %s not available; skipping", name)
                    continue
                bus.register_handler(cls, handler)
                self._handlers.append((cls, handler))
            self._attached = True
        return self

    def detach(self) -> None:
        """Remove this listener's handlers from the event bus (idempotent)."""
        with self._lock:
            self._attached = False
            bus, handlers = self._bus, self._handlers
            self._handlers = []
            self._bus = None
        if bus is None:
            return
        off = getattr(bus, "off", None)
        for cls, handler in handlers:
            try:
                if callable(off):
                    off(cls, handler)
                    continue
                registry = getattr(bus, "_handlers", None)
                if not isinstance(registry, dict):
                    continue
                lst = registry.get(cls)
                if isinstance(lst, list):
                    lst[:] = [h for h in lst if h != handler]
                    if not lst:
                        registry.pop(cls, None)
            except Exception:
                # Handlers check self._attached, so a leftover one records nothing.
                logger.debug("Could not unregister %s handler", cls, exc_info=True)

    def __enter__(self) -> VCRCrewCallback:
        return self.attach()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.detach()

    def __del__(self) -> None:
        try:
            if not self._used:
                warnings.warn(
                    "VCRCrewCallback was never attached and recorded nothing. CrewAI's "
                    "Crew has no 'callbacks' field, so Crew(callbacks=[VCRCrewCallback(...)]) "
                    "is ignored. Use 'with VCRCrewCallback(recorder): crew.kickoff()' or "
                    "VCRCrewAI(recorder).kickoff(crew).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif self._attached:
                self.detach()
        except Exception:
            pass

    def _guard(self, what: str, fn: Callable[[], None]) -> None:
        if not self._attached:
            return
        try:
            fn()
        except Exception:
            logger.warning("Agent VCR failed to record CrewAI %s", what, exc_info=True)

    # ------------------------------------------------------------------
    # Event handlers (event bus signature: handler(source, event))
    # ------------------------------------------------------------------

    @staticmethod
    def _event_task(source: Any, event: Any) -> Any:
        task = getattr(event, "task", None)
        return task if task is not None else source

    def _ev_task_started(self, source: Any, event: Any) -> None:
        task = self._event_task(source, event)
        self._guard(
            "task start",
            lambda: self.on_task_start(
                _task_node_name(task), _agent_role(getattr(task, "agent", None)) or ""
            ),
        )

    def _ev_task_completed(self, source: Any, event: Any) -> None:
        task = self._event_task(source, event)
        output = getattr(event, "output", None)

        def record() -> None:
            extra: dict[str, Any] = {}
            desc = _full_description(task) or _full_description(output)
            if desc is not None:
                extra["description"] = desc
            role = _agent_role(getattr(task, "agent", None)) or _agent_role(
                getattr(output, "agent", None)
            )
            raw = getattr(output, "raw", None)
            out: Any = {"raw": raw} if isinstance(raw, str) else output
            self.on_task_end(_task_node_name(task), role or "", out, **extra)

        self._guard("task completion", record)

    def _ev_task_failed(self, source: Any, event: Any) -> None:
        task = self._event_task(source, event)
        self._guard(
            "task failure",
            lambda: self.on_task_error(
                _task_node_name(task),
                _agent_role(getattr(task, "agent", None)) or "",
                RuntimeError(str(getattr(event, "error", "task failed"))),
            ),
        )

    def _ev_agent_completed(self, source: Any, event: Any) -> None:
        def record() -> None:
            role = _agent_role(getattr(event, "agent", None)) or "agent"
            self.recorder.record_step(
                node_name=f"agent:{role}",
                input_state={"task": _task_node_name(getattr(event, "task", None))},
                output_state={"output": getattr(event, "output", None)},
                metadata=FrameMetadata(),
            )

        self._guard("agent execution", record)

    def _ev_agent_error(self, source: Any, event: Any) -> None:
        def record() -> None:
            role = _agent_role(getattr(event, "agent", None)) or "agent"
            self.recorder.record_error(
                node_name=f"agent:{role}",
                input_state={"task": _task_node_name(getattr(event, "task", None))},
                error=RuntimeError(str(getattr(event, "error", "agent error"))),
                latency_ms=0.0,
            )

        self._guard("agent error", record)

    def _ev_tool_finished(self, source: Any, event: Any) -> None:
        def record() -> None:
            latency_ms = 0.0
            started, finished = (
                getattr(event, "started_at", None),
                getattr(event, "finished_at", None),
            )
            if started is not None and finished is not None:
                latency_ms = (finished - started).total_seconds() * 1000
            self.on_tool_end(
                tool_name=str(getattr(event, "tool_name", "tool")),
                tool_input=getattr(event, "tool_args", None),
                tool_output=getattr(event, "output", None),
                latency_ms=latency_ms,
            )

        self._guard("tool usage", record)

    def _ev_tool_error(self, source: Any, event: Any) -> None:
        self._guard(
            "tool error",
            lambda: self.on_tool_end(
                tool_name=str(getattr(event, "tool_name", "tool")),
                tool_input=getattr(event, "tool_args", None),
                tool_output=None,
                error=str(getattr(event, "error", "tool error")),
            ),
        )

    def _ev_llm_completed(self, source: Any, event: Any) -> None:
        def record() -> None:
            model = getattr(event, "model", None)
            self.recorder.record_step(
                node_name=f"llm:{model or 'unknown'}",
                input_state={
                    "messages": getattr(event, "messages", None),
                    "agent_role": getattr(event, "agent_role", None),
                    "task_name": getattr(event, "task_name", None),
                },
                output_state={"response": getattr(event, "response", None)},
                metadata=FrameMetadata(model=model if isinstance(model, str) else None),
                frame_type=FrameType.LLM_CALL,
            )

        self._guard("LLM call", record)

    def _ev_llm_failed(self, source: Any, event: Any) -> None:
        def record() -> None:
            model = getattr(event, "model", None)
            self.recorder.record_error(
                node_name=f"llm:{model or 'unknown'}",
                input_state={"agent_role": getattr(event, "agent_role", None)},
                error=RuntimeError(str(getattr(event, "error", "LLM call failed"))),
                latency_ms=0.0,
                model=model if isinstance(model, str) else None,
            )

        self._guard("LLM failure", record)

    # ------------------------------------------------------------------
    # Manual callback API
    # ------------------------------------------------------------------

    def on_task_start(self, task_description: str, agent_role: str, **kwargs: Any) -> None:
        """Called when a CrewAI task starts."""
        self._used = True
        key = f"{agent_role}::{task_description}"
        with self._lock:
            self._task_start_times[key] = time.perf_counter()
        logger.debug("VCRCrewCallback: task start — %s (%s)", task_description, agent_role)

    def on_task_end(
        self,
        task_description: str,
        agent_role: str,
        output: Any,
        **kwargs: Any,
    ) -> None:
        """Called when a CrewAI task ends successfully."""
        self._used = True
        key = f"{agent_role}::{task_description}"
        now = time.perf_counter()
        with self._lock:
            start = self._task_start_times.pop(key, now)
        latency_ms = (now - start) * 1000

        self.recorder.record_step(
            node_name=task_description,
            input_state={"agent_role": agent_role, **kwargs},
            output_state=self._extract_state(output),
            metadata=FrameMetadata(latency_ms=latency_ms),
        )

    def on_agent_action(
        self,
        agent_role: str,
        action: str,
        action_input: Any,
        **kwargs: Any,
    ) -> None:
        """Called when a CrewAI agent takes an action (e.g., tool call)."""
        self._used = True
        self.recorder.record_step(
            node_name=f"agent_action::{agent_role}",
            input_state={"action": action, "action_input": self._extract_state(action_input)},
            output_state={},
            metadata=FrameMetadata(latency_ms=0),
            frame_type=FrameType.TOOL_CALL,
        )

    def on_tool_end(
        self,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_ms: float = 0.0,
        error: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool used by a CrewAI agent finishes."""
        self._used = True
        self.recorder.record_tool_call(
            tool_name=tool_name,
            tool_input=self._extract_state(tool_input),
            tool_output=self._extract_state(tool_output),
            latency_ms=latency_ms,
            error=error,
        )

    def on_task_error(
        self,
        task_description: str,
        agent_role: str,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Called when a CrewAI task raises an exception."""
        self._used = True
        key = f"{agent_role}::{task_description}"
        now = time.perf_counter()
        with self._lock:
            start = self._task_start_times.pop(key, now)
        self.recorder.record_error(
            node_name=task_description,
            input_state={"agent_role": agent_role},
            error=error,
            latency_ms=(now - start) * 1000,
        )

    def _extract_state(self, state: Any) -> dict[str, Any]:
        return _safe_state(state)


#: Alias making the event-bus nature explicit.
VCRCrewListener = VCRCrewCallback


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def vcr_task(
    recorder: VCRRecorder,
    task_name: str | None = None,
) -> Callable[..., Any]:
    """Decorator to record a synchronous CrewAI task function.

    Usage::

        @vcr_task(recorder, task_name="research_step")
        def research(context: dict) -> str:
            return "findings..."
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_name = task_name or func.__name__
            input_state: dict[str, Any] = {}
            if args:
                first = args[0]
                input_state = dict(first) if isinstance(first, dict) else {"_raw": str(first)}
            input_state.update({k: str(v) for k, v in kwargs.items()})

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                recorder.record_step(
                    node_name=effective_name,
                    input_state=input_state,
                    output_state={"result": result},
                    metadata=FrameMetadata(latency_ms=latency_ms),
                )
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.warning("Error in task '%s': %s", effective_name, e)
                recorder.record_error(
                    node_name=effective_name,
                    input_state=input_state,
                    error=e,
                    latency_ms=latency_ms,
                )
                raise

        return wrapper

    return decorator


def vcr_task_async(
    recorder: VCRRecorder,
    task_name: str | None = None,
) -> Callable[..., Any]:
    """Decorator to record an async CrewAI task function.

    Usage::

        @vcr_task_async(recorder, task_name="async_research_step")
        async def research(context: dict) -> str:
            return "async findings..."
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_name = task_name or func.__name__
            input_state: dict[str, Any] = {}
            if args:
                first = args[0]
                input_state = dict(first) if isinstance(first, dict) else {"_raw": str(first)}
            input_state.update({k: str(v) for k, v in kwargs.items()})

            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start) * 1000
                recorder.record_step(
                    node_name=effective_name,
                    input_state=input_state,
                    output_state={"result": result},
                    metadata=FrameMetadata(latency_ms=latency_ms),
                )
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.warning("Error in async task '%s': %s", effective_name, e)
                recorder.record_error(
                    node_name=effective_name,
                    input_state=input_state,
                    error=e,
                    latency_ms=latency_ms,
                )
                raise

        return wrapper

    return decorator
