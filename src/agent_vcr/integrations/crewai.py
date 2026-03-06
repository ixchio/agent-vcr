"""CrewAI integration for Agent VCR."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from agent_vcr.models import FrameMetadata, FrameType
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)


class VCRCrewAI:
    """Drop-in wrapper for CrewAI that auto-records every task execution to VCR.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def kickoff(self, crew: Any, inputs: dict | None = None) -> Any:
        """Wrap crew.kickoff(), recording each task as a VCR frame.

        Args:
            crew: A CrewAI ``Crew`` instance.
            inputs: Optional inputs forwarded to ``crew.kickoff()``.

        Returns:
            The same value returned by ``crew.kickoff()``.
        """
        self._validate_crew(crew)
        original_tasks = self._patch_tasks(crew)
        try:
            start = time.perf_counter()
            result = crew.kickoff(inputs=inputs) if inputs else crew.kickoff()
            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug("Crew kickoff completed in %.2fms", latency_ms)
            return result
        except Exception as e:
            self.recorder.record_error(
                node_name="crew.kickoff",
                input_state=inputs or {},
                error=e,
            )
            raise
        finally:
            self._restore_tasks(crew, original_tasks)

    def wrap_task(self, task: Any) -> Any:
        """Wrap a single CrewAI Task so its execution is auto-recorded."""
        self._validate_task(task)
        original_execute = task.execute
        task.execute = self._make_task_wrapper(
            task_description=getattr(task, "description", "unknown_task"),
            original_execute=original_execute,
        )
        return task

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_crew(self, crew: Any) -> None:
        if not hasattr(crew, "tasks"):
            raise ValueError(
                "crew must have a 'tasks' attribute. "
                "Make sure you are passing a crewai.Crew instance."
            )

    def _validate_task(self, task: Any) -> None:
        if not hasattr(task, "execute"):
            raise ValueError(
                "task must have an 'execute' method. "
                "Make sure you are passing a crewai.Task instance."
            )

    def _patch_tasks(self, crew: Any) -> dict[Any, Callable]:
        """Monkey-patch every task in the crew, return originals for restore."""
        originals: dict[Any, Callable] = {}
        for task in getattr(crew, "tasks", []):
            if hasattr(task, "execute"):
                originals[task] = task.execute
                desc = getattr(task, "description", f"task_{id(task)}")
                task.execute = self._make_task_wrapper(desc, task.execute)
        return originals

    def _restore_tasks(self, crew: Any, originals: dict[Any, Callable]) -> None:
        for task, original in originals.items():
            task.execute = original

    def _make_task_wrapper(self, task_description: str, original_execute: Callable) -> Callable:
        recorder = self.recorder

        def wrapped_execute(*args: Any, **kwargs: Any) -> Any:
            input_state = self._extract_state(kwargs.get("context") or (args[0] if args else {}))
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
                latency_ms = (time.perf_counter() - start) * 1000
                output_state = self._extract_state(output) if output is not None else {}

                if error:
                    logger.warning("Error in CrewAI task '%s': %s", task_description, error)
                    recorder.record_error(
                        node_name=task_description,
                        input_state=input_state,
                        error=error,
                        latency_ms=latency_ms,
                    )
                else:
                    recorder.record_step(
                        node_name=task_description,
                        input_state=input_state,
                        output_state=output_state,
                        metadata=FrameMetadata(latency_ms=latency_ms),
                    )

        return wrapped_execute

    def _extract_state(self, state: Any) -> dict[str, Any]:
        """Safely convert any CrewAI state to a plain dict."""
        if state is None:
            return {}
        if isinstance(state, dict):
            return dict(state)
        if hasattr(state, "model_dump"):
            return state.model_dump()
        if hasattr(state, "__dict__"):
            return dict(state.__dict__)
        return {"_raw": str(state)}


# ---------------------------------------------------------------------------
# Event-driven callback handler
# ---------------------------------------------------------------------------


class VCRCrewCallback:
    """CrewAI callback handler that records agent events to VCR.

    Implements the CrewAI callback interface
    (``on_task_start``, ``on_task_end``, ``on_agent_action``, ``on_tool_end``).

    Usage::

        callback = VCRCrewCallback(recorder)
        crew = Crew(agents=[...], tasks=[...], callbacks=[callback])
        crew.kickoff()
    """

    def __init__(self, recorder: VCRRecorder):
        self.recorder = recorder
        self._task_start_times: dict[str, float] = {}

    def on_task_start(self, task_description: str, agent_role: str, **kwargs: Any) -> None:
        """Called when a CrewAI task starts."""
        key = f"{agent_role}::{task_description}"
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
        key = f"{agent_role}::{task_description}"
        start = self._task_start_times.pop(key, time.perf_counter())
        latency_ms = (time.perf_counter() - start) * 1000

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
        key = f"{agent_role}::{task_description}"
        self._task_start_times.pop(key, None)
        self.recorder.record_error(
            node_name=task_description,
            input_state={"agent_role": agent_role},
            error=error,
        )

    def _extract_state(self, state: Any) -> dict[str, Any]:
        if state is None:
            return {}
        if isinstance(state, dict):
            return dict(state)
        if hasattr(state, "model_dump"):
            return state.model_dump()
        if hasattr(state, "__dict__"):
            return dict(state.__dict__)
        return {"_raw": str(state)}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def vcr_task(
    recorder: VCRRecorder,
    task_name: str | None = None,
) -> Callable:
    """Decorator to record a synchronous CrewAI task function.

    Usage::

        @vcr_task(recorder, task_name="research_step")
        def research(context: dict) -> str:
            return "findings..."
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_name = task_name or func.__name__
            input_state: dict[str, Any] = {}
            if args:
                first = args[0]
                input_state = (
                    dict(first)
                    if isinstance(first, dict)
                    else {"_raw": str(first)}
                )
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
) -> Callable:
    """Decorator to record an async CrewAI task function.

    Usage::

        @vcr_task_async(recorder, task_name="async_research_step")
        async def research(context: dict) -> str:
            return "async findings..."
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            effective_name = task_name or func.__name__
            input_state: dict[str, Any] = {}
            if args:
                first = args[0]
                input_state = (
                    dict(first)
                    if isinstance(first, dict)
                    else {"_raw": str(first)}
                )
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
