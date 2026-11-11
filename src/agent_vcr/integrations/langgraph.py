"""LangGraph integration for Agent VCR."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import threading
import time
from typing import Any, Callable

from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder

logger = logging.getLogger(__name__)


class VCRLangGraph:
    """Drop-in wrapper for LangGraph that auto-records to VCR."""

    def __init__(
        self,
        recorder: VCRRecorder,
        record_llm_calls: bool = True,
        record_tool_calls: bool = True,
    ):
        self.recorder = recorder
        self.record_llm_calls = record_llm_calls
        self.record_tool_calls = record_tool_calls
        self._original_nodes: dict[str, Callable] = {}
        self._node_call_stack: list[str] = []

    def wrap_graph(self, graph: Any) -> Any:
        """Wrap every node of a LangGraph graph with VCR recording (in place).

        Accepts an uncompiled ``StateGraph`` builder (recommended: call this
        before ``compile()``), an already compiled graph, or — for custom
        graph-like objects — any object whose ``nodes`` mapping holds plain
        callables. Node runnables are wrapped rather than replaced, so
        LangGraph keeps injecting ``config``/``runtime``/``store``, async
        nodes are awaited, and retry/cache policies and ``Command`` returns
        keep working.
        """
        if not hasattr(graph, "nodes"):
            raise ValueError("Graph must have a 'nodes' attribute")

        for node_name, node in list(graph.nodes.items()):
            if str(node_name).startswith("__"):
                continue  # LangGraph internals such as __start__
            if hasattr(node, "runnable"):  # StateGraph builder: StateNodeSpec
                if isinstance(node.runnable, _VCRNodeRunnableMarker) or _is_subgraph(
                    node.runnable
                ):
                    continue
                self._original_nodes[node_name] = node.runnable
                wrapped = _make_node_runnable(self, node_name, node.runnable)
                if dataclasses.is_dataclass(node) and not isinstance(node, type):
                    graph.nodes[node_name] = dataclasses.replace(node, runnable=wrapped)
                else:  # NamedTuple in older langgraph releases
                    graph.nodes[node_name] = node._replace(runnable=wrapped)
            elif hasattr(node, "bound") and hasattr(node, "writers"):  # compiled: PregelNode
                if (
                    _is_default_bound(node.bound)
                    or isinstance(node.bound, _VCRNodeRunnableMarker)
                    or _is_subgraph(node.bound)
                ):
                    continue
                self._original_nodes[node_name] = node.bound
                node.bound = _make_node_runnable(self, node_name, node.bound)
                # PregelNode caches the combined runnable; drop it so the new bound is used.
                vars(node).pop("node", None)
            elif callable(node):
                self._original_nodes[node_name] = node
                graph.nodes[node_name] = self._create_wrapped_node(node_name, node)
            else:
                raise TypeError(
                    f"Don't know how to wrap LangGraph node {node_name!r} of type "
                    f"{type(node).__name__}"
                )

        return graph

    def wrap_node(self, node_name: str, node_func: Callable) -> Callable:
        """Wrap a single node function with VCR recording.

        Use at ``builder.add_node("name", vcr.wrap_node("name", fn))`` time.
        Sync and async functions are supported, and the original signature is
        preserved so LangGraph still injects ``config``/``runtime``/``store``.
        """
        self._original_nodes[node_name] = node_func
        return self._create_wrapped_node(node_name, node_func)

    def _create_wrapped_node(self, node_name: str, node_func: Callable) -> Callable:
        """Create a wrapped version of a node function (sync or async)."""
        if inspect.iscoroutinefunction(node_func):

            @functools.wraps(node_func)
            async def async_wrapped_node(state: Any, *args: Any, **kwargs: Any) -> Any:
                start_time = time.perf_counter()
                try:
                    output = await node_func(state, *args, **kwargs)
                except BaseException as exc:
                    self._record_failure(node_name, state, exc, start_time)
                    raise
                self._record_success(node_name, state, output, start_time)
                return output

            return async_wrapped_node

        @functools.wraps(node_func)
        def wrapped_node(state: Any, *args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                output = node_func(state, *args, **kwargs)
            except BaseException as exc:
                self._record_failure(node_name, state, exc, start_time)
                raise
            self._record_success(node_name, state, output, start_time)
            return output

        return wrapped_node

    def _record_success(
        self, node_name: str, state: Any, output: Any, start_time: float
    ) -> None:
        latency_ms = (time.perf_counter() - start_time) * 1000
        try:
            self.recorder.record_step(
                node_name=node_name,
                input_state=self._extract_state(state),
                output_state=self._extract_state(output),
                metadata=FrameMetadata(latency_ms=latency_ms),
            )
        except Exception:  # recording must never break the user's graph
            logger.warning("Agent VCR failed to record node %s", node_name, exc_info=True)

    def _record_failure(
        self, node_name: str, state: Any, exc: BaseException, start_time: float
    ) -> None:
        if not isinstance(exc, Exception) or _is_control_flow(exc):
            # interrupt()/Command(goto=parent)/cancellation are not node failures.
            return
        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.warning("Error in node %s: %s", node_name, exc)
        try:
            self.recorder.record_error(
                node_name=node_name,
                input_state=self._extract_state(state),
                error=exc,
                latency_ms=latency_ms,
            )
        except Exception:
            logger.warning("Agent VCR failed to record error for %s", node_name, exc_info=True)

    def _extract_state(self, state: Any) -> dict[str, Any]:
        """Extract serializable state from a LangGraph state / node return value."""
        if state is None:
            return {}

        if type(state).__name__ == "Command" and hasattr(state, "update"):
            out = {"__command__": {"goto": _jsonable_goto(getattr(state, "goto", None))}}
            update = state.update
            if update is not None:
                out["update"] = self._extract_state(update) if not isinstance(
                    update, (list, tuple)
                ) else {"_items": [list(u) if isinstance(u, tuple) else u for u in update]}
            return out

        if isinstance(state, dict):
            return dict(state)

        if hasattr(state, "model_dump"):
            return state.model_dump()

        if isinstance(state, (list, tuple)):
            return {"_items": list(state)}

        if hasattr(state, "__dict__"):
            return dict(state.__dict__)

        return {"_raw": str(state)}


class _VCRNodeRunnableMarker:
    """Mixin tag so already-wrapped nodes are not wrapped twice."""


def _is_default_bound(bound: Any) -> bool:
    try:
        from langgraph.pregel._read import DEFAULT_BOUND
    except ImportError:  # older langgraph layouts
        return type(bound).__name__ == "RunnablePassthrough"
    return bound is DEFAULT_BOUND


def _is_subgraph(runnable: Any) -> bool:
    """Compiled subgraphs must stay visible to LangGraph (checkpoint namespaces)."""
    try:
        from langgraph.pregel import Pregel
    except ImportError:
        return False
    return isinstance(runnable, Pregel)


def _is_control_flow(exc: BaseException) -> bool:
    """LangGraph uses exceptions (GraphBubbleUp) for interrupts and parent commands."""
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:
        return False
    return isinstance(exc, GraphBubbleUp)


def _jsonable_goto(goto: Any) -> Any:
    if goto is None or isinstance(goto, str):
        return goto
    if isinstance(goto, (list, tuple)):
        return [_jsonable_goto(g) for g in goto]
    return getattr(goto, "node", None) or str(goto)


def _make_node_runnable(vcr: VCRLangGraph, node_name: str, inner: Any) -> Any:
    """Wrap a LangChain ``Runnable`` so each invocation is recorded.

    Delegates ``invoke``/``ainvoke`` with the original ``config`` and kwargs,
    so everything LangGraph relies on (config injection, callbacks, async)
    behaves exactly as before.
    """
    from langchain_core.runnables import Runnable

    class VCRNodeRunnable(Runnable, _VCRNodeRunnableMarker):  # type: ignore[misc]
        def __init__(self) -> None:
            self.inner = inner
            self.name = getattr(inner, "name", None) or node_name

        def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                output = self.inner.invoke(input, config, **kwargs)
            except BaseException as exc:
                vcr._record_failure(node_name, input, exc, start)
                raise
            vcr._record_success(node_name, input, output, start)
            return output

        async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                output = await self.inner.ainvoke(input, config, **kwargs)
            except BaseException as exc:
                vcr._record_failure(node_name, input, exc, start)
                raise
            vcr._record_success(node_name, input, output, start)
            return output

    return VCRNodeRunnable()


try:  # Real LangChain callback base when available (langgraph depends on it).
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
except ImportError:  # pragma: no cover - langchain_core absent

    class _CallbackBase:  # type: ignore[no-redef]
        """Fallback so the manual API works without langchain_core."""


class LangGraphCallback(_CallbackBase):  # type: ignore[misc,valid-type]
    """Record a LangGraph run through LangChain callbacks.

    Pass it per run, no graph changes needed::

        app.invoke(inputs, config={"callbacks": [LangGraphCallback(recorder)]})

    Records one frame per graph node (using LangGraph's ``langgraph_node``
    metadata), plus LLM calls (with token usage when the provider reports it)
    and tool calls. The legacy manual methods (``on_node_end(node_name=...)``,
    ``on_llm_end(model=...)``, ``on_tool_end(tool_name=...)``) keep working:
    calls carrying LangChain's ``run_id`` are treated as callback events,
    everything else as a manual call.
    """

    def __init__(
        self,
        recorder: VCRRecorder,
        record_llm_calls: bool = True,
        record_tool_calls: bool = True,
    ):
        super().__init__()
        self.recorder = recorder
        self.record_llm_calls = record_llm_calls
        self.record_tool_calls = record_tool_calls
        self._runs: dict[Any, dict[str, Any]] = {}
        self._runs_lock = threading.Lock()

    # LangChain callback events

    def on_chain_start(
        self, serialized: Any, inputs: Any, *, run_id: Any, tags: Any = None,
        metadata: Any = None, **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        # Only the node's own run (named after the node, tagged with its graph
        # step), not every runnable nested inside the node.
        if not node or kwargs.get("name") != node:
            return
        if not any(str(t).startswith("graph:step:") for t in (tags or [])):
            return
        self._start(run_id, kind="node", name=str(node), inputs=inputs)

    def on_chain_end(self, outputs: Any, *, run_id: Any, **kwargs: Any) -> None:
        run = self._finish(run_id, "node")
        if run is None:
            return
        self._safe(
            self.recorder.record_step,
            node_name=run["name"],
            input_state=self._extract_state(run["inputs"]),
            output_state=self._extract_state(outputs),
            metadata=FrameMetadata(latency_ms=run["latency_ms"]),
        )

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        run = self._finish(run_id, "node")
        if run is None or not isinstance(error, Exception) or _is_control_flow(error):
            return
        self._safe(
            self.recorder.record_error,
            node_name=run["name"],
            input_state=self._extract_state(run["inputs"]),
            error=error,
            latency_ms=run["latency_ms"],
        )

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        if self.record_llm_calls:
            flat = [m for batch in (messages or []) for m in batch]
            self._start(run_id, kind="llm", name=_model_name(serialized, kwargs),
                        inputs=[_message_dict(m) for m in flat])

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        if "run_id" not in kwargs:
            return  # legacy manual hook was a no-op
        if self.record_llm_calls:
            serialized = args[0] if args else kwargs.get("serialized")
            prompts = args[1] if len(args) > 1 else kwargs.get("prompts", [])
            self._start(kwargs["run_id"], kind="llm", name=_model_name(serialized, kwargs),
                        inputs=[{"role": "user", "content": p} for p in prompts or []])

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        if "run_id" not in kwargs:
            self._manual_llm_end(*args, **kwargs)
            return
        run = self._finish(kwargs["run_id"], "llm")
        if run is None:
            return
        response = args[0] if args else kwargs.get("response")
        text, tokens_in, tokens_out = _llm_result_summary(response)
        self._safe(
            self.recorder.record_llm_call,
            model=run["name"],
            messages=run["inputs"],
            response={"content": text},
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=run["latency_ms"],
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        run = self._finish(run_id, "llm")
        if run is not None and isinstance(error, Exception):
            self._safe(self.recorder.record_error, node_name=f"llm:{run['name']}",
                       input_state={"messages": run["inputs"]}, error=error,
                       latency_ms=run["latency_ms"], model=run["name"])

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        if "run_id" not in kwargs or not self.record_tool_calls:
            return  # legacy manual hook was a no-op
        serialized = (args[0] if args else kwargs.get("serialized")) or {}
        input_str = args[1] if len(args) > 1 else kwargs.get("input_str", "")
        tool_input = kwargs.get("inputs") or {"input": input_str}
        name = serialized.get("name") if isinstance(serialized, dict) else None
        self._start(kwargs["run_id"], kind="tool", name=str(name or "tool"), inputs=tool_input)

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        if "run_id" not in kwargs:
            self._manual_tool_end(*args, **kwargs)
            return
        run = self._finish(kwargs["run_id"], "tool")
        if run is None:
            return
        output = args[0] if args else kwargs.get("output")
        content = getattr(output, "content", output)
        self._safe(self.recorder.record_tool_call, tool_name=run["name"],
                   tool_input=dict(run["inputs"]), tool_output=content,
                   latency_ms=run["latency_ms"])

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        run = self._finish(run_id, "tool")
        if run is not None:
            self._safe(self.recorder.record_tool_call, tool_name=run["name"],
                       tool_input=dict(run["inputs"]), tool_output=None,
                       latency_ms=run["latency_ms"], error=str(error))

    # Legacy manual API

    def on_node_start(self, node_name: str, state: Any) -> None:
        """Called when a node starts execution (manual API; no-op)."""

    def on_node_end(
        self,
        node_name: str,
        state: Any,
        output: Any,
        latency_ms: float,
    ) -> None:
        """Called when a node finishes execution (manual API)."""
        self.recorder.record_step(
            node_name=node_name,
            input_state=self._extract_state(state),
            output_state=self._extract_state(output),
            metadata=FrameMetadata(latency_ms=latency_ms),
        )

    def _manual_llm_end(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
    ) -> None:
        self.recorder.record_llm_call(
            model=model,
            messages=messages,
            response=response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
        )

    def _manual_tool_end(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        self.recorder.record_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=latency_ms,
            error=error,
        )

    # Helpers

    def _start(self, run_id: Any, *, kind: str, name: str, inputs: Any) -> None:
        with self._runs_lock:
            self._runs[run_id] = {
                "kind": kind, "name": name, "inputs": inputs, "t0": time.perf_counter(),
            }

    def _finish(self, run_id: Any, kind: str) -> dict[str, Any] | None:
        with self._runs_lock:
            run = self._runs.get(run_id)
            if run is None or run["kind"] != kind:
                return None
            del self._runs[run_id]
        run["latency_ms"] = (time.perf_counter() - run["t0"]) * 1000
        return run

    @staticmethod
    def _safe(fn: Callable[..., Any], **kwargs: Any) -> None:
        """Recording must never break the user's graph run."""
        try:
            fn(**kwargs)
        except Exception:
            logger.warning("Agent VCR failed to record LangGraph event", exc_info=True)

    def _extract_state(self, state: Any) -> dict[str, Any]:
        """Extract serializable state."""
        if state is None:
            return {}
        if isinstance(state, dict):
            return dict(state)
        if hasattr(state, "model_dump"):
            return state.model_dump()
        if hasattr(state, "__dict__"):
            return dict(state.__dict__)
        return {"_raw": str(state)}


def _model_name(serialized: Any, kwargs: dict[str, Any]) -> str:
    params = kwargs.get("invocation_params") or {}
    meta = kwargs.get("metadata") or {}
    for candidate in (
        params.get("model"), params.get("model_name"), meta.get("ls_model_name"),
        (serialized or {}).get("name") if isinstance(serialized, dict) else None,
    ):
        if candidate:
            return str(candidate)
    return "llm"


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {
        "role": getattr(message, "type", "message"),
        "content": getattr(message, "content", str(message)),
    }


def _llm_result_summary(response: Any) -> tuple[str, int, int]:
    """(text, input_tokens, output_tokens) from a LangChain ``LLMResult``."""
    texts: list[str] = []
    tokens_in = tokens_out = 0
    for batch in getattr(response, "generations", None) or []:
        for gen in batch:
            msg = getattr(gen, "message", None)
            texts.append(str(getattr(msg, "content", None) or getattr(gen, "text", "")))
            usage = getattr(msg, "usage_metadata", None) or {}
            tokens_in += int(usage.get("input_tokens", 0) or 0)
            tokens_out += int(usage.get("output_tokens", 0) or 0)
    if not (tokens_in or tokens_out):
        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        tokens_in = int(usage.get("prompt_tokens", 0) or 0)
        tokens_out = int(usage.get("completion_tokens", 0) or 0)
    return "\n".join(texts), tokens_in, tokens_out


def vcr_record(
    recorder: VCRRecorder,
    node_name: str | None = None,
) -> Callable:
    """Decorator to record a function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_state = {"args": args, "kwargs": kwargs}
            start_time = time.perf_counter()
            effective_name = node_name or func.__name__

            try:
                result = func(*args, **kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000

                recorder.record_step(
                    node_name=effective_name,
                    input_state=input_state,
                    output_state={"result": result},
                    metadata=FrameMetadata(latency_ms=latency_ms),
                )

                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.warning("Error in %s: %s", effective_name, e)

                recorder.record_error(
                    node_name=effective_name,
                    input_state=input_state,
                    error=e,
                    latency_ms=latency_ms,
                )
                raise

        return wrapper
    return decorator
