"""LangGraph integration for Agent VCR."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

from agent_vcr.models import FrameMetadata, FrameType
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
        """Wrap all nodes in a LangGraph with VCR recording."""
        try:
            from langgraph.graph import StateGraph
        except ImportError:
            raise ImportError("langgraph is required. Install with: pip install langgraph")

        if not hasattr(graph, "nodes"):
            raise ValueError("Graph must have a 'nodes' attribute")

        for node_name, node_func in graph.nodes.items():
            self._original_nodes[node_name] = node_func
            wrapped_func = self._create_wrapped_node(node_name, node_func)
            graph.nodes[node_name] = wrapped_func

        return graph

    def wrap_node(self, node_name: str, node_func: Callable) -> Callable:
        """Wrap a single node function with VCR recording."""
        self._original_nodes[node_name] = node_func
        return self._create_wrapped_node(node_name, node_func)

    def _create_wrapped_node(self, node_name: str, node_func: Callable) -> Callable:
        """Create a wrapped version of a node function."""

        def wrapped_node(state: Any) -> Any:
            self._node_call_stack.append(node_name)
            input_state = self._extract_state(state)
            logger.debug("Recording node: %s", node_name)

            start_time = time.perf_counter()
            error: Optional[Exception] = None
            output_state: Any = None

            try:
                output_state = node_func(state)
                return output_state
            except Exception as e:
                error = e
                raise
            finally:
                latency_ms = (time.perf_counter() - start_time) * 1000
                extracted_output = self._extract_state(output_state) if output_state else {}

                if error:
                    logger.warning("Error in node %s: %s", node_name, error)
                    self.recorder.record_error(
                        node_name=node_name,
                        input_state=input_state,
                        error=error,
                        latency_ms=latency_ms,
                    )
                else:
                    metadata = FrameMetadata(latency_ms=latency_ms)
                    self.recorder.record_step(
                        node_name=node_name,
                        input_state=input_state,
                        output_state=extracted_output,
                        metadata=metadata,
                    )

                self._node_call_stack.pop()

        return wrapped_node

    def _extract_state(self, state: Any) -> dict[str, Any]:
        """Extract serializable state from LangGraph state object."""
        if state is None:
            return {}

        if isinstance(state, dict):
            return dict(state)

        if hasattr(state, "model_dump"):
            return state.model_dump()

        if hasattr(state, "__dict__"):
            return dict(state.__dict__)

        return {"_raw": str(state)}


class LangGraphCallback:
    """Callback handler for LangGraph events."""

    def __init__(self, recorder: VCRRecorder):
        self.recorder = recorder

    def on_node_start(self, node_name: str, state: Any) -> None:
        """Called when a node starts execution."""
        pass

    def on_node_end(
        self,
        node_name: str,
        state: Any,
        output: Any,
        latency_ms: float,
    ) -> None:
        """Called when a node finishes execution."""
        self.recorder.record_step(
            node_name=node_name,
            input_state=self._extract_state(state),
            output_state=self._extract_state(output),
            metadata=FrameMetadata(latency_ms=latency_ms),
        )

    def on_llm_start(self, model: str, messages: list[dict]) -> None:
        """Called when an LLM call starts."""
        pass

    def on_llm_end(
        self,
        model: str,
        messages: list[dict],
        response: dict,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
    ) -> None:
        """Called when an LLM call ends."""
        self.recorder.record_llm_call(
            model=model,
            messages=messages,
            response=response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
        )

    def on_tool_start(self, tool_name: str, tool_input: dict) -> None:
        """Called when a tool call starts."""
        pass

    def on_tool_end(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Called when a tool call ends."""
        self.recorder.record_tool_call(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            latency_ms=latency_ms,
            error=error,
        )

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


def vcr_record(
    recorder: VCRRecorder,
    node_name: Optional[str] = None,
):
    """Decorator to record a function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
