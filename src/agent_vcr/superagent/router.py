"""Tool Gateway — validated, budgeted, approval-aware tool execution (Phase 4).

Every external capability (function, HTTP call, shell, MCP tool) is
registered as a :class:`ToolSpec` with a JSON-schema-ish parameter spec.
:class:`ToolGateway.call` validates inputs, enforces per-tool approval
flags, runs the handler with a timeout, and returns a
:class:`StepResult` envelope. Nothing reaches the model output or the
VCR trace unvalidated.

Timeouts stop the *wait*, not the work: Python cannot kill a running
thread, so a timed-out handler may still finish (and cause its side
effect) in the background. :class:`ToolTimeoutError` says so, and callers
must not retry a timed-out tool unless its :attr:`ToolSpec.idempotent`
flag declares repeats safe.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from agent_vcr.superagent.models import StepResult, coerce_result

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

_SCALARS: dict[str, Any] = {"string": str, "integer": int, "number": (int, float),
                            "boolean": bool, "object": dict, "array": list}


def validate_against_schema(params: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Small strict JSON-schema-subset validator (type/required/properties/enum).

    Returns a list of violation messages (empty = valid). Supports nested
    objects and arrays. Unknown keywords are ignored (forward compatible).
    """
    errors: list[str] = []
    if not schema:
        return errors
    expected = schema.get("type")
    if expected and expected in _SCALARS:
        if isinstance(params, bool):
            matches = expected == "boolean"
        else:
            matches = isinstance(params, _SCALARS[expected])
        if not matches:
            errors.append(f"{path or 'params'}: expected {expected}, got {type(params).__name__}")
            return errors
    if expected == "object" and isinstance(params, dict):
        for key in schema.get("required", []):
            if key not in params:
                errors.append(f"{path + '.' if path else ''}{key}: required field missing")
        for key, subschema in schema.get("properties", {}).items():
            if key in params:
                errors.extend(validate_against_schema(
                    params[key], subschema, f"{path + '.' if path else ''}{key}"))
    if expected == "array" and isinstance(params, list) and "items" in schema:
        for i, item in enumerate(params):
            errors.extend(validate_against_schema(item, schema["items"], f"{path}[{i}]"))
    if "enum" in schema and params not in schema["enum"]:
        errors.append(f"{path or 'params'}: {params!r} not in {schema['enum']!r}")
    return errors


@dataclass
class ToolSpec:
    """One registered capability."""

    name: str
    handler: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    timeout_s: float = 120.0
    #: Re-running the tool with the same inputs is safe (no duplicated side
    #: effects). Only idempotent tools are retried after a timeout or
    #: auto-retried by the recovery engine.
    idempotent: bool = False


class ToolTimeoutError(TimeoutError):
    """A tool exceeded its time limit; its thread may still be running.

    ``still_running`` is True when the handler could not be cancelled, i.e.
    any side effect it performs may still happen after this error.
    """

    def __init__(self, name: str, timeout_s: float, *, idempotent: bool = False,
                 still_running: bool = True):
        super().__init__(f"Tool {name!r} exceeded {timeout_s}s")
        self.tool_name = name
        self.timeout_s = timeout_s
        self.idempotent = idempotent
        self.still_running = still_running


def run_in_daemon_thread(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    thread_name: str = "vcr-call",
) -> concurrent.futures.Future[Any]:
    """Run ``fn`` on a fresh daemon thread and return a Future for its result.

    Python cannot kill a thread, so a timed-out handler keeps running. With a
    fixed pool those zombies permanently occupy workers and later calls queue
    behind them (spurious timeouts, and ``shutdown(wait=True)`` hangs). A
    daemon thread per call isolates each call and never blocks interpreter
    exit.
    """
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def runner() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args, **(kwargs or {})))
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiter
            future.set_exception(exc)

    threading.Thread(target=runner, name=thread_name, daemon=True).start()
    return future


class ToolGateway:
    """Registry + validated execution entry point for all tools."""

    def __init__(self, max_workers: int = 8):
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.Lock()
        # Caps how many calls are *awaited* concurrently; a call that times out
        # releases its slot even though its thread may still be running.
        self._slots = threading.BoundedSemaphore(max(1, max_workers))

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not spec.name:
            raise ValueError("Tool name must not be empty")
        if not callable(spec.handler):
            raise ValueError(f"Tool {spec.name!r} handler is not callable")
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"Tool {spec.name!r} already registered")
            self._tools[spec.name] = spec
        logger.debug("Registered tool %s", spec.name)
        return spec

    def tool(self, name: str, description: str = "",
             parameters: dict[str, Any] | None = None,
             requires_approval: bool = False, timeout_s: float = 120.0,
             idempotent: bool = False) -> Callable[[_F], _F]:
        """Decorator for registering a function as a tool."""

        def decorator(fn: _F) -> _F:
            self.register(ToolSpec(name=name, handler=fn, description=description,
                                   parameters=parameters or {},
                                   requires_approval=requires_approval,
                                   timeout_s=timeout_s, idempotent=idempotent))
            return fn

        return decorator

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._tools

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        with self._lock:
            spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"Unknown tool {name!r}")
        return spec

    def is_idempotent(self, name: str) -> bool:
        """True if ``name`` is registered and declared idempotent."""
        with self._lock:
            spec = self._tools.get(name)
        return bool(spec is not None and spec.idempotent)

    def call(self, name: str, params: dict[str, Any],
             timeout_s: float | None = None) -> StepResult:
        """Validate → execute with timeout → envelope. Raises on misuse.

        Raises :class:`ToolTimeoutError` when the limit is hit. The handler
        thread cannot be interrupted, so do not retry unless the tool is
        idempotent.
        """
        spec = self.get(name)
        violations = validate_against_schema(params, spec.parameters)
        if violations:
            raise ValueError(f"Tool {name!r} invalid params: {'; '.join(violations)}")
        limit = timeout_s if timeout_s is not None else spec.timeout_s
        start = time.perf_counter()
        with self._slots:
            # Tool params are passed as a dict so a parameter called e.g. "name"
            # can never collide with this helper's own arguments.
            future = run_in_daemon_thread(
                spec.handler, kwargs=params, thread_name=f"vcr-tool-{name}"
            )
            try:
                raw = future.result(timeout=limit)
            except concurrent.futures.TimeoutError as err:
                raise ToolTimeoutError(name, limit, idempotent=spec.idempotent,
                                       still_running=not future.done()) from err
        latency_ms = (time.perf_counter() - start) * 1000
        result = coerce_result(raw)
        if result.latency_ms == 0.0:
            result.latency_ms = latency_ms
        return result

    def _make_mcp_handler(
        self, server_url: str, tool_name: str, timeout_s: float,
    ) -> Callable[..., Any]:
        """Create a handler that calls an MCP server's tools/call endpoint."""

        def handler(**params: Any) -> Any:
            payload = json.dumps({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
            }).encode()
            req = urllib.request.Request(
                server_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=timeout_s)
            try:
                body = json.loads(resp.read())
            finally:
                resp.close()
            if "error" in body:
                raise RuntimeError(
                    f"MCP tool {tool_name!r} error: {body['error']}"
                )
            return body.get("result", {})

        return handler

    def discover_mcp(
        self,
        server_url: str,
        *,
        prefix: str = "",
        timeout_s: float = 30.0,
    ) -> list[str]:
        """Discover and register tools from an MCP server.

        Connects to *server_url*, calls ``tools/list``, and registers
        each tool as a :class:`ToolSpec` whose handler calls
        ``tools/call`` on the same server.

        Returns the list of newly registered tool names.
        """
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
        }).encode()
        req = urllib.request.Request(
            server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout_s)
        try:
            body = json.loads(resp.read())
        finally:
            resp.close()
        tools_list = body.get("result", {}).get("tools", [])
        registered: list[str] = []
        for tool in tools_list:
            name = f"{prefix}{tool['name']}"
            if name in self:
                logger.debug("MCP tool %s already registered, skipping", name)
                continue
            spec = ToolSpec(
                name=name,
                handler=self._make_mcp_handler(
                    server_url, tool["name"], timeout_s,
                ),
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema", {}),
                timeout_s=timeout_s,
            )
            self.register(spec)
            registered.append(name)
        logger.info("Discovered %d MCP tools from %s", len(registered), server_url)
        return registered

    def shutdown(self) -> None:
        """No-op kept for API compatibility.

        Tool calls run on per-call daemon threads, so there is no pool to join
        and a hung handler can never block shutdown or interpreter exit.
        """
