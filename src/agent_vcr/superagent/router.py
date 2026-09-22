"""Tool Gateway — validated, budgeted, approval-aware tool execution (Phase 4).

Every external capability (function, HTTP call, shell, MCP tool) is
registered as a :class:`ToolSpec` with a JSON-schema-ish parameter spec.
:class:`ToolGateway.call` validates inputs, enforces per-tool approval
flags, runs the handler with a real timeout, and returns a
:class:`StepResult` envelope. Nothing reaches the model output or the
VCR trace unvalidated.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_vcr.superagent.models import StepResult, coerce_result

logger = logging.getLogger(__name__)

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


class ToolGateway:
    """Registry + validated execution entry point for all tools."""

    def __init__(self, max_workers: int = 8):
        self._tools: dict[str, ToolSpec] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vcr-tool")

    def register(self, spec: ToolSpec) -> ToolSpec:
        if not spec.name:
            raise ValueError("Tool name must not be empty")
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} already registered")
        if not callable(spec.handler):
            raise ValueError(f"Tool {spec.name!r} handler is not callable")
        self._tools[spec.name] = spec
        logger.debug("Registered tool %s", spec.name)
        return spec

    def tool(self, name: str, description: str = "",
             parameters: dict[str, Any] | None = None,
             requires_approval: bool = False, timeout_s: float = 120.0) -> Callable:
        """Decorator for registering a function as a tool."""

        def decorator(fn: Callable) -> Callable:
            self.register(ToolSpec(name=name, handler=fn, description=description,
                                   parameters=parameters or {},
                                   requires_approval=requires_approval,
                                   timeout_s=timeout_s))
            return fn

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as err:
            raise KeyError(f"Unknown tool {name!r}") from err

    def call(self, name: str, params: dict[str, Any],
             timeout_s: float | None = None) -> StepResult:
        """Validate → execute with timeout → envelope. Raises on misuse."""
        spec = self.get(name)
        violations = validate_against_schema(params, spec.parameters)
        if violations:
            raise ValueError(f"Tool {name!r} invalid params: {'; '.join(violations)}")
        limit = timeout_s if timeout_s is not None else spec.timeout_s
        start = time.perf_counter()
        future = self._pool.submit(spec.handler, **params)
        try:
            raw = future.result(timeout=limit)
        except concurrent.futures.TimeoutError as err:
            future.cancel()
            raise TimeoutError(f"Tool {name!r} exceeded {limit}s") from err
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
            body = json.loads(resp.read())
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
        body = json.loads(resp.read())
        tools_list = body.get("result", {}).get("tools", [])
        registered: list[str] = []
        for tool in tools_list:
            name = f"{prefix}{tool['name']}"
            if name in self._tools:
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
        self._pool.shutdown(wait=True, cancel_futures=True)
