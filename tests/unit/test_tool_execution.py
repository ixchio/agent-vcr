"""Tests for fixes that span the superagent, workflow and sandbox packages."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from agent_vcr.sandbox import LocalSandbox
from agent_vcr.superagent.router import ToolGateway, ToolTimeoutError


def test_timed_out_tool_does_not_starve_later_calls() -> None:
    gateway = ToolGateway(max_workers=1)
    release = threading.Event()

    @gateway.tool("hang", parameters={"type": "object"}, timeout_s=0.2)
    def hang() -> dict:
        release.wait(10)
        return {"output": {}}

    @gateway.tool("quick", parameters={"type": "object"}, timeout_s=1.0)
    def quick() -> dict:
        return {"output": {"ok": True}}

    try:
        with pytest.raises(ToolTimeoutError) as info:
            gateway.call("hang", {})
        assert info.value.still_running
        # With a single fixed pool worker the zombie call would block this one.
        start = time.perf_counter()
        assert gateway.call("quick", {}).output == {"ok": True}
        assert time.perf_counter() - start < 0.5
        gateway.shutdown()  # must not wait for the hung handler
    finally:
        release.set()


def test_workflow_engine_redacts_recorded_outputs(tmp_path: Path) -> None:
    from agent_vcr.player import VCRPlayer
    from agent_vcr.workflows import WorkflowDef, WorkflowEngine

    gateway = ToolGateway()

    @gateway.tool("leak", parameters={"type": "object"})
    def leak() -> dict:
        return {"output": {"api_key": "sk-live-1234567890abcdef", "status": "ok"}}

    from agent_vcr.workflows import RunStatus, RunStore, WorkflowStep

    engine = WorkflowEngine(gateway, run_store=RunStore(tmp_path / "runs"),
                            vcr_dir=str(tmp_path / "vcr"))
    try:
        record = engine.start(WorkflowDef(name="leaky", steps=[WorkflowStep(id="s", ref="leak")]))
    finally:
        engine.shutdown()
    assert record.status == RunStatus.COMMITTED
    files = list((tmp_path / "vcr").glob("*.vcr"))
    text = "".join(p.read_text() for p in files)
    assert "sk-live-1234567890abcdef" not in text
    assert '"status":"ok"' in text.replace(" ", "")  # non-secret output is still recorded
    assert any(VCRPlayer.load(p).frames for p in files)


def test_sandbox_async_context_manager_does_not_block_loop(tmp_path: Path) -> None:
    async def run() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        task = asyncio.create_task(ticker())
        async with LocalSandbox(workdir=tmp_path / "sb") as sb:
            assert sb.execute_command("echo hi").success
        task.cancel()
        return ticks

    assert asyncio.run(run()) >= 1


def test_tool_parameter_named_name_does_not_collide() -> None:
    gateway = ToolGateway()

    @gateway.tool("greet", parameters={"type": "object"})
    def greet(name: str = "", thread_name: str = "", args: str = "", kwargs: str = "") -> dict:
        return {"output": {"hello": name}}

    result = gateway.call("greet", {"name": "vcr", "thread_name": "t", "args": "a", "kwargs": "k"})
    assert result.output == {"hello": "vcr"}
