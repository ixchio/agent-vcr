"""
CrewAI integration example for Agent VCR.

Demonstrates how to record a CrewAI pipeline without a real LLM — all agents
and tasks are fully mocked so you can run this script offline.
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

from agent_vcr import VCRPlayer, VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI, VCRCrewCallback, vcr_task

# ---------------------------------------------------------------------------
# Simulate CrewAI objects (replace with real crewai imports in production)
# ---------------------------------------------------------------------------

def _make_mock_task(name: str, output: Any = None) -> MagicMock:
    """Build a minimal mock that looks like a crewai Task."""
    task = MagicMock()
    task.description = name
    task.execute = MagicMock(return_value=output or {"result": f"{name}_result"})
    return task


def _make_mock_crew(tasks: list) -> MagicMock:
    """Build a minimal mock that looks like a crewai Crew."""
    crew = MagicMock()
    crew.tasks = tasks
    crew.kickoff = MagicMock(return_value="Final crew output")
    return crew


# ---------------------------------------------------------------------------
# Example 1 — VCRCrewAI wrapper (auto-records all tasks in kickoff)
# ---------------------------------------------------------------------------

def example_vcr_crew_ai_wrapper() -> None:
    print("\n=== Example 1: VCRCrewAI wrapper ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session("crew_run_demo")

        # Build mock crew with 3 tasks
        tasks = [
            _make_mock_task("research_topic", {"findings": "AI is advancing fast"}),
            _make_mock_task("draft_report",   {"draft": "Here are the findings..."}),
            _make_mock_task("review_draft",   {"approved": True}),
        ]
        crew = _make_mock_crew(tasks)

        # 1-liner: wrap and run
        vcr_crew = VCRCrewAI(recorder)
        result = vcr_crew.kickoff(crew)

        vcr_path = recorder.save()

        print(f"  Crew result : {result}")
        print(f"  Frames      : {len(recorder.frames)}")
        print(f"  Saved to    : {vcr_path}")

        # Time-travel: inspect each recorded task
        player = VCRPlayer.load(vcr_path)
        for i, frame in enumerate(player.frames):
            print(f"  Frame {i}: [{frame.node_name}] → {frame.output_state}")


# ---------------------------------------------------------------------------
# Example 2 — VCRCrewCallback (event-driven, attach to real Crew)
# ---------------------------------------------------------------------------

def example_vcr_crew_callback() -> None:
    print("\n=== Example 2: VCRCrewCallback ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session("crew_callback_demo")

        callback = VCRCrewCallback(recorder)

        # Simulate event sequence (in real CrewAI these are fired internally)
        callback.on_task_start("research_topic", agent_role="Researcher")
        callback.on_tool_end(
            tool_name="search_web",
            tool_input={"query": "AI 2025 trends"},
            tool_output={"hits": 10},
            latency_ms=120.0,
        )
        callback.on_task_end(
            "research_topic",
            agent_role="Researcher",
            output={"summary": "Top AI trends identified"},
        )

        callback.on_task_start("draft_report", agent_role="Writer")
        callback.on_task_end(
            "draft_report",
            agent_role="Writer",
            output={"draft": "AI is evolving..."},
        )

        vcr_path = recorder.save()
        print(f"  Frames recorded : {len(recorder.frames)}")
        print(f"  Saved to        : {vcr_path}")


# ---------------------------------------------------------------------------
# Example 3 — vcr_task decorator for individual task functions
# ---------------------------------------------------------------------------

def example_vcr_task_decorator() -> None:
    print("\n=== Example 3: @vcr_task decorator ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session("vcr_task_demo")

        @vcr_task(recorder, task_name="research_step")
        def research(context: dict) -> str:
            topic = context.get("topic", "unknown")
            return f"Research findings about {topic}"

        @vcr_task(recorder, task_name="write_step")
        def write_report(context: dict) -> str:
            findings = context.get("findings", "")
            return f"Report based on: {findings}"

        # Run the pipeline
        findings = research({"topic": "Agentic AI"})
        report = write_report({"findings": findings})

        vcr_path = recorder.save()
        player = VCRPlayer.load(vcr_path)

        print(f"  Final report    : {report}")
        print(f"  Frames recorded : {len(player.frames)}")

        # Time-travel back to step 0 and inspect state
        state_at_0 = player.goto_frame(0)
        print(f"  State at frame 0: {state_at_0}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_vcr_crew_ai_wrapper()
    example_vcr_crew_callback()
    example_vcr_task_decorator()
    print("\n✅ All CrewAI examples completed successfully.")
