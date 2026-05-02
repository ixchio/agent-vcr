"""Framework integrations for Agent VCR."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_vcr.integrations.crewai import (
        VCRCrewAI,
        VCRCrewCallback,
        vcr_task,
        vcr_task_async,
    )
    from agent_vcr.integrations.langgraph import VCRLangGraph

__all__ = [
    "VCRLangGraph",
    "VCRCrewAI",
    "VCRCrewCallback",
    "vcr_task",
    "vcr_task_async",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load integration modules to avoid crashing on missing optional deps."""
    if name == "VCRLangGraph":
        from agent_vcr.integrations.langgraph import VCRLangGraph

        return VCRLangGraph
    if name == "VCRCrewAI":
        from agent_vcr.integrations.crewai import VCRCrewAI

        return VCRCrewAI
    if name == "VCRCrewCallback":
        from agent_vcr.integrations.crewai import VCRCrewCallback

        return VCRCrewCallback
    if name == "vcr_task":
        from agent_vcr.integrations.crewai import vcr_task

        return vcr_task
    if name == "vcr_task_async":
        from agent_vcr.integrations.crewai import vcr_task_async

        return vcr_task_async
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
