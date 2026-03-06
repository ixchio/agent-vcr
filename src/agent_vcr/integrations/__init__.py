"""Framework integrations for Agent VCR."""

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
