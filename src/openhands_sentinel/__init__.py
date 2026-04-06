"""
OpenHands Sentinel — Real-time code quality guardian for AI agents.

Hooks into the OpenHands EventStream, analyzes every file write using AST,
and warns the agent when it duplicates functions, explodes complexity,
or bloats files. The agent self-corrects. Zero human intervention.

Every quality check is recorded natively into agent-vcr for a full
time-travel audit trail.
"""

from openhands_sentinel.analyzer import CodeAnalyzer, Violation, Severity
from openhands_sentinel.sentinel import Sentinel, SentinelConfig

__version__ = "0.1.0"
__all__ = [
    "CodeAnalyzer",
    "Violation",
    "Severity",
    "Sentinel",
    "SentinelConfig",
]
