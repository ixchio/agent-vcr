"""Reliability intelligence — clusters, risk, recovery, routing, costs (Phase 6)."""

from agent_vcr.intelligence.costs import breakdown
from agent_vcr.intelligence.failures import cluster_key, cluster_sessions, normalize_message
from agent_vcr.intelligence.models import (
    CostBreakdown,
    Diagnosis,
    FailureCluster,
    FailureKind,
    ModelStats,
    RecoveryAction,
    RecoveryPlan,
    RiskScore,
    RouteDecision,
)
from agent_vcr.intelligence.query import diagnose
from agent_vcr.intelligence.recovery import (
    auto_recover_workflow,
    classify_player,
    failure_signature,
    plan_recovery,
)
from agent_vcr.intelligence.risk import score_agent, score_session
from agent_vcr.intelligence.routing import ModelRegistry, route

__all__ = [
    "CostBreakdown",
    "Diagnosis",
    "FailureCluster",
    "FailureKind",
    "ModelStats",
    "RecoveryAction",
    "RecoveryPlan",
    "RiskScore",
    "RouteDecision",
    "breakdown",
    "cluster_key",
    "cluster_sessions",
    "normalize_message",
    "diagnose",
    "auto_recover_workflow",
    "classify_player",
    "failure_signature",
    "plan_recovery",
    "score_agent",
    "score_session",
    "ModelRegistry",
    "route",
]
