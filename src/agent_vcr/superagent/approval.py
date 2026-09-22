"""Human-in-the-loop approvals (Phase 4).

When policy (or a step, or a tool) demands approval, the orchestrator
creates an :class:`ApprovalRequest` and asks an :class:`Approver`:

- :class:`AutoApprover` — deterministic policy: approve iff the step is
  under per-decision cost/latency ceilings and not on a deny list.
  This is what runs unattended in CI.
- :class:`CallbackApprover` — delegates to a caller function
  ``decide(request) -> bool`` (CLI prompt, Slack button, API callback).

Every request + decision is recorded as a VCR frame by the orchestrator,
so approvals are auditable and replayable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRequest:
    """One pending human decision."""

    step_id: str
    ref: str
    kind: str
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)
    cost_estimate_usd: float | None = None


@dataclass
class ApprovalDecision:
    approved: bool
    decided_by: str
    note: str = ""


class Approver:
    """Base approver interface."""

    name: str = "base"

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        raise NotImplementedError


class AutoApprover(Approver):
    """Deterministic auto-decisions for unattended runs.

    Approves when the step ref is allow-listed (empty allow-list = all
    refs eligible) AND under the ceilings AND not deny-listed.
    Everything else is denied — fail-closed.
    """

    name = "auto"

    def __init__(self, allow_refs: list[str] | None = None,
                 deny_refs: list[str] | None = None,
                 max_auto_cost_usd: float = 0.0,
                 max_auto_latency_ms: float = 0.0):
        self.allow_refs = allow_refs or []
        self.deny_refs = deny_refs or []
        self.max_auto_cost_usd = max_auto_cost_usd
        self.max_auto_latency_ms = max_auto_latency_ms

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.ref in self.deny_refs:
            return ApprovalDecision(False, self.name, f"{request.ref!r} is deny-listed")
        if self.allow_refs and request.ref not in self.allow_refs:
            return ApprovalDecision(False, self.name, f"{request.ref!r} not in auto-allow list")
        cost = request.cost_estimate_usd or 0.0
        if cost > self.max_auto_cost_usd:
            return ApprovalDecision(
                False, self.name,
                f"estimated ${cost:.4f} exceeds auto ceiling ${self.max_auto_cost_usd:.4f}")
        return ApprovalDecision(True, self.name, "within auto-approve policy")


class CallbackApprover(Approver):
    """Delegate each decision to ``decide_fn`` (human UI, API, tests)."""

    name = "callback"

    def __init__(self, decide_fn: Callable[[ApprovalRequest], bool], name: str = "callback"):
        if not callable(decide_fn):
            raise ValueError("decide_fn must be callable")
        self.decide_fn = decide_fn
        self.name = name

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        try:
            approved = bool(self.decide_fn(request))
        except Exception as err:
            logger.warning("Approver callback failed, denying: %s", err)
            return ApprovalDecision(False, self.name, f"callback error: {err}")
        return ApprovalDecision(approved, self.name, "human decision" if approved else "human denied")
