"""Policy engine + budget tracker (Phase 4).

:class:`Policy` is a static allow/deny rule-set evaluated BEFORE every
step executes: forbidden tools, approval-required refs, step-count caps,
and irreversible-action gates. :class:`BudgetTracker` is the dynamic
ledger: tokens, dollars, milliseconds accumulated per run, with
per-step and per-run caps that halt execution before overspend.

Both are pure, deterministic, and fully unit-tested. The orchestrator
records every decision as a VCR frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Decision = Literal["allow", "deny", "require_approval"]


@dataclass
class Policy:
    """Static execution policy for one run."""

    forbid_tools: list[str] = field(default_factory=list)
    forbid_agents: list[str] = field(default_factory=list)
    approval_required: list[str] = field(default_factory=list)  # tool/agent refs or step ids
    max_steps: int | None = None
    irreversible_refs: list[str] = field(default_factory=list)  # always need approval

    def decide(self, ref: str, kind: str, step_id: str, step_index: int,
               tool_requires_approval: bool = False) -> tuple[Decision, str]:
        """Evaluate one pending step. Returns (decision, reason)."""
        if kind == "tool" and ref in self.forbid_tools:
            return "deny", f"tool {ref!r} is forbidden by policy"
        if kind == "agent" and ref in self.forbid_agents:
            return "deny", f"agent {ref!r} is forbidden by policy"
        if self.max_steps is not None and step_index >= self.max_steps:
            return "deny", f"step budget exceeded (max {self.max_steps})"
        if ref in self.irreversible_refs or step_id in self.irreversible_refs:
            return "require_approval", f"{ref!r} is irreversible — approval required"
        if ref in self.approval_required or step_id in self.approval_required:
            return "require_approval", f"{ref!r} requires approval by policy"
        if tool_requires_approval:
            return "require_approval", f"tool {ref!r} requires approval"
        return "allow", ""


@dataclass
class BudgetTracker:
    """Running ledger with caps; :meth:`check` before, :meth:`account` after."""

    max_total_cost_usd: float | None = None
    max_total_tokens: int | None = None
    max_total_latency_ms: float | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def reset_usage(self) -> None:
        """Clear one execution's usage while preserving configured caps."""
        self.tokens_used = 0
        self.cost_usd = 0.0
        self.latency_ms = 0.0

    def check(self, step_max_cost: float | None = None) -> tuple[bool, str]:
        """Pre-step gate: would the run already be over budget?"""
        if self.max_total_cost_usd is not None and self.cost_usd >= self.max_total_cost_usd:
            return False, f"run cost ${self.cost_usd:.4f} hit cap ${self.max_total_cost_usd:.4f}"
        if self.max_total_tokens is not None and self.tokens_used >= self.max_total_tokens:
            return False, f"run tokens {self.tokens_used} hit cap {self.max_total_tokens}"
        if self.max_total_latency_ms is not None and self.latency_ms >= self.max_total_latency_ms:
            return False, f"run latency {self.latency_ms:.0f}ms hit cap {self.max_total_latency_ms:.0f}ms"
        if step_max_cost is not None and self.max_total_cost_usd is not None \
                and self.cost_usd + step_max_cost > self.max_total_cost_usd:
            return False, f"step cap ${step_max_cost:.4f} would breach run cap"
        return True, ""

    def account(self, tokens: int = 0, cost_usd: float = 0.0, latency_ms: float = 0.0) -> None:
        self.tokens_used += tokens
        self.cost_usd += cost_usd
        self.latency_ms += latency_ms

    def summary(self) -> dict[str, Any]:
        return {"tokens_used": self.tokens_used, "cost_usd": round(self.cost_usd, 6),
                "latency_ms": round(self.latency_ms, 2)}
