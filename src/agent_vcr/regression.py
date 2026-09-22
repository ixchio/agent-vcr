"""Regression engine for Agent CI (Phase 2, design doc §10).

Pipeline: ``candidate .vcr`` × ``dataset golden`` → per-case verdicts
(``PASS`` / ``DEGRADED`` / ``FAIL``) → aggregate report with cost/latency
deltas and a ``LOW`` / ``MEDIUM`` / ``HIGH`` risk rating.

Regression dimensions (§10.2): output correctness, tool correctness,
state transitions, cost, latency, token usage, failure rate, policy
violations (forbidden tools), checkpoint coverage.

All evaluation is offline and deterministic: no LLM calls, no network.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_vcr.datasets import ReplayDataset, TestCase
from agent_vcr.models import FrameType, StateDiff

logger = logging.getLogger(__name__)


class CaseVerdict(str, Enum):
    PASS = "pass"
    DEGRADED = "degraded"
    FAIL = "fail"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SessionDiff(BaseModel):
    """Baseline-vs-candidate comparison (§10.2 regression dimensions)."""

    baseline_frames: int = 0
    candidate_frames: int = 0
    states_match: bool = True
    state_diff: dict[str, Any] = Field(default_factory=dict)
    added_tools: list[str] = Field(default_factory=list)
    removed_tools: list[str] = Field(default_factory=list)
    added_models: list[str] = Field(default_factory=list)
    removed_models: list[str] = Field(default_factory=list)
    cost_delta_usd: float = 0.0
    cost_delta_pct: float | None = None
    latency_delta_ms: float = 0.0
    latency_delta_pct: float | None = None
    token_delta: int = 0
    token_delta_pct: float | None = None
    baseline_errors: int = 0
    candidate_errors: int = 0
    missing_checkpoints: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Verdict for one dataset case."""

    case: str
    verdict: CaseVerdict = CaseVerdict.PASS
    reasons: list[str] = Field(default_factory=list)
    diff: SessionDiff = Field(default_factory=SessionDiff)


class RegressionReport(BaseModel):
    """Full Agent CI report — JSON-serializable, PR-comment renderable."""

    dataset: str
    dataset_version: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    passed: int = 0
    degraded: int = 0
    failed: int = 0
    missing: int = 0
    missing_cases: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.LOW
    total_cost_delta_usd: float = 0.0
    total_latency_delta_ms: float = 0.0
    results: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results) + self.missing

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.model_dump(), indent=2) + "\n")
        return out

    def to_markdown(self) -> str:
        """PR-comment body in the design-doc §10.1 format."""
        lines = [
            "## Agent CI — replay regression report",
            "",
            f"Dataset `{self.dataset}` (v{self.dataset_version}) · "
            f"{self.total} cases: "
            f"**{self.passed} passed**, {self.degraded} degraded, "
            f"{self.failed} failed{(', ' + str(self.missing) + ' missing') if self.missing else ''}",
            "",
            f"**Risk: {self.risk.value}** · "
            f"cost Δ `${self.total_cost_delta_usd:+.4f}` · "
            f"latency Δ `{self.total_latency_delta_ms:+.0f}ms`",
            "",
            "| case | verdict | cost Δ | latency Δ | tokens Δ | notes |",
            "|---|---|---|---|---|---|",
        ]
        for res in self.results:
            d = res.diff
            cost = f"{d.cost_delta_usd:+.4f}" if d.cost_delta_pct is not None else "n/a"
            lat = f"{d.latency_delta_ms:+.0f}ms" if d.latency_delta_pct is not None else "n/a"
            tok = f"{d.token_delta:+d}" if d.token_delta_pct is not None else "n/a"
            notes = "; ".join(res.reasons) if res.reasons else "—"
            icon = {"pass": "✅", "degraded": "⚠️", "fail": "❌"}[res.verdict.value]
            lines.append(f"| {res.case} | {icon} {res.verdict.value} | {cost} | {lat} | {tok} | {notes} |")
        if self.missing_cases:
            lines.append(f"| _missing: {', '.join(self.missing_cases)}_ | ❌ no candidate | — | — | — | no candidate recording supplied |")
        lines += ["", "_Generated offline by `vcr ci` — zero LLM cost (Ghost Replay)._"]
        return "\n".join(lines) + "\n"


# -- session introspection -------------------------------------------------


def _session_facts(player: Any) -> dict[str, Any]:
    tools: list[str] = []
    models: list[str] = []
    errors = 0
    for frame in player.frames:
        if frame.node_name.startswith("tool:"):
            tool = frame.node_name.split("tool:", 1)[1]
            if tool not in tools:
                tools.append(tool)
        if frame.metadata and frame.metadata.model and frame.metadata.model not in models:
            models.append(frame.metadata.model)
        if frame.frame_type == FrameType.ERROR:
            errors += 1
    final_state: dict[str, Any] = {}
    if player.frames:
        from agent_vcr.models import StateSerializer

        final_state = StateSerializer.deserialize(player.frames[-1].output_state)
    checkpoints = {
        (cp.get("label") or cp.get("node_name")) for cp in player.list_checkpoints()
    }
    return {
        "frames": len(player.frames),
        "tools": tools,
        "models": models,
        "errors": errors,
        "final_state": final_state if isinstance(final_state, dict) else {},
        "checkpoints": {c for c in checkpoints if c},
        "cost": player.get_total_cost(),
        "latency": player.get_total_latency(),
        "tokens": player.get_total_tokens(),
    }


def _pct(new: float, old: float) -> float | None:
    if old == 0:
        return None if new == 0 else float("inf")
    return (new - old) / abs(old)


def compare_sessions(baseline: Any, candidate: Any) -> SessionDiff:
    """Compare a golden baseline player against a candidate player."""
    base = _session_facts(baseline)
    cand = _session_facts(candidate)
    state_diff = StateDiff.compute(base["final_state"], cand["final_state"])
    states_match = not (state_diff["added"] or state_diff["removed"] or state_diff["modified"])
    baseline_cps = set(base["checkpoints"])
    return SessionDiff(
        baseline_frames=base["frames"],
        candidate_frames=cand["frames"],
        states_match=states_match,
        state_diff=state_diff,
        added_tools=[t for t in cand["tools"] if t not in base["tools"]],
        removed_tools=[t for t in base["tools"] if t not in cand["tools"]],
        added_models=[m for m in cand["models"] if m not in base["models"]],
        removed_models=[m for m in base["models"] if m not in cand["models"]],
        cost_delta_usd=cand["cost"] - base["cost"],
        cost_delta_pct=_pct(cand["cost"], base["cost"]),
        latency_delta_ms=cand["latency"] - base["latency"],
        latency_delta_pct=_pct(cand["latency"], base["latency"]),
        token_delta=cand["tokens"] - base["tokens"],
        token_delta_pct=_pct(float(cand["tokens"]), float(base["tokens"])),
        baseline_errors=base["errors"],
        candidate_errors=cand["errors"],
        missing_checkpoints=sorted(baseline_cps - set(cand["checkpoints"])),
    )


# -- verdicts ---------------------------------------------------------------


def _within_subset(expected: Any, actual: Any) -> tuple[bool, str]:
    """Partial-match: every key/value in ``expected`` must equal ``actual``."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"expected object, got {type(actual).__name__}"
        for key, val in expected.items():
            if key not in actual:
                return False, f"missing key {key!r}"
            ok, reason = _within_subset(val, actual[key])
            if not ok:
                return False, f"{key!r}: {reason}"
        return True, ""
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) < len(expected):
            return False, f"expected list prefix of len {len(expected)}"
        for i, val in enumerate(expected):
            ok, reason = _within_subset(val, actual[i])
            if not ok:
                return False, f"[{i}]: {reason}"
        return True, ""
    if expected != actual:
        return False, f"expected {expected!r}, got {actual!r}"
    return True, ""


def evaluate_candidate(
    case: TestCase, candidate: Any, baseline: Any | None = None
) -> CaseResult:
    """Apply a case's annotations (+ optional baseline regression gates)."""
    fails: list[str] = []
    degradations: list[str] = []
    cand = _session_facts(candidate)
    diff = compare_sessions(baseline, candidate) if baseline is not None else SessionDiff()

    exp = case.expect
    if exp.require_no_errors and cand["errors"] > 0:
        fails.append(f"{cand['errors']} error frame(s)")
    if exp.max_cost_usd is not None and cand["cost"] > exp.max_cost_usd:
        fails.append(f"cost ${cand['cost']:.4f} > budget ${exp.max_cost_usd:.4f}")
    if exp.max_latency_ms is not None and cand["latency"] > exp.max_latency_ms:
        fails.append(f"latency {cand['latency']:.0f}ms > budget {exp.max_latency_ms:.0f}ms")
    if exp.max_tokens is not None and cand["tokens"] > exp.max_tokens:
        fails.append(f"tokens {cand['tokens']} > budget {exp.max_tokens}")
    if exp.expect_state:
        ok, reason = _within_subset(exp.expect_state, cand["final_state"])
        if not ok:
            fails.append(f"final state mismatch: {reason}")
    forbidden = [t for t in exp.forbid_tools if t in cand["tools"]]
    if forbidden:
        fails.append(f"forbidden tool(s) used: {', '.join(forbidden)}")
    missing_cps = [c for c in exp.require_checkpoints if c not in cand["checkpoints"]]
    if missing_cps:
        fails.append(f"missing checkpoint(s): {', '.join(missing_cps)}")

    if baseline is not None:
        cost_pct = diff.cost_delta_pct or 0.0
        lat_pct = diff.latency_delta_pct or 0.0
        tok_pct = diff.token_delta_pct or 0.0
        if exp.max_cost_delta is not None and cost_pct > exp.max_cost_delta:
            degradations.append(f"cost +{cost_pct * 100:.0f}% vs baseline")
        if exp.max_latency_delta is not None and lat_pct > exp.max_latency_delta:
            degradations.append(f"latency +{lat_pct * 100:.0f}% vs baseline")
        if exp.max_token_delta is not None and tok_pct > exp.max_token_delta:
            degradations.append(f"tokens +{tok_pct * 100:.0f}% vs baseline")
        if diff.added_tools:
            degradations.append(f"new tool(s) vs baseline: {', '.join(diff.added_tools)}")
        if diff.added_models:
            degradations.append(f"new model(s) vs baseline: {', '.join(diff.added_models)}")
        if diff.missing_checkpoints:
            degradations.append(
                f"dropped checkpoint(s): {', '.join(diff.missing_checkpoints)}"
            )
        if not diff.states_match:
            degradations.append("final state differs from baseline")

    verdict = CaseVerdict.FAIL if fails else (CaseVerdict.DEGRADED if degradations else CaseVerdict.PASS)
    return CaseResult(case=case.name, verdict=verdict, reasons=fails + degradations, diff=diff)


def _risk_for(report: RegressionReport) -> RiskLevel:
    if report.failed > 0:
        return RiskLevel.HIGH
    if report.degraded > 0 or report.missing > 0:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def run_regression(
    dataset: ReplayDataset,
    candidates: dict[str, Any],
    golden_cache: Any,
    *,
    tags: list[str] | None = None,
) -> RegressionReport:
    """Evaluate candidates against dataset goldens.

    Args:
        dataset: Dataset whose cases define goldens + expectations.
        candidates: Mapping of case name → loaded player (or ``.vcr`` path).
        golden_cache: GoldenRunCache holding the dataset's golden runs.
        tags: Optional tag filter (subset of ``ReplayDataset.select``).
    """
    from agent_vcr.player import VCRPlayer

    report = RegressionReport(dataset=dataset.name, dataset_version=dataset.version)
    for case in dataset.select(tags):
        raw = candidates.get(case.name)
        if raw is None:
            report.missing += 1
            report.missing_cases.append(case.name)
            continue
        candidate = VCRPlayer.load(raw) if isinstance(raw, (str, Path)) else raw
        info = golden_cache.get_golden_info(case.task, case.identity)
        if info is None:
            report.results.append(
                CaseResult(
                    case=case.name,
                    verdict=CaseVerdict.FAIL,
                    reasons=["golden run missing from cache — re-add the case"],
                )
            )
            continue
        baseline = VCRPlayer.load(info["file"])
        result = evaluate_candidate(case, candidate, baseline)
        report.results.append(result)
        if result.verdict == CaseVerdict.PASS:
            report.passed += 1
        elif result.verdict == CaseVerdict.DEGRADED:
            report.degraded += 1
        else:
            report.failed += 1
        report.total_cost_delta_usd += result.diff.cost_delta_usd
        report.total_latency_delta_ms += result.diff.latency_delta_ms

    report.risk = _risk_for(report)
    return report


def write_github_summary(markdown: str) -> None:
    """Append to the GitHub Actions step summary when running in CI."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a") as f:
        f.write(markdown + "\n")


def emit_github_annotations(report: RegressionReport) -> None:
    """Emit ::warning::/::error:: workflow commands for degraded/failed cases."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for res in report.results:
        if res.verdict == CaseVerdict.FAIL:
            print(f"::error::AgentCI {res.case}: {'; '.join(res.reasons)}")
        elif res.verdict == CaseVerdict.DEGRADED:
            print(f"::warning::AgentCI {res.case}: {'; '.join(res.reasons)}")
