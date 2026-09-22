"""Output verification for plan steps (Phase 4).

A step's ``check`` spec declares what "done" means; :class:`Verifier`
evaluates it against the step output and returns pass/fail with reasons.
Supported check keys (combinable; empty spec always passes):

- ``equals``: exact dict equality with the output.
- ``contains``: partial (subset) match — every listed key/value present.
- ``no_error``: output must not contain an ``error`` key / result error.
- ``required_keys``: list of keys that must exist in the output.
- ``forbid_keys``: list of keys that must NOT exist in the output.
"""

from __future__ import annotations

from typing import Any

from agent_vcr.regression import _within_subset as _is_subset  # shared subset matcher
from agent_vcr.superagent.models import StepResult


class Verdict:
    """Result of one verification (kept light — not a pydantic model)."""

    def __init__(self, passed: bool, reasons: list[str] | None = None):
        self.passed = passed
        self.reasons = reasons or []

    def __bool__(self) -> bool:
        return self.passed


class Verifier:
    """Evaluate step ``check`` specs against outputs."""

    def verify(self, check: dict[str, Any], result: StepResult) -> Verdict:
        reasons: list[str] = []
        output = result.output

        # An error envelope is never a successful result.  Without this
        # guard an empty check spec marked ``StepResult(error=...)`` verified
        # before the executor aborted later, leaving a misleading trace.
        if result.error:
            reasons.append(f"step errored: {result.error}")
        if not check:
            return Verdict(not reasons, reasons)
        if "equals" in check and output != check["equals"]:
            reasons.append(f"output != expected: {output!r}")
        if "contains" in check:
            ok, reason = _is_subset(check["contains"], output)
            if not ok:
                reasons.append(f"output missing {reason}")
        for key in check.get("required_keys", []):
            if key not in output:
                reasons.append(f"required key {key!r} missing")
        for key in check.get("forbid_keys", []):
            if key in output:
                reasons.append(f"forbidden key {key!r} present")

        return Verdict(not reasons, reasons)
