"""Plan construction + validation (Phase 4).

:class:`PlanBuilder` is the production gate between "an idea for a plan"
(any dict/JSON, hand-written or LLM-generated) and an executable
:class:`Plan`: unknown refs, dependency cycles, and bound violations are
rejected with precise errors — never executed.

:class:`LLMPlanner` is a real LiteLLM-backed planner: it asks a model for
a plan JSON, parses it, and runs it through the same builder. Invalid
model output triggers a repair retry with the validation errors fed back.
Requires ``litellm`` + model credentials at call time (no key needed to
import or to unit-test parsing via :meth:`LLMPlanner.parse_json`).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent_vcr.superagent.models import Plan

logger = logging.getLogger(__name__)

_PLAN_SYSTEM_PROMPT = """You are a planner for an autonomous agent runtime.
Respond with a single JSON object and nothing else, matching this schema:
{"goal": str, "steps": [{"id": str, "kind": "tool"|"agent", "ref": str,
"inputs": object, "depends_on": [str], "checkpoint_after": bool,
"requires_approval": bool, "timeout_s": number, "max_retries": int,
"on_failure": "abort"|"retry"|"skip", "check": object}],
"max_total_cost_usd": number|null, "max_steps": int|null}
Rules: ids unique; depends_on only references earlier-listed ids (DAG, no cycles);
mark requires_approval=true for payments, deletions, external messages;
set checkpoint_after=true before any irreversible step."""


class PlanBuilder:
    """Validate + finalize raw plan specs into executable plans."""

    def __init__(self, known_tools: set[str] | None = None,
                 known_agents: set[str] | None = None):
        self.known_tools = known_tools or set()
        self.known_agents = known_agents or set()

    def build(self, spec: dict[str, Any] | Plan, goal: str = "") -> Plan:
        plan = spec if isinstance(spec, Plan) else Plan(**spec)
        if goal:
            plan.goal = goal
        self._validate(plan)
        return plan

    def _validate(self, plan: Plan) -> None:
        if not plan.steps:
            raise ValueError("Plan has no steps")
        ids = [s.id for s in plan.steps]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate step ids: {dupes}")
        known = set(ids)
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in known:
                    raise ValueError(f"Step {step.id!r} depends on unknown {dep!r}")
                if dep == step.id:
                    raise ValueError(f"Step {step.id!r} depends on itself")
            if step.kind.value == "tool" and self.known_tools and step.ref not in self.known_tools:
                raise ValueError(f"Step {step.id!r} references unknown tool {step.ref!r}")
            if step.kind.value == "agent" and self.known_agents and step.ref not in self.known_agents:
                raise ValueError(f"Step {step.id!r} references unknown agent {step.ref!r}")
            if step.timeout_s <= 0:
                raise ValueError(f"Step {step.id!r} has non-positive timeout")
            if step.max_retries < 0:
                raise ValueError(f"Step {step.id!r} has negative max_retries")
        self.order(plan)  # raises on cycles
        if plan.max_steps is not None and len(plan.steps) > plan.max_steps:
            raise ValueError(f"Plan has {len(plan.steps)} steps, max is {plan.max_steps}")

    @staticmethod
    def order(plan: Plan) -> list[str]:
        """Topological order of step ids; raises ValueError on cycles."""
        edges = {s.id: set(s.depends_on) for s in plan.steps}
        ordered: list[str] = []
        visiting: set[str] = set()

        def visit(node: str, stack: list[str]) -> None:
            if node in ordered:
                return
            if node in visiting:
                cycle = stack[stack.index(node):] + [node]
                raise ValueError(f"Dependency cycle: {' -> '.join(cycle)}")
            visiting.add(node)
            for dep in sorted(edges[node]):
                visit(dep, stack + [node])
            visiting.discard(node)
            ordered.append(node)

        for step_id in edges:
            visit(step_id, [])
        return ordered

    def step_inputs(self, plan: Plan, step_id: str,
                    outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Resolve a step's inputs: explicit inputs + `$dep.path` references.

        Any string value of the form ``"$<step-id>.<dotted.path>"`` is
        replaced with the referenced value from that step's output.
        """
        step = next(s for s in plan.steps if s.id == step_id)
        return {k: self._resolve(v, outputs) for k, v in step.inputs.items()}

    def _resolve(self, value: Any, outputs: dict[str, dict[str, Any]]) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            ref, _, path = value[1:].partition(".")
            if ref not in outputs:
                raise KeyError(f"Input reference to unknown step output {ref!r}")
            current: Any = outputs[ref]
            for part in path.split(".") if path else []:
                if not isinstance(current, dict) or part not in current:
                    raise KeyError(f"Unresolvable input reference {value!r}")
                current = current[part]
            return current
        if isinstance(value, dict):
            return {k: self._resolve(v, outputs) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v, outputs) for v in value]
        return value


class LLMPlanner:
    """LiteLLM-backed planner producing validated executable plans.

    The model proposes; :class:`PlanBuilder` disposes. On validation
    failure the errors are fed back for up to ``max_repair_attempts``
    additional tries before giving up.
    """

    def __init__(
        self,
        model: str,
        builder: PlanBuilder | None = None,
        max_repair_attempts: int = 1,
        complete_fn: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.builder = builder or PlanBuilder()
        self.max_repair_attempts = max_repair_attempts
        self._complete_fn = complete_fn  # injectable transport (default: litellm)

    def plan(self, goal: str, context: str = "") -> Plan:
        errors = ""
        attempts = 1 + max(0, self.max_repair_attempts)
        last_error = "no attempt"
        for _ in range(attempts):
            try:
                spec = self._propose(goal, context, errors)
                return self.builder.build(spec, goal=goal)
            except (ValueError, KeyError, TypeError) as err:
                last_error = str(err)
                errors = last_error
                logger.warning("Plan proposal failed validation, requesting repair: %s", err)
        raise ValueError(f"Model produced no valid plan after {attempts} attempts: {last_error}")

    def _propose(self, goal: str, context: str, errors: str) -> dict[str, Any]:
        user = f"Goal: {goal}"
        if context:
            user += f"\nContext: {context}"
        if errors:
            user += f"\nYour previous plan was invalid: {errors}\nFix it and respond with corrected JSON only."
        if self._complete_fn is not None:
            response = self._complete_fn(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        else:
            import litellm

            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        return self.parse_json(response.choices[0].message.content or "")

    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        """Strict plan-JSON parsing (pure function — no credentials needed)."""
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Planner did not return JSON: {err}") from err
        if not isinstance(spec, dict) or "steps" not in spec:
            raise ValueError("Planner JSON must be an object with a 'steps' list")
        return spec
