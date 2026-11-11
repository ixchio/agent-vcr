"""
Code Analyzer — Pure AST-based static analysis engine.

Zero external dependencies. Uses Python's built-in `ast` module to detect:
- Duplicate function definitions across files
- Function length explosion (> configurable threshold)
- Cyclomatic complexity spikes
- File size growth rate anomalies
- Class bloat detection

This is NOT a linter. This is trajectory-aware analysis.
A linter checks one file. Sentinel tracks the agent's behavior ACROSS
multiple actions and catches patterns that only emerge over time.
"""

from __future__ import annotations

import ast
import enum
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Union

_FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]

logger = logging.getLogger(__name__)


class Severity(str, enum.Enum):
    """Violation severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKER = "blocker"


@dataclass
class Violation:
    """A single code quality violation detected by the analyzer."""
    rule: str
    message: str
    severity: Severity
    file_path: str
    line: int | None = None
    function_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "message": self.message,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line": self.line,
            "function_name": self.function_name,
            "details": self.details,
        }

    def to_agent_warning(self) -> str:
        """Format as a terse warning the LLM agent can act on."""
        prefix = f"⚠️ SENTINEL [{self.severity.value.upper()}]"
        location = f"{self.file_path}"
        if self.line:
            location += f":{self.line}"
        return f"{prefix} {location} — {self.message}"


@dataclass
class AnalysisResult:
    """Aggregated result of analyzing a code change."""
    file_path: str
    violations: list[Violation] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Set by Sentinel when a BLOCKER was found and ``block_on_blocker`` is on:
    # the action that produced this content should not be allowed to proceed.
    blocked: bool = False

    @property
    def passed(self) -> bool:
        return not any(
            v.severity in (Severity.CRITICAL, Severity.BLOCKER)
            for v in self.violations
        )

    @property
    def blocker_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.BLOCKER)

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.WARNING)


@dataclass
class _FunctionInfo:
    """A function definition plus the scope it was found in."""
    node: _FunctionNode
    qualname: str  # e.g. "helper", "Repo.save", "outer.<locals>.inner"
    kind: str  # "function" (module level), "method", or "nested"
    is_static: bool = False


def _iter_functions(tree: ast.Module) -> list[_FunctionInfo]:
    """Collect every function with its qualified name (iterative, source order)."""
    found: list[_FunctionInfo] = []
    # (node, scope parts, kind of the innermost enclosing scope)
    stack: list[tuple[ast.AST, tuple[str, ...], str]] = [(tree, (), "module")]
    while stack:
        node, scope, parent = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join((*scope, child.name))
                if parent == "module":
                    kind = "function"
                elif parent == "class":
                    kind = "method"
                else:
                    kind = "nested"
                is_static = any(
                    isinstance(d, ast.Name) and d.id == "staticmethod"
                    for d in child.decorator_list
                )
                found.append(_FunctionInfo(child, qualname, kind, is_static))
                stack.append((child, (*scope, child.name, "<locals>"), "function"))
            elif isinstance(child, ast.ClassDef):
                stack.append((child, (*scope, child.name), "class"))
            else:
                stack.append((child, scope, parent))
    found.sort(key=lambda info: (info.node.lineno, info.node.col_offset))
    return found


def _count_params(info: _FunctionInfo) -> int:
    """Count every parameter kind, excluding the implicit ``self``/``cls`` of methods."""
    args = info.node.args
    positional = [*args.posonlyargs, *args.args]
    implicit = ("self", "cls", "mcs", "metacls")
    if (
        info.kind == "method"
        and not info.is_static
        and positional
        and positional[0].arg in implicit
    ):
        positional = positional[1:]
    count = len(positional) + len(args.kwonlyargs)
    count += int(args.vararg is not None) + int(args.kwarg is not None)
    return count


def _body_hash(node: _FunctionNode) -> str:
    """Hash of the normalized AST (signature + body), ignoring name, position and comments."""
    dumped = ast.dump(node.args) + "|" + "|".join(ast.dump(stmt) for stmt in node.body)
    if node.returns is not None:
        dumped += "|->" + ast.dump(node.returns)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


_DECISION_NODES = (
    ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
    ast.With, ast.AsyncWith, ast.Assert,
)


class CodeAnalyzer:
    """
    Trajectory-aware code quality analyzer.

    Unlike a linter that checks files in isolation, CodeAnalyzer maintains
    a memory of ALL functions the agent has written across ALL files during
    a session. This allows detection of cross-file duplicates and patterns
    that only emerge over multiple agent actions.
    """

    def __init__(
        self,
        max_function_lines: int = 50,
        max_complexity: int = 10,
        max_file_lines: int = 500,
        max_class_methods: int = 15,
        max_function_params: int = 7,
        min_growth_lines: int = 50,
    ) -> None:
        self.max_function_lines = max_function_lines
        self.max_complexity = max_complexity
        self.max_file_lines = max_file_lines
        self.max_class_methods = max_class_methods
        self.max_function_params = max_function_params
        # A single edit must add at least this many lines (and grow > 200%)
        # before it is reported as rapid growth.
        self.min_growth_lines = min_growth_lines

        # Trajectory memory — persists across multiple analyze() calls
        # Maps qualified function name ("func" / "Class.method")
        #   -> [(file_path, line_number, normalized_ast_hash)]
        self._known_functions: dict[str, list[tuple[str, int, str]]] = {}
        # Maps class_name -> (file_path, line_number)
        self._known_classes: dict[str, list[tuple[str, int]]] = {}
        # File size history for growth rate tracking
        self._file_history: dict[str, list[int]] = {}

    def analyze(self, file_path: str, content: str) -> AnalysisResult:
        """
        Analyze a Python file's content for quality violations.

        Args:
            file_path: Path to the file being written/modified.
            content: The full content of the file.

        Returns:
            AnalysisResult with all detected violations and metrics.
        """
        result = AnalysisResult(file_path=file_path)

        # Only analyze Python files
        if not file_path.endswith(".py"):
            return result

        # Track file size history
        line_count = content.count("\n") + 1
        self._file_history.setdefault(file_path, []).append(line_count)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            result.violations.append(Violation(
                rule="syntax_error",
                message=f"Syntax error: {e.msg}",
                severity=Severity.BLOCKER,
                file_path=file_path,
                line=e.lineno,
            ))
            return result

        # The new content replaces everything this file previously defined.
        self._drop_definitions(file_path)

        # Run all checks
        self._check_file_size(file_path, content, line_count, result)
        self._check_file_growth_rate(file_path, result)
        self._check_functions(file_path, content, tree, result)
        self._check_classes(file_path, tree, result)
        self._check_imports(file_path, tree, result)

        # Compute aggregate metrics
        result.metrics = {
            "line_count": line_count,
            "function_count": sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            "class_count": sum(
                1 for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            ),
            "import_count": sum(
                1 for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ),
        }

        return result

    def get_trajectory_summary(self) -> dict[str, Any]:
        """Get a summary of the analyzer's trajectory memory."""
        return {
            "tracked_functions": len(self._known_functions),
            "tracked_classes": len(self._known_classes),
            "tracked_files": len(self._file_history),
            "duplicate_candidates": sum(
                1 for locations in self._known_functions.values()
                if len(locations) > 1
            ),
        }

    def forget_file(self, file_path: str) -> None:
        """Drop all trajectory memory for a file (e.g. after it was deleted)."""
        self._drop_definitions(file_path)
        self._file_history.pop(file_path, None)

    def _drop_definitions(self, file_path: str) -> None:
        """Remove a file's functions/classes from the cross-file duplicate index."""
        for name in list(self._known_functions):
            kept = [loc for loc in self._known_functions[name] if loc[0] != file_path]
            if kept:
                self._known_functions[name] = kept
            else:
                del self._known_functions[name]
        for cls_name in list(self._known_classes):
            kept_cls = [loc for loc in self._known_classes[cls_name] if loc[0] != file_path]
            if kept_cls:
                self._known_classes[cls_name] = kept_cls
            else:
                del self._known_classes[cls_name]

    def reset(self) -> None:
        """Clear trajectory memory. Call between unrelated sessions."""
        self._known_functions.clear()
        self._known_classes.clear()
        self._file_history.clear()

    # ──────────────────────────────────────────────
    #  Individual Checks
    # ──────────────────────────────────────────────

    def _check_file_size(
        self, file_path: str, content: str, line_count: int, result: AnalysisResult
    ) -> None:
        """Check if the file exceeds the maximum line count."""
        if line_count > self.max_file_lines:
            result.violations.append(Violation(
                rule="file_too_large",
                message=(
                    f"File is {line_count} lines (max {self.max_file_lines}). "
                    f"Consider splitting into smaller modules."
                ),
                severity=Severity.WARNING if line_count < self.max_file_lines * 2 else Severity.CRITICAL,
                file_path=file_path,
                details={"line_count": line_count, "max": self.max_file_lines},
            ))

    def _check_file_growth_rate(self, file_path: str, result: AnalysisResult) -> None:
        """Detect suspiciously rapid file growth in a single agent edit.

        Growth is measured against the previously analyzed version of the file
        (per-edit delta), and only edits that add at least ``min_growth_lines``
        lines are reported, so filling in a small stub is not flagged.
        """
        history = self._file_history.get(file_path, [])
        if len(history) < 2:
            return

        previous = history[-2]
        current = history[-1]
        added = current - previous
        if previous <= 0 or added < self.min_growth_lines:
            return
        growth_pct = (added / previous) * 100
        if growth_pct > 200:
            result.violations.append(Violation(
                rule="rapid_growth",
                message=(
                    f"File grew {growth_pct:.0f}% ({previous} → {current} lines) "
                    f"in a single agent step. The agent may be dumping "
                    f"monolithic code instead of modularizing."
                ),
                severity=Severity.CRITICAL,
                file_path=file_path,
                details={
                    "initial_lines": history[0],
                    "previous_lines": previous,
                    "current_lines": current,
                    "added_lines": added,
                    "growth_percent": round(growth_pct, 1),
                    "steps": len(history),
                },
            ))

    def _check_functions(
        self, file_path: str, content: str, tree: ast.Module, result: AnalysisResult
    ) -> None:
        """Check all function definitions for violations."""
        for info in _iter_functions(tree):
            node = info.node

            # Skip dunder methods — they have valid reasons to be long
            if node.name.startswith("__") and node.name.endswith("__"):
                continue

            # --- Function length ---
            if node.end_lineno and node.lineno:
                length = node.end_lineno - node.lineno + 1
                if length > self.max_function_lines:
                    severity = Severity.WARNING
                    if length > self.max_function_lines * 2:
                        severity = Severity.CRITICAL
                    if length > self.max_function_lines * 4:
                        severity = Severity.BLOCKER

                    result.violations.append(Violation(
                        rule="function_too_long",
                        message=(
                            f"`{info.qualname}()` is {length} lines "
                            f"(max {self.max_function_lines}). Break it up."
                        ),
                        severity=severity,
                        file_path=file_path,
                        line=node.lineno,
                        function_name=node.name,
                        details={
                            "length": length,
                            "max": self.max_function_lines,
                            "qualified_name": info.qualname,
                        },
                    ))

            # --- Too many parameters ---
            param_count = _count_params(info)
            if param_count > self.max_function_params:
                result.violations.append(Violation(
                    rule="too_many_params",
                    message=(
                        f"`{info.qualname}()` has {param_count} parameters "
                        f"(max {self.max_function_params}). Use a config object."
                    ),
                    severity=Severity.WARNING,
                    file_path=file_path,
                    line=node.lineno,
                    function_name=node.name,
                    details={
                        "param_count": param_count,
                        "max": self.max_function_params,
                        "qualified_name": info.qualname,
                    },
                ))

            # --- Cyclomatic complexity ---
            complexity = self._compute_complexity(node)
            if complexity > self.max_complexity:
                result.violations.append(Violation(
                    rule="high_complexity",
                    message=(
                        f"`{info.qualname}()` has cyclomatic complexity {complexity} "
                        f"(max {self.max_complexity}). Simplify the logic."
                    ),
                    severity=(
                        Severity.WARNING
                        if complexity < self.max_complexity * 2
                        else Severity.CRITICAL
                    ),
                    file_path=file_path,
                    line=node.lineno,
                    function_name=node.name,
                    details={
                        "complexity": complexity,
                        "max": self.max_complexity,
                        "qualified_name": info.qualname,
                    },
                ))

            self._check_duplicate(file_path, info, result)

    def _check_duplicate(
        self, file_path: str, info: _FunctionInfo, result: AnalysisResult
    ) -> None:
        """Trajectory-aware cross-file duplicate detection, keyed by qualified name.

        - Module-level functions: same name in another file is a duplicate.
          CRITICAL when the normalized AST is identical (copy-paste), WARNING
          when only the name matches (likely re-implementation or name clash).
        - Methods: flagged only when the same ``Class.method`` exists in another
          file with an identical normalized AST. ``A.run`` vs ``B.run`` is fine.
        - Nested/local functions are not tracked.
        """
        if info.kind == "nested" or "<locals>" in info.qualname:
            return
        node = info.node
        func_hash = _body_hash(node)
        locations = self._known_functions.setdefault(info.qualname, [])

        for prev_path, prev_line, prev_hash in locations:
            if prev_path == file_path:
                continue
            same_body = func_hash == prev_hash
            if info.kind == "method" and not same_body:
                continue
            result.violations.append(Violation(
                rule="duplicate_function",
                message=(
                    f"`{info.qualname}()` already exists in "
                    f"{prev_path}:{prev_line}"
                    + (" with an identical body" if same_body else "")
                    + ". Reuse it instead of duplicating."
                ),
                severity=Severity.CRITICAL if same_body else Severity.WARNING,
                file_path=file_path,
                line=node.lineno,
                function_name=info.qualname,
                details={
                    "original_file": prev_path,
                    "original_line": prev_line,
                    "same_body": same_body,
                    "qualified_name": info.qualname,
                },
            ))

        locations.append((file_path, node.lineno, func_hash))

    def _check_classes(
        self, file_path: str, tree: ast.Module, result: AnalysisResult
    ) -> None:
        """Check class definitions for bloat."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            method_count = sum(
                1 for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )

            if method_count > self.max_class_methods:
                result.violations.append(Violation(
                    rule="class_too_large",
                    message=(
                        f"Class `{node.name}` has {method_count} methods "
                        f"(max {self.max_class_methods}). Consider splitting."
                    ),
                    severity=Severity.WARNING,
                    file_path=file_path,
                    line=node.lineno,
                    details={"method_count": method_count, "max": self.max_class_methods},
                ))

            # Track classes for trajectory awareness
            cls_locations = self._known_classes.setdefault(node.name, [])
            for prev_path, prev_line in cls_locations:
                if prev_path != file_path:
                    result.violations.append(Violation(
                        rule="duplicate_class",
                        message=(
                            f"Class `{node.name}` already exists in "
                            f"{prev_path}:{prev_line}."
                        ),
                        severity=Severity.WARNING,
                        file_path=file_path,
                        line=node.lineno,
                        details={"original_file": prev_path, "original_line": prev_line},
                    ))
            cls_locations[:] = [(p, ln) for p, ln in cls_locations if p != file_path]
            cls_locations.append((file_path, node.lineno))

    def _check_imports(
        self, file_path: str, tree: ast.Module, result: AnalysisResult
    ) -> None:
        """Check for wildcard imports (common agent mistake)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.names:
                for alias in node.names:
                    if alias.name == "*":
                        result.violations.append(Violation(
                            rule="wildcard_import",
                            message=(
                                f"Wildcard import `from {node.module} import *` "
                                f"— import specific names instead."
                            ),
                            severity=Severity.WARNING,
                            file_path=file_path,
                            line=node.lineno,
                        ))

    # ──────────────────────────────────────────────
    #  Complexity Calculator
    # ──────────────────────────────────────────────

    @staticmethod
    def _compute_complexity(node: ast.AST) -> int:
        """
        Compute cyclomatic complexity of a function using AST.

        CC = 1 + number of decision points (if/elif/for/while/except/
        and/or/assert/with/comprehensions).

        This is a simplified but effective approximation that requires
        zero external dependencies.
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, _DECISION_NODES):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each `and`/`or` adds a decision point
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += len(child.generators)

        return complexity
