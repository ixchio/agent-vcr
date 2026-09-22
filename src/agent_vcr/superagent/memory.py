"""Working memory with secret redaction + JSON persistence (Phase 4).

:class:`MemoryStore` is the agent's scratchpad: step-scoped writes,
namespaced reads, substring search, and snapshot/restore for
fork/rollback flows. Values that look like secrets (API keys, tokens,
passwords, private keys) are redacted in `redacted()` views and in
the persisted file — the raw process memory is the caller's
responsibility, the *recorded* memory never leaks credentials.

This is working memory, not the vector long-term store (Phase 6).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|apikey|secret|passwd|password|token|bearer)\b\s*[:=]\s*\S+"),
    re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9]{8,}|xox[bpas]-[A-Za-z0-9-]{8,})\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

REDACTED = "[REDACTED]"


def redact_text(text: str) -> str:
    """Replace secret-looking spans with [REDACTED] (pure function)."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_value(value: Any) -> Any:
    """Deep-redact secrets inside nested structures."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


class MemoryStore:
    """Namespaced working memory: `<step-id>.<key>` + globals."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._namespaces: dict[str, dict[str, Any]] = {}

    def put(self, key: str, value: Any, namespace: str = "global") -> None:
        if namespace == "global":
            self._data[key] = value
        else:
            self._namespaces.setdefault(namespace, {})[key] = value

    def get(self, key: str, default: Any = None, namespace: str = "global") -> Any:
        if namespace == "global":
            return self._data.get(key, default)
        return self._namespaces.get(namespace, {}).get(key, default)

    def snapshot(self, namespace: str) -> dict[str, Any]:
        """Copy one step's outputs for downstream `$ref` resolution."""
        return dict(self._namespaces.get(namespace, {}))

    def search(self, query: str) -> list[tuple[str, Any]]:
        """Case-insensitive substring search over redacted values."""
        q = query.lower()
        hits: list[tuple[str, Any]] = []
        for key, value in self._data.items():
            if q in key.lower() or q in json.dumps(value, default=str).lower():
                hits.append((f"global.{key}", redact_value(value)))
        for ns, entries in self._namespaces.items():
            for key, value in entries.items():
                if q in key.lower() or q in json.dumps(value, default=str).lower():
                    hits.append((f"{ns}.{key}", redact_value(value)))
        return hits

    def redacted(self) -> dict[str, Any]:
        """Full memory view safe to record/log (secrets redacted)."""
        return {
            "global": redact_value(self._data),
            "namespaces": {ns: redact_value(v) for ns, v in self._namespaces.items()},
        }

    def save(self, path: str | Path) -> Path:
        """Persist the REDACTED view (credentials never touch disk)."""
        out = Path(path)
        out.write_text(json.dumps(self.redacted(), indent=2) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        store = cls()
        payload = json.loads(Path(path).read_text())
        store._data = payload.get("global", {})
        store._namespaces = payload.get("namespaces", {})
        return store
