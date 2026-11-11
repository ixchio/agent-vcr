"""Working memory with secret redaction + JSON persistence (Phase 4).

:class:`MemoryStore` is the agent's scratchpad: step-scoped writes,
namespaced reads, substring search, and snapshot/restore for
fork/rollback flows. Values that look like secrets (API keys, tokens,
passwords, private keys) are redacted in `redacted()` views and in
the persisted file — the raw process memory is the caller's
responsibility, the *recorded* memory never leaks credentials.

Redaction covers two cases:

- **Sensitive keys** — any value stored under a key such as ``password``,
  ``api_key``, ``secret``, ``token``, ``authorization`` or ``cookie``
  (case-insensitive, camelCase/kebab-case aware, at any nesting depth) is
  replaced wholesale. Numbers and booleans are kept (``max_tokens``-style
  counters are not secrets).
- **Secret-looking text** — ``Bearer``/``Basic`` credentials, ``key=value``
  assignments, JSON ``"password": "..."`` pairs, PEM private-key blocks and
  well-known token formats (``sk-…``, ``AKIA…``, ``ghp_…``, ``xox…-``, JWTs).

This is working memory, not the vector long-term store (Phase 6).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

_KEY_WORDS = (
    r"password|passwd|passphrase|secrets?|token|api_?key|apikey|access_?key|"
    r"secret_?key|private_?key|client_?secret|authorization|credentials?|"
    r"cookie|bearer"
)
#: Key names (after camelCase/kebab normalisation) whose values are secrets.
#: ``auth`` only matches as the whole key (``auth_method`` is not a secret).
_SENSITIVE_KEY = re.compile(rf"(?:^|_)(?:{_KEY_WORDS})(?:$|_)|^auth$")

_TEXT_KEYS = (
    r"api[_-]?key|apikey|x-api-key|access[_-]?key|secret[_-]?key|client[_-]?secret|"
    r"secret|passwd|password|token|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"authorization"
)

# Order matters: whole PEM blocks and HTTP auth schemes first, then the
# generic ``key: value`` forms, then free-floating token formats.
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)",
    re.S,
)
_AUTH_SCHEME = re.compile(r"\b(Bearer|bearer|BEARER|Basic|basic|BASIC)(\s+)([A-Za-z0-9._~+/-]+=*)")
_JSON_PAIR = re.compile(rf"(?i)(\"(?:{_TEXT_KEYS})\"\s*:\s*)\"(?:[^\"\\]|\\.)*\"")
_ASSIGNMENT = re.compile(rf"(?i)\b(?:{_TEXT_KEYS})\b\s*[:=]\s*(?!\[REDACTED\])\S+")
_TOKEN_FORMATS = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_-]{8,}"  # OpenAI / Anthropic style
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"  # AWS access key ids
    r"|gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{20,}"  # GitHub
    r"|xox[abposr]-[A-Za-z0-9-]{8,}"  # Slack
    r"|AIza[0-9A-Za-z_-]{30,}"  # Google API keys
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"  # JWT
    r")"
)

# Backwards-compatible export (some callers iterate the pattern list).
_SECRET_PATTERNS = [_PRIVATE_KEY_BLOCK, _JSON_PAIR, _ASSIGNMENT, _TOKEN_FORMATS]


def _looks_like_credential(token: str) -> bool:
    """Heuristic for the word after ``Bearer``/``Basic`` (avoid redacting prose)."""
    if len(token) < 6:
        return False
    if len(token) >= 24 or any(ch.isdigit() or ch in "+/=_.~-" for ch in token):
        return True
    return any(ch.isupper() for ch in token[1:])  # base64 is mixed-case


def _auth_scheme_sub(match: re.Match[str]) -> str:
    scheme, space, token = match.group(1), match.group(2), match.group(3)
    if not _looks_like_credential(token):
        return match.group(0)
    return f"{scheme}{space}{REDACTED}"


def is_sensitive_key(key: Any) -> bool:
    """True if values stored under ``key`` must be redacted (case-insensitive)."""
    if not isinstance(key, str) or not key:
        return False
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")
    return bool(_SENSITIVE_KEY.search(normalized))


def redact_text(text: str) -> str:
    """Replace secret-looking spans with [REDACTED] (pure function)."""
    redacted = _PRIVATE_KEY_BLOCK.sub(REDACTED, text)
    redacted = _AUTH_SCHEME.sub(_auth_scheme_sub, redacted)
    redacted = _JSON_PAIR.sub(lambda m: f'{m.group(1)}"{REDACTED}"', redacted)
    redacted = _ASSIGNMENT.sub(REDACTED, redacted)
    return _TOKEN_FORMATS.sub(REDACTED, redacted)


def _redact_secret_value(value: Any) -> Any:
    # Counters/flags under a sensitive-looking key (``token_count``) are not secrets.
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return REDACTED


def redact_value(value: Any) -> Any:
    """Deep-redact secrets inside nested structures (keys and values)."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            k: _redact_secret_value(v) if is_sensitive_key(k) else redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    return value


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a sibling temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class MemoryStore:
    """Namespaced working memory: `<step-id>.<key>` + globals (thread-safe)."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._namespaces: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def put(self, key: str, value: Any, namespace: str = "global") -> None:
        with self._lock:
            if namespace == "global":
                self._data[key] = value
            else:
                self._namespaces.setdefault(namespace, {})[key] = value

    def get(self, key: str, default: Any = None, namespace: str = "global") -> Any:
        with self._lock:
            if namespace == "global":
                return self._data.get(key, default)
            return self._namespaces.get(namespace, {}).get(key, default)

    def snapshot(self, namespace: str) -> dict[str, Any]:
        """Copy one step's outputs for downstream `$ref` resolution."""
        with self._lock:
            return dict(self._namespaces.get(namespace, {}))

    def search(self, query: str) -> list[tuple[str, Any]]:
        """Case-insensitive substring search; returned values are redacted."""
        q = query.lower()
        hits: list[tuple[str, Any]] = []
        with self._lock:
            entries = [("global", k, v) for k, v in self._data.items()]
            entries += [(ns, k, v) for ns, items in self._namespaces.items()
                        for k, v in items.items()]
        for ns, key, value in entries:
            if q in key.lower() or q in json.dumps(value, default=str).lower():
                clean = redact_value({key: value})[key]
                hits.append((f"{ns}.{key}", clean))
        return hits

    def redacted(self) -> dict[str, Any]:
        """Full memory view safe to record/log (secrets redacted)."""
        with self._lock:
            return {
                "global": redact_value(self._data),
                "namespaces": {ns: redact_value(v) for ns, v in self._namespaces.items()},
            }

    def save(self, path: str | Path) -> Path:
        """Persist the REDACTED view atomically (credentials never touch disk)."""
        out = Path(path)
        atomic_write_text(out, json.dumps(self.redacted(), indent=2, default=str) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> MemoryStore:
        store = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store._data = payload.get("global", {})
        store._namespaces = payload.get("namespaces", {})
        return store
