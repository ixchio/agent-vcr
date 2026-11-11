"""Factory for instantiating sandboxes based on configuration and environment (Phase 7)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from agent_vcr.sandbox.base import BaseSandbox
from agent_vcr.sandbox.e2b_adapter import E2BSandbox
from agent_vcr.sandbox.local import LocalSandbox
from agent_vcr.sandbox.models import SandboxConfig

logger = logging.getLogger(__name__)


#: Accepted spellings (case-insensitive) -> canonical SandboxConfig provider.
_PROVIDER_ALIASES = {
    "auto": "auto",
    "e2b": "e2b",
    "local": "local",
    "fallback": "local",
    "mock": "mock",
}


def _normalize_provider(provider: str) -> str:
    key = (provider or "auto").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(
            f"Unknown sandbox provider: {provider!r} (expected one of {sorted(_PROVIDER_ALIASES)})"
        )
    return _PROVIDER_ALIASES[key]


def _e2b_installed() -> bool:
    try:
        import e2b  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def create_sandbox(
    provider: str = "auto",
    config: SandboxConfig | None = None,
    workdir: str | Path | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseSandbox:
    """Create a sandbox instance.

    Supported providers (case-insensitive):
    - 'auto': chooses E2B if e2b is installed and an API key is available
      (``api_key``, ``config.api_key`` or ``E2B_API_KEY``); otherwise LocalSandbox.
    - 'e2b': E2B microVM adapter.
    - 'local' (alias 'fallback'): local scratch-directory execution.
    - 'mock': local execution or custom injected client.

    When ``config`` is given and ``provider`` is 'auto', ``config.provider``
    decides. An explicit ``provider`` argument wins over a conflicting
    ``config.provider`` (with a warning). Extra ``kwargs`` override fields of
    ``config``. The returned sandbox's ``config.provider`` is the resolved
    provider, never 'auto'.
    """
    requested = _normalize_provider(provider)
    if config is not None:
        cfg_provider = _normalize_provider(config.provider)
        if requested == "auto":
            requested = cfg_provider
        elif cfg_provider not in ("auto", requested):
            logger.warning("create_sandbox(provider=%r) overrides config.provider=%r",
                           provider, config.provider)
        cfg = SandboxConfig(**{**config.model_dump(), **kwargs}) if kwargs else config
    else:
        cfg = SandboxConfig(provider=requested, **kwargs)  # type: ignore[arg-type]

    target_provider = requested
    if target_provider == "auto":
        has_api_key = bool(api_key or cfg.api_key or os.environ.get("E2B_API_KEY"))
        target_provider = "e2b" if (has_api_key and _e2b_installed()) else "local"

    if cfg.provider != target_provider:
        cfg = cfg.model_copy(update={"provider": target_provider})

    if target_provider == "e2b":
        return E2BSandbox(config=cfg, client=client, api_key=api_key)
    return LocalSandbox(config=cfg, workdir=workdir)
