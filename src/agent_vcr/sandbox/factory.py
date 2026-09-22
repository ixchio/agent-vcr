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


def create_sandbox(
    provider: str = "auto",
    config: SandboxConfig | None = None,
    workdir: str | Path | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseSandbox:
    """Create a sandbox instance.

    Supported providers:
    - 'auto': chooses E2B if e2b is installed and E2B_API_KEY is present; otherwise LocalSandbox.
    - 'e2b': E2B microVM adapter.
    - 'local': local isolated directory execution.
    - 'mock': local execution or custom injected client.
    """
    cfg = config or SandboxConfig(provider=provider, **kwargs)  # type: ignore[arg-type]

    target_provider = provider.lower()
    if target_provider == "auto":
        has_api_key = bool(api_key or cfg.api_key or os.environ.get("E2B_API_KEY"))
        try:
            import e2b  # noqa: F401
            e2b_installed = True
        except ImportError:
            e2b_installed = False

        target_provider = "e2b" if (e2b_installed and has_api_key) else "local"

    if target_provider == "e2b":
        return E2BSandbox(config=cfg, client=client, api_key=api_key)
    elif target_provider in ("local", "fallback", "mock"):
        return LocalSandbox(config=cfg, workdir=workdir)
    else:
        raise ValueError(f"Unknown sandbox provider: {provider}")
