"""Every place that shows the release version agrees with the package version."""

from __future__ import annotations

import re
from pathlib import Path

import agent_vcr
import openhands_sentinel

ROOT = Path(__file__).resolve().parents[2]


def test_package_versions_match_pyproject() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    assert match is not None
    assert agent_vcr.__version__ == match.group(1)
    assert openhands_sentinel.__version__ == match.group(1)


def test_dashboard_sidebar_shows_the_package_version() -> None:
    bundle = "".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "src/agent_vcr/dashboard/assets").glob("index-*.js")
    )
    shown = re.findall(r"sidebar-logo-version`,children:`v([^`]+)`", bundle)
    assert shown == [agent_vcr.__version__]


def test_docs_pin_the_current_release() -> None:
    version = agent_vcr.__version__
    agent_ci = (ROOT / "docs/agent-ci.md").read_text(encoding="utf-8")
    assert f"ixchio/agent-vcr@v{version}" in agent_ci
    assert f"ai-agent-vcr=={version}" in agent_ci
    assert f"# vcr {version}" in (ROOT / "docs/quickstart.md").read_text(encoding="utf-8")
    assert f"**Version:** `{version}`" in (ROOT / "docs/architecture.md").read_text(
        encoding="utf-8")
