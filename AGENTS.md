# AGENTS.md — Repository Knowledge

## Project Overview
- **Name:** Agent VCR (`ai-agent-vcr` on PyPI)
- **Purpose:** ACID transactions, time-travel debugging, and zero-cost Ghost Replay for AI agents
- **Key differentiators:** Filesystem rollback via git (not just state), Ghost Replay (zero-token re-runs), Sentinel (real-time AST quality guard)
- **Frameworks:** LangGraph, CrewAI, raw Python — framework-agnostic
- **Version:** 0.8.0

## Architecture
- `src/agent_vcr/` — Core package: recorder, player, golden_cache, models, server, tui
- `src/openhands_sentinel/` — Sentinel code quality guardian (analyzer, cli, sentinel orchestrator)
- `src/agent_vcr/integrations/` — LangGraph, CrewAI, OpenHands (ACIDWorkspace) integrations
- Storage format: JSONL (one JSON object per line, append-only, streamable)
- ACID workspace uses git branches for isolation, git commits for savepoints, git reset --hard for rollback

## Build & Test
- Build system: hatchling
- Install dev: `pip install -e ".[dev,tui]"`
- One-command local verification: `make verify`
- Run tests: `pytest tests/unit/ -v`
- Run benchmarks: `python3 -m pytest tests/benchmarks/ -v --benchmark-only --no-cov`
- Lint: `ruff check .`
- Type check: `mypy src/agent_vcr/ src/openhands_sentinel/`
- CI: GitHub Actions (Python 3.9-3.13), benchmarks on main + PR regression check
- Benchmark dashboard: `docs/dev/bench/index.html` (reads `data.js` from github-action-benchmark)
- Dashboard generator: `scripts/generate_bench_dashboard.py` (extracts summary.json from data.js)

## Key Files
- `pyproject.toml` — Package metadata, SEO keywords, dependencies
- `docs/index.html` — Landing page (has OG tags, structured data, LangGraph FAQ)
- `README.md` — Main README with honest comparison table, LangGraph FAQ section
- `Makefile` — Local install, verify, build, benchmark, and cleanup commands
- `tests/benchmarks/test_performance.py` — Performance regression tests with hard limits

## Competitive Positioning
- NOT an observability tool — it's an intervention tool
- LangGraph has built-in checkpointer but no filesystem rollback, Ghost Replay, or Sentinel
- Comparison table honestly acknowledges competitor strengths (dashboards, evals)
- PyPI name collision: `agent-vcr` (different developer) vs `ai-agent-vcr` (this project)

## Style
- Line length: 100 (black + ruff)
- Type hints required (mypy strict)
- Imports at top of file
- Python 3.9+ compatibility (uses `from __future__ import annotations`)
