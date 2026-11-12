# AGENTS.md — Repository Knowledge

## Project Overview
- **Name:** Agent VCR (`ai-agent-vcr` on PyPI)
- **Purpose:** an undo button for AI agents — snapshot before every agent tool call (airbag),
  time-travel recording, ACID filesystem + state rollback, zero-cost Ghost Replay, Agent CI
- **Key differentiators:** undo that covers shell commands for 9 coding agents (airbag),
  filesystem rollback via git (not just state), Ghost Replay, Sentinel AST guard
- **Frameworks / agents:** Claude Code, Codex CLI, Gemini CLI, Cursor, Copilot CLI, OpenHands,
  OpenCode, Cline, Devin Local; LangGraph, CrewAI, raw Python
- **Version:** 1.7.0

## Architecture (full map: docs/architecture.md)
- `src/agent_vcr/` — recording kernel: models, recorder, async_recorder, player, async_player
- `src/agent_vcr/airbag/` — store (shadow git outside the project), classify, core (decisions,
  undo/redo/rewind), agents (per-agent parse/render/install), hook (entry point), cli, runner,
  bisect (`vcr bisect`), databases (opt-in local Postgres dumps)
- `src/agent_vcr/mcp_server.py` — `vcr mcp` stdio MCP server (stdlib only)
- `src/agent_vcr/pytest_plugin.py` — pytest11 plugin (`agent_vcr` fixture); keep it import-light
- `src/agent_vcr/integrations/` — LangGraph, CrewAI, OpenHands (ACIDWorkspace, openhands_airbag)
- `src/agent_vcr/` replay & CI — golden_cache, datasets, regression, bundle; `action.yml`
- `src/agent_vcr/` server, cloud, cloud_store, hosted, tui — dashboard and self-hosted store
- Experimental: `superagent/`, `workflows/`, `sandbox/`, `intelligence/`
- `src/openhands_sentinel/` — Sentinel code quality guardian
- Storage format: JSONL (one JSON object per line, append-only, streamable)
- `agent_vcr/__init__.py` is lazy (PEP 562) — keep hook/CLI import paths free of heavy deps
- Airbag invariants: never grant a permission; never delete/overwrite data that was not
  snapshotted; the user's own `.git` is never written except restoring a missing/broken one

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
- `pyproject.toml` — package metadata, keywords, dependencies
- `README.md` — main README; `docs/README.md` — documentation index
- `docs/airbag.md` — airbag guide incl. per-agent hook contracts and limits
- `docs/architecture.md` — package map, stability levels, airbag design
- `docs/index.html` — GitHub Pages landing page
- `Makefile` — install, verify, build, benchmark, cleanup
- `tests/benchmarks/test_performance.py` — performance regression tests with hard limits
- `tests/unit/test_airbag_core.py`, `test_airbag_agents.py` — airbag engine + per-agent hooks

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

## Sync & Concurrency Standards
These patterns are **required** for all async code. No exceptions.

- **Shared mutable state → `asyncio.Lock`**: Never access shared dicts/lists from coroutines without a lock. Wrap reads AND writes.
- **Signaling between coroutines → `asyncio.Event`**: Use `event.set()` / `await event.wait()` instead of polling with `asyncio.sleep`.
- **Producer/consumer → `asyncio.Queue`**: Never use raw `list.append` for cross-coroutine data passing. Use `Queue(maxsize=N)` for backpressure. Always `await queue.join()` before shutdown to drain — zero lost frames.
- **File writes → atomic (`tmp` + `os.replace`)**: Write to a `.tmp` sibling first, then `os.replace(tmp, dest)` (POSIX atomic). On failure, unlink `.tmp`. Never write directly to the final path mid-stream.
- **All I/O → `asyncio.wait_for(coro, timeout=N)`**: Every network call, file op, and subprocess must have a timeout. No hung processes.
- **Idempotent operations**: Check-then-skip before recording/writing. If the frame/key already exists, return early — never error on duplicate. Safe retries = reliability.
- **Graceful shutdown order**: (1) stop accepting new work, (2) `await queue.join()`, (3) `task.cancel()` + `asyncio.gather(*tasks, return_exceptions=True)`.

### Quick Reference
| Rule | Why |
|---|---|
| Lock around every shared dict/set/list | Race conditions corrupt recordings |
| `asyncio.Queue` not `list.append` | Thread/coroutine-safe, backpressure built-in |
| Write `.tmp` → `os.replace()` | Crash-safe, atomic on POSIX |
| `await queue.join()` before exit | No lost frames |
| `asyncio.wait_for()` on every I/O | No hung processes |
| Idempotent writes (check-then-skip) | Safe retries = bulletproof replay |
