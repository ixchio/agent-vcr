# Contributing to Agent VCR

Thanks for helping! Bug reports, docs fixes, new agent integrations and code are all welcome.
Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first; report security issues privately
as described in [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tui,dashboard]"
pre-commit install            # optional: ruff + mypy on every commit
```

Python 3.9–3.13 is supported; `git` must be on `PATH` (the airbag and ACID workspace use it).

## Checks

| Command | What it runs |
|---|---|
| `make verify` | Everything CI runs before a release: lint, types, unit + e2e tests, benchmarks, build |
| `pytest tests/unit -q --no-cov` | Fast unit suite (about 30 s) |
| `ruff check .` / `ruff format .` | Lint / format (line length 100) |
| `mypy src/agent_vcr/ src/openhands_sentinel/` | Type check (`disallow_untyped_defs`) |
| `python -m pytest tests/benchmarks/ --benchmark-only --no-cov` | Performance limits |

Some tests exercise real third-party packages and skip themselves when those are not installed:
`test_langgraph_real.py` (`pip install langgraph`), `test_crewai_real.py` (`pip install crewai`),
`test_openhands_airbag.py` (`pip install openhands-sdk openhands-tools`). Run them when you change
the matching integration.

## Project layout

```text
src/agent_vcr/          the package (see docs/architecture.md for the layer map)
  airbag/               snapshot-before-every-tool-call + vcr undo/redo/rewind
  integrations/         LangGraph, CrewAI, OpenHands (ACID workspace, airbag)
  hooks/                Claude Code recording hook
src/openhands_sentinel/ Sentinel AST code-quality guard
tests/unit/             unit tests, one file per module or behavior
tests/e2e/              end-to-end scenarios
tests/benchmarks/       performance regression limits
docs/                   guides (markdown) and the GitHub Pages site (html)
schemas/                exported JSON Schemas of the data contracts
examples/               runnable, credential-free examples
deploy/                 optional hosted-deployment configuration
```

## Coding standards

* Type hints everywhere, `from __future__ import annotations`, Python 3.9-compatible syntax.
* Recorders and hooks must **never break the agent they observe**: catch, log, continue.
* File writes are atomic (temp file + `os.replace`); I/O and subprocesses have timeouts.
* The async rules in [AGENTS.md](AGENTS.md) (locks around shared state, queues, graceful
  shutdown) apply to all async code.
* Anything that restores files must not delete or overwrite data it did not snapshot.

## Adding an agent to the airbag

Each agent is an adapter in [`src/agent_vcr/airbag/agents.py`](src/agent_vcr/airbag/agents.py):

1. **Parse** — make sure `parse_payload()` understands the agent's pre-tool payload (tool name,
   shell command, edited paths, working directory). Add the tool names to `SHELL_TOOLS` /
   `EDIT_TOOLS` if needed.
2. **Render** — write `render_<agent>(decision, call, options)` that answers in the agent's
   format. Never output a decision that *grants* permission; when the agent has no "ask",
   escalate irreversible commands with a block/deny that includes a reason.
3. **Install** — add an installer that merges into the agent's config without touching the
   user's own hooks, and register an `Adapter` in `ADAPTERS`.
4. **Test** — add a case to `tests/unit/test_airbag_agents.py` that pipes the agent's documented
   payload through the real hook process, plus an install / re-install / uninstall check.
5. **Document** — add a row to the table in [docs/airbag.md](docs/airbag.md) with a link to the
   agent's official hook documentation, and say what was verified (docs, installed source,
   live session).

## Pull requests

1. Branch from `main`; keep each PR focused on one change.
2. Add a test for new behavior and a regression test for every bug fix.
3. Update the docs and the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md).
4. Commit messages follow the existing style: `fix: …`, `feat: …`, `docs: …`, `chore: …`.
5. Open the PR with the template filled in and make sure CI is green.

## Cleanup

Generated caches, build output and local recordings are git-ignored. `make clean` removes
Python/build artifacts; `make clean-runtime` removes local `.vcr` sessions.
