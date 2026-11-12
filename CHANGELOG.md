# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.0] - 2026-11-12

### Added
- **`vcr bisect -- <test command>`**: finds the agent action that broke your tests by
  binary-searching the airbag's snapshots in temporary checkouts. No model tokens, and the
  project, its `.git` and the snapshots are never modified. Reports the culprit command, the
  files it changed, the failing output and the `vcr rewind` command to go back. `--good`,
  `--bad`, `--timeout`, `--copy-ignored`, `--json`.
- **Local Postgres rewind**: list local databases in `AGENT_VCR_AIRBAG_DATABASES` (or
  `.vcr/airbag-databases.json`). A destructive SQL command against one gets a `pg_dump` first,
  and `vcr undo` / `vcr rewind` restore the database together with the files (with a dump of the
  current state first, so the restore is undoable too). Only localhost / unix-socket databases
  are accepted. `vcr airbag db list|snapshot|restore`.
- **`vcr mcp`**: a stdio MCP server (standard library only) that gives any MCP-capable agent
  `airbag_status`, `airbag_timeline`, `airbag_check`, `airbag_snapshot`, `airbag_undo`,
  `airbag_rewind`, `recordings_list` and `recording_summary`. Undo and rewind only preview
  unless called with `dry_run: false` and `confirm: true`. Tested live in Claude Code.
- **pytest plugin** (auto-registered): the `agent_vcr` fixture records an agent run on the
  first test run and replays it with zero model calls afterwards;
  `assert_matches_golden()` fails the test on behavior, cost, token or latency regressions with
  a readable diff. `--vcr-record=none|once|all`, `--vcr-dir`, `--vcr-fail-on`,
  `@pytest.mark.agent_vcr(...)`, xdist-safe.
- **Dashboard sign-in for token-protected servers**: open `/dashboard?token=...` once, or use the
  `/dashboard/login` form, to get a short-lived HttpOnly, SameSite=Strict session cookie (Secure
  on HTTPS). Cookie-only state-changing requests and the live WebSocket must come from the
  server's own origin. The live feed uses `wss://` on HTTPS pages.
- `VCRRecorder(max_cached_frames=N)` / `AsyncVCRRecorder(max_cached_frames=N)`: bounded memory
  for long runs; evicted frames are read back from the `.vcr` file when needed.
- OpenHands SDK 1.x support for `VCRRuntime` (`Conversation(callbacks=[vcr])`, one frame per
  tool call) and Sentinel (`sentinel.bind(conversation)`; blocks the next action through the
  SDK's own rejection mechanism). The V0 event-stream API still works.
- CI runs the test suite on macOS and Windows.

### Changed
- The `openhands` extra installs `openhands-sdk` and `openhands-tools` (Python 3.12+); the old
  `openhands-ai` dependency moved to `openhands-legacy`. The unused `inngest` extra was removed.
- `create_workflow_backend("inngest")` raises `InngestBackendUnavailable` instead of returning
  a backend that never ran your steps; `"auto"` always picks the local backend.
- Airbag output: the timeline names file edits as `Edit src/app.py` instead of an absolute path,
  `vcr undo` marks files it changed back with `~` (as `--dry-run` does), and `vcr redo` says
  which action it redid.

### Fixed
- Airbag: writes and moves to a path outside the project (`cp a.txt ../x.txt`, `-t` targets,
  `mv` out of the project, `rsync` to `host:path`, through a symlink inside the project) are
  irreversible, not plain writes. Projects under symlinked directories (`/var`, `/tmp`,
  `/home` on macOS) are classified the same as on Linux.
- TUI: Ctrl+S saves in the state editor (the hint was there, the binding wasn't).
- The dashboard sidebar showed a stale version.
- `vcr bundle verify` exits 1 for a tampered or unreadable bundle (it printed `"ok": false`
  but exited 0).
- `vcr undo --dry-run` shows the same `+`/`-` signs as the real undo.
- Airbag: `ssh`/`scp` to another machine, mail senders, cloud `send`/`publish` commands, and
  schema migrations against a database server are now treated as irreversible.
- The n8n template sends `inputs` as a JSON object and includes the workflow secret.

### Documentation
- Rebuilt the docs so every topic lives on exactly one page: new hub, 5-minute quickstart, FAQ,
  Sentinel and dashboard guides; experimental SuperAgent, workflow and intelligence guides under
  `docs/experimental/`; `docs/development.md` folded into `CONTRIBUTING.md`. Every command and
  snippet was run against the code; stale or untrue claims were removed.
- The website's landing and guides pages now lead with the airbag.

## [1.6.0] - 2026-11-11

### Repository
- README rewritten around the airbag, with supported agents, an architecture overview and
  stability levels; `docs/architecture.md` now has a package map (stable vs experimental
  layers) and the airbag design; `examples/README.md` indexes every example, and
  `examples/airbag_demo.sh` reproduces the undo demo without an agent.
- Added `CODE_OF_CONDUCT.md`, issue templates and a pull-request template; `CONTRIBUTING.md`
  covers the checks, project layout and how to add an agent to the airbag.
- Test files are named after what they test.

### Added
- **Airbag** (`vcr airbag`, `vcr undo`, `vcr redo`, `vcr rewind`): snapshots the project before
  every agent tool call — shell commands included — into a shadow git store outside the
  project, and undoes any of them with one command. See `docs/airbag.md`.
  - Hook adapters and installers for Claude Code, OpenAI Codex CLI, Gemini CLI, Cursor,
    GitHub Copilot CLI, OpenHands, OpenCode (plugin API v1), Cline and Devin Local (formerly
    Windsurf), plus
    `vcr airbag run -- <agent>` (periodic snapshots) for agents without hooks and an
    in-process OpenHands SDK integration (`agent_vcr.integrations.openhands_airbag`).
  - Protects tracked and untracked files, small ignored "precious" files (`.env*`, SQLite
    databases — copied with the online-backup API for consistency), and the project's
    `.git` (hardlink copy) when a command would delete it. Never touches the user's own
    `.git`.
  - Command classifier (`vcr airbag check`): read / write / destructive / irreversible.
    Irreversible commands (remote or server databases, cloud/infra deletions, force-push,
    publishing, writes outside the project) are escalated: "ask" where the agent supports
    it, otherwise blocked until `vcr airbag allow`. The airbag never grants permission.
  - One tool call reaching the airbag through two agents' hooks (Cursor, Copilot CLI and
    Devin also run `.claude/settings.json` hooks) is handled once.

### Fixed
- `ToolGateway` tools with a parameter named `name` failed with "got multiple values for
  keyword argument 'name'" after the per-call thread change.

### Security
- **Dashboard server:**
  - `vcr-server` now refuses to bind a non-loopback host without a token. Set `AGENT_VCR_AUTH_TOKEN` / `AGENT_VCR_ADMIN_TOKEN` (new env vars), or pass `--allow-unauthenticated`. The hosted Docker image was previously open to the internet.
  - Project creation and token minting always require an admin-level token.
  - Admin and project tokens now work alongside `--auth-token`.
  - When any token is configured, every `/api` route and the WebSocket require one.
  - Token comparisons are constant-time.
  - Uploads are size-checked while they stream.
- **Bundles:** `vcr bundle import` rejects bundles whose manifest `session_id` would write outside the destination (path traversal).
- **GitHub Action:**
  - Fixed the invalid (38-character) `upload-artifact` pin. It is now `v4.6.2`.
  - Inputs are passed through `env:` to prevent script injection.
  - The `report` output is set even when the gate fails.
- **Sandbox:** `LocalSandbox` no longer passes the host environment (API keys) to commands. Opt in with `SandboxConfig(inherit_env=True)`. Its docs now say plainly that it is not a security boundary.
- **Approvals:** `AutoApprover()` is fail-closed. With no `allow_refs` and no `allow_all=True` it denies everything, and an unknown cost estimate only auto-approves explicitly allow-listed refs.

### Fixed
- **LangGraph:** `VCRLangGraph.wrap_graph` works on real `StateGraph` builders and compiled graphs. It previously crashed `compile()`.
  - Node runnables are wrapped, not replaced, so `config`/runtime injection, async nodes, retries and `Command` returns keep working.
  - `wrap_node` supports async nodes.
- **ACIDWorkspace:**
  - `abort()` restores pre-existing uncommitted edits instead of wiping them.
  - Subdirectory workspaces never touch files outside the workspace.
  - `commit()` includes changes made after the last savepoint and deletes the `acid/<id>` branch.
  - A detached HEAD is refused with a clear error.
  - Git calls have timeouts, captured output, and hooks and signing disabled.
  - Files that change mid-savepoint no longer leave orphan commits.
- **Recorder:**
  - Starting a session with an existing id no longer truncates the old recording. It is rotated to `<id>.vcr.prev`; see `start_session(if_exists=...)`.
  - Diff mode no longer drops inputs that differ from the previous output (LLM prompts were lost).
  - Values that can't be serialized (sets, bytes, dataclasses, arbitrary objects) no longer crash the agent or duplicate frames on disk.
  - Caller-supplied `FrameMetadata` is no longer mutated.
- **Claude Code hook:**
  - Records one file per Claude session (`claude-<session_id>.vcr`).
  - Reads `hook_event_name` and `tool_input.file_path` from the real payload.
  - Truncates very large fields.
  - Never fails the hook.
  - `vcr init --claude-code` writes the current interpreter path instead of a bare `python`, and upgrades previously installed entries in place.
- **Qdrant:** vector search uses `query_points`, because `search` was removed in current qdrant-client.
- **LangGraph callback:** `LangGraphCallback` is a real LangChain `BaseCallbackHandler`. Pass it with `config={"callbacks": [LangGraphCallback(rec)]}` to record one frame per graph node, plus LLM calls (with token usage) and tool calls. Passing it used to crash `invoke`. The old manual `on_*` methods still work.
- **CrewAI:**
  - `VCRCrewAI.kickoff` restores agent- and task-level callbacks, so runs no longer leak into earlier recorders. It chains to user callbacks instead of replacing them, works with pydantic `Task` objects, and adds `kickoff_async` / `kickoff_for_each`.
  - `VCRCrewCallback` listens on CrewAI's event bus (`with VCRCrewCallback(rec):` or `.attach()`) and warns if it's never attached, as in `Crew(callbacks=[...])`, which CrewAI ignores.
  - Task frames are named `task:<name>`, with the full description kept in the input.
- **Sentinel:**
  - Scanning a project under a dot-directory, such as `.claude/worktrees/...`, no longer silently passes with 0 files.
  - New `scan` exit codes: 3 when no Python files are found, and 4 when the target is missing. A missing directory is no longer created.
  - Duplicate detection uses qualified names and body hashes, so `A.run` and `B.run` are no longer flagged.
  - Growth is measured per edit, stale entries are cleared when a file is re-analyzed, and parameter counts include keyword-only arguments and `*args` but exclude `self`/`cls`.
  - `block_on_blocker` and `auto_analyze` are wired up, and OpenHands `str_replace` edits are analyzed from the observation.
- **Ghost Replay:**
  - The golden index stores file names relative to the cache dir, so cached runs and `vcr ci run` work from any working directory. Legacy indexes still load.
  - Replays keep frame types (LLM, tool and so on).
  - `savings_percent` falls back to tokens, then to replayed steps, when the golden run has no cost data, so a replay that re-ran everything reports 0% instead of 100%.
  - Golden files are written atomically.

### Changed
- `import agent_vcr` is lazy (PEP 562). Optional layers load on first use, which cuts import time from about 2.3 s to about 0.2 s, and the Claude Code hook's startup from about 1.6 s to about 0.17 s.

## [1.5.4] - 2026-10-15

### Changed
- Replaced the legacy PyPI summary with the product-focused message used by the redesigned public
  site: record runs, roll back state and files, replay verified work, and gate regressions in CI.

## [1.5.3] - 2026-10-15

### Changed
- Rebuilt GitHub Pages around a concise product homepage and developer guide hub for recording,
  ACID rollback, Ghost Replay, Agent CI, integrations, and operations; corrected stale install
  commands and qualified replay/performance guarantees.

### Fixed
- Finalized and committed all Git-owned workspace changes on the isolated ACID branch before
  checkout and merge, so work performed after the last savepoint cannot be left dirty and omitted
  from a successful `ACIDWorkspace.commit()`.
- Rejected ACID escrow and restore paths whose resolved parent escapes the workspace through a
  directory symlink, in addition to existing absolute-path and `..` validation.
- Made live WebSocket broadcasts snapshot their shared connection list under an `asyncio.Lock`,
  remove disconnected clients safely, and unregister every endpoint connection in `finally`.
- Added regression coverage for concurrent async JSONL writes, immutable SHA-256 escrow blobs over
  repeated rollback, recursively canonical Ghost Replay identities, path escapes, and dropped
  WebSocket clients.

## [1.5.2] - 2026-10-14

### Changed
- Replaced the zero-dependency vector embedder's MD5 token hashing with deterministic SHA-256 hashing so redacted memory text never enters a weak cryptographic primitive. Existing persisted vector indexes should be rebuilt after upgrading.

### Fixed
- Fixed [Issue #2](https://github.com/ixchio/agent-vcr/issues/2): ACID rollback is now content-aware for eligible pre-existing untracked and ignored files. Per-savepoint private escrow restores bytes, permissions, symlinks, deletions, renames, and type changes without writing local secrets to Git objects or VCR recordings.
- Made Git staging secret-safe by staging tracked changes and newly generated files selectively; baseline-local files are never handed to Git, so ignored credentials cannot survive as unreachable Git objects.
- Added bounded local-file handling: 10 MiB default file cap, dependency/cache exclusions, preservation of skipped paths, atomic restores, and escrow cleanup on commit or abort.
- Confined cloud diagnosis and replay paths to their validated project session directory, closing the CodeQL path-injection finding for user-controlled session IDs.

## [1.5.1] - 2026-10-14

### Changed
- Restored the MIT license across package metadata, documentation, and distribution artifacts.
- Published the GitHub repository so source, issues, and CI status are publicly accessible.
- Replaced the inaccessible private-repository CI badge with the public GitHub Actions badge.

## [1.5.0] - 2026-10-14

### Added
- Reliability intelligence (`agent_vcr.intelligence`): failure clustering with volatile-span normalization, 0–100 risk scoring (errors/denials/drift/cost/latency), deterministic failure classification, ranked recovery plans (auto only when side-effect free), evidence-based model routing, cost breakdowns with ghost-cacheable duplicate detection, and one-call `diagnose` ("why did my agent fail?").
- `record_error` accepts optional model attribution (sync + async) for routing evidence.
- `vcr intel failures|risk|why|recover|routes|costs` CLI; project-scoped `/intel/*` cloud endpoints.
- JSON Schemas for cluster, risk score, and diagnosis under `schemas/intel/`.

### Fixed
- Cloud login accepts scoped project tokens (403 on the admin listing counts as proof of validity).
- Session uploads stream in 1 MB chunks with a mid-stream 100 MB cap (413) instead of buffering unbounded bodies.
- Project/token registry mutations are inter-process locked (flock, fail-open); body exceptions always propagate.
- Schedules normalize naive/aware datetimes to UTC (mixed usage no longer crashes) and pre-parse cron expressions (impossible schedules resolve fast).
- CLI prints clean `Error:` + exit 1 for cloud/run/dataset failures instead of tracebacks.
- Session headers stay accurate across buffered saves and crash-resume attaches (atomic rewrite on count divergence).
- `Scheduler.list` renamed `all()` (builtin shadow); test suite no longer creates `.vcr` in the repo checkout.

## [1.4.0] - 2026-09-09

### Added
- Workflow engine (`agent_vcr.workflows`): durable DAG runs on the VCR kernel — conditions, parallel fan-out/join, `for_each` loops with caps, retries with backoff, timeouts, approvals, checkpoints, saga compensation.
- Durable state: `RunStore` pins the full definition snapshot per run and persists after every top-level step; `step_once`/`resume`/`attach_session` recover crashes without re-executing completed work.
- Compensation runs in reverse completion order, survives partial failures, and never re-runs after an abort-crash (error-persisted resume path).
- `WorkflowRegistry` with content-based version bumps; `Scheduler` with interval/cron(minimal real parser)/one-shot `at` schedules, `tick` + `run_pending`.
- Server workflow API: registry, webhook triggers with secrets (`409` without handlers), run inspection, resume; `vcr-server --handlers`.
- `vcr workflow validate/run/list/runs/show/resume/schedule put|list|remove|tick` CLI.
- Fixed a real durability bug found along the way: JSONL session headers went stale on append/attach — `save()` now rewrites header + frames atomically when counts diverge.
- JSON Schemas for workflow definition and run record under `schemas/workflows/`.

## [1.3.0] - 2026-09-09

### Added
- SuperAgent runtime (`agent_vcr.superagent`): policy-gated DAG execution on the VCR kernel — policy → budget → approval → execute → verify → checkpoint, every transition recorded.
- Executable plans (`Plan`/`PlanStep` with deps, `$ref` input wiring, retries, `abort|retry|skip` failure modes); `PlanBuilder` validation (cycles, unknown refs, bound checks, topological order).
- `LLMPlanner`: LiteLLM-backed plan generation with validation-feedback repair retries; strict JSON parsing unit-tested without credentials.
- `ToolGateway`: registry, strict parameter validation, real per-call timeouts, decorator + ToolSpec registration.
- `MemoryStore`: namespaced working memory with secret redaction (persisted views never touch credentials), search, snapshot, save/load.
- `Verifier` (equals/contains/no_error/required/forbid keys), `Policy` (forbid/approval/max-steps/irreversible gates), `BudgetTracker` (token/cost/latency caps), `AutoApprover` + `CallbackApprover` (fail-closed).
- `SuperAgent.run` with retry, skip-cascade, nested sub-agents via `as_handler`, `RunReport`; `vcr superagent run --plan --handlers` CLI with real stdin approvals.
- JSON Schemas for plan and run report under `schemas/superagent/`.

## [1.2.0] - 2026-09-09

### Added
- VCR Cloud: project registry with slug-validated namespaces under `<root>/projects/<id>/sessions/`.
- Scoped API tokens (`owner` read+write / `member` read-only), sha256-hashed at rest, constant-time verification, mint/list/revoke endpoints; legacy global token and admin token honored everywhere.
- Execution ingestion gate: multipart upload parses + `verify_integrity` before visibility; dedup (200), conflict (409 + `?overwrite`), corrupt (422), 100 MB cap.
- Pull (`download` byte-identical), paginated/searchable listing (header-only fast path), owner-gated delete, server-side Ghost FULL replay endpoint.
- `CloudClient` (stdlib urllib, timeouts, `CloudError` statuses) + `vcr cloud login/status/projects/sessions/push/pull/replay` with `0600` credential profiles and `--token-stdin`.
- `GET /api/health` liveness probe; `vcr-server --data-dir/--admin-token`.
- JSON Schemas for project and token under `schemas/cloud/`.

## [1.1.0] - 2026-09-09

### Added
- Agent CI: versioned replay datasets (`ReplayDataset`, `TestCase`, `Expectation` test annotations) stored under `.vcr/datasets/`.
- Regression engine (`run_regression`, `compare_sessions`, `evaluate_candidate`) with PASS/DEGRADED/FAIL verdicts, cost/latency/token deltas, LOW/MEDIUM/HIGH risk, and PR-style markdown reports.
- `vcr dataset create/add/list/show/remove` and `vcr ci run` CLI with `--fail-on` gates, JSON/markdown reports, `$GITHUB_STEP_SUMMARY` and `::error::`/`::warning::` annotations.
- Composite GitHub Action (`action.yml`) plus `examples/agent-ci.yml` workflow and `examples/agent_ci_candidates.py` demo helper.
- Dogfood `agent-ci` CI job proving the gate passes clean runs and blocks regressions.
- JSON Schemas for dataset and regression report under `schemas/ci/`.

## [1.0.0] - 2026-09-08

### Added
- Frozen `execution/v1` schema with `schema_version`, forward-compatible readers, and exported JSON Schemas under `schemas/`.
- Canonical `StateDiff` helpers shared by recorder (JSON-patch) and player (added/removed/modified).
- Explicit checkpoint API: `record_checkpoint` / `list_checkpoints` / `goto_checkpoint` (sync + async).
- Player durability: `load(strict=...)`, `skipped_lines`, `verify_integrity`, `rebuild_manifest`, fsync'd session headers.
- Replay Bundle (`.vcrb`): portable `execution.vcr` + identity manifest with sha256 verification; `vcr bundle export/import/verify`, `vcr reproduce`.
- Ghost Replay modes: `FULL` / `HYBRID` / `LIVE` / `BRANCH` via `replay_with_mode`.
- `ACIDWorkspace` hardening: `abort()`, `savepoints()`, `status()`, context-manager support.

## [0.8.0] - 2026-07-15

### Added
- Safe ACID rollback now preserves pre-existing untracked and ignored files while deleting only files introduced during the transaction.
- `ACIDWorkspace` rejects dirty tracked worktrees by default, with `dirty_worktree_policy="allow"` available for advanced callers.
- Ghost Replay identities can now include model, prompt hash, code commit, tool schema hash, dependency lock hash, environment, and custom fields.
- Ghost Replay ledgers now expose cache-hit reasons and per-step replay/rerun sources.
- `vcr-server` supports optional token auth for API and WebSocket access.
- `sentinel watch` provides polling-based live scans without adding a runtime watcher dependency.
- `Makefile` provides repeatable install, verify, build, benchmark, and cleanup commands.

### Changed
- `vcr-server` binds to `127.0.0.1` by default instead of `0.0.0.0`.
- Partial Ghost Replay is now conservative by default: if a step changes, downstream steps are re-executed unless `allow_partial_replay=True`.
- Dashboard/server dependencies moved out of the core install and into the `dashboard` extra.
- CI now type-checks and measures coverage for `openhands_sentinel`.
- Build metadata now explicitly excludes local caches, runtime recordings, and build artifacts from release outputs.

## [0.7.1] - 2026-07-03

### Added
- `vcr init --claude-code` scaffolds Claude Code lifecycle hooks for Agent VCR recording.

### Fixed
- `fork()` now materializes fork state as a real checkpoint frame.
- `diff_mode` resume reconstructs omitted input state from prior output state.
- ACID rollback removes ignored generated files while preserving VCR audit files.
- Async recorder helper methods now match the sync recorder more closely.
- README/API drift around CLI, cache listing, and recorder helpers.

## [0.5.0] - 2026-03-26

### Added
- **ACID Transactions for Agents** — Wrap any agent execution in full transactional semantics. `BEGIN` snapshots your workspace, `SAVEPOINT` checkpoints filesystem + memory together, `ROLLBACK` physically reverts files on disk via git (not just in-memory state), and `COMMIT` locks in the successful path. Two agents working in parallel get isolated git branches so they can't clobber each other's work.
- **Golden Run Cache** — The feature nobody else has. When your agent succeeds, save that entire execution as a "golden path." Next time you run the same task, it replays the golden path and skips every LLM call. Same task. Zero tokens. Instant. The `CostLedger` tracks exactly how much you saved — tokens, dollars, and milliseconds.
- **`ACIDWorkspace`** — New integration class in `agent_vcr.integrations.openhands` providing `begin()`, `savepoint()`, `rollback()`, and `commit()` methods backed by git branch isolation.
- **`GoldenRunCache`** — New class in `agent_vcr.golden_cache` with `save_golden_run()`, `replay()`, `invalidate()`, and `list_golden_runs()`. Task fingerprinting is case-insensitive and deterministic.
- **`CostLedger`** — Tracks original vs replay costs and produces a clean summary dict with savings percentages.
- **19 new production tests** covering the full ACID lifecycle and golden cache replay logic. 153 tests total, 81% coverage.



### Added
- **Legit React Dashboard** — We built a stunning, glassmorphism-themed React dashboard right into the package. Fire up `vcr-server`, pop open `localhost:8000`, and you get a beautiful UI to visualize live frame streaming, track token usage/latency, and inspect state changes with dedicated JSON diffing tabs. 100% local. No cloud service.
- **Search & Filter Support** — Added `GET /api/tags` and search queries to easily track down specific sessions in the dashboard.
- **`[dashboard]` Extra** — We kept the core lightweight. Run `pip install "ai-agent-vcr[dashboard]"` if you want the FastAPI server and UI bundled in.

## [0.3.1] - 2026-03-09

### Fixed
- **`record_step()` latency override bug** — User-provided `latency_ms` was being overwritten by internal serialization time. Now only auto-sets latency when the caller did not supply one.
- **`record_tool_call()` error_type** — Was always `"str"` instead of `"ToolError"` since the error parameter is a string.
- **`get_frames()` empty after flush** — Frames were lost after `save()` or auto-flush. Now delegates to the cache for persistence.
- **Double `frame_count` increment** — `VCRCache.add_frame()` and `record_step()` both incremented `session.frame_count`, causing double-counting.
- **File descriptor leak in `AsyncVCRRecorder`** — The `mkstemp` fd was not closed before `aiofiles` reopened the file by path.
- **Missing `on_frame_recorded` callback in `AsyncVCRRecorder`** — The async recorder now supports both sync and async callbacks for live WebSocket streaming.
- **Missing cache integration in `AsyncVCRRecorder`** — `cache.add_frame()` was never called, causing `get_frames()` to fail after flush.
- **Inconsistent node naming** — Async recorder used `llm_model` / `tool_name` while sync used `llm:model` / `tool:name`. Standardized to colon separator.
- **Hardcoded version `0.1.0` in server** — Now dynamically reads from `__version__`.
- **Blocking file I/O in async server endpoints** — All `VCRPlayer.load()` calls in async handlers wrapped with `asyncio.to_thread`.
- **`format` parameter shadowed built-in** — Renamed to `export_format` in the export endpoint.
- **CrewAI callback latency** — Callbacks were measuring VCR serialization overhead, not actual agent execution time. Removed misleading measurement.

## [0.3.0] - 2026-03-07

### Added
- **Real-time live streaming** via WebSocket push from recorder.
- **DAG visualization** for parallel execution branches.
- **Interactive TUI debugger** — navigate execution with keyboard, edit state, resume.
- **`AsyncVCRRecorder`** — native async support for non-blocking recording.
- **`AsyncVCRPlayer`** — async playback for non-blocking replay.
- **`diff_mode`** recording to reduce VCR file sizes.
- **`StateSerializer`** for preserving type information during serialization.
- **`VCRFileWatcher`** for real-time dashboard updates.
- **Performance benchmarks** in CI with regression detection.
- **CrewAI integration** with automatic callback-based recording.

## [0.2.0] - 2026-03-07

### Added
- FastAPI server (`VCRServer`) for visualizing sessions via HTTP and WebSockets.
- Codecov integration for coverage reporting.
- Extended API with session comparison and export endpoints.


## [0.1.0] - Initial Release

### Added
- Core `VCRRecorder` and `VCRPlayer` sync functionality.
- `AsyncVCRRecorder` and `AsyncVCRPlayer` for non-blocking I/O support.
- API server (`VCRServer`) for visualizing sessions over HTTP and WebSockets.
- Terminal Textual User Interface (TUI) accessible via `vcr` or `vcr-tui` CLI commands.
- Support for LangGraph integration (`VCRLangGraph`, `vcr_record` decorator).
- High test coverage with unit, e2e, and integration tests.
- CI/CD workflow with GitHub Actions.
