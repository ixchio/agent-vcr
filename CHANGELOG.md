# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
