# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
