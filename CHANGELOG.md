# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
