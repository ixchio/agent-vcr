# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Initial Release

### Added
- Core `VCRRecorder` and `VCRPlayer` sync functionality.
- `AsyncVCRRecorder` and `AsyncVCRPlayer` for non-blocking I/O support.
- API server (`VCRServer`) for visualizing sessions over HTTP and WebSockets.
- Terminal Textual User Interface (TUI) accessible via `vcr` or `vcr-tui` CLI commands.
- Support for LangGraph integration (`VCRLangGraph`, `vcr_record` decorator).
- High test coverage with unit, e2e, and integration tests.
- CI/CD workflow with GitHub Actions.
