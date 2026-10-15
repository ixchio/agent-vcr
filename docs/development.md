# Agent VCR: Contributor & Developer Engineering Guide

> **Package:** `ai-agent-vcr`
> **Repository:** `ixchio/agent-vcr`
> **Build Backend:** `hatchling`
> **Target Python Versions:** 3.9, 3.10, 3.11, 3.12, 3.13

---

## 1. Local Environment Setup & Dependency Matrix

Agent VCR adopts a modular packaging architecture. The core durability kernel depends strictly on `pydantic` and `aiofiles` (with `eval-type-backport` for Python 3.9 compatibility). All heavy integrations (microVM sandboxes, vector stores, distributed workflow backends, and TUI debuggers) are partitioned into explicit dependency extras.

### 1.1 Python Toolchain & Virtual Environment

Ensure you have Python 3.9 or newer installed (Python 3.11+ recommended for local development).

```bash
# Clone the repository
git clone https://github.com/ixchio/agent-vcr.git
cd agent-vcr

# Create isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging primitives
pip install --upgrade pip setuptools wheel build
```

### 1.2 Dependency Extras Breakdown

| Extra | Dependencies Included | Purpose |
|---|---|---|
| `dev` | `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-benchmark`, `httpx`, `black`, `ruff`, `mypy`, `build`, `twine`, `fastapi`, `uvicorn`, `python-multipart` | Test runner, coverage, linters, type checkers, and test server dependencies. |
| `tui` | `rich>=13.0.0`, `textual>=0.40.0` | Terminal user interface for time-travel session inspection (`vcr inspect`). |
| `dashboard` | `fastapi`, `uvicorn`, `websockets`, `watchdog`, `python-multipart` | Local WebSocket/HTTP streaming server (`vcr-server`) and web viewer. |
| `sandbox` | `e2b>=2.46.0` | Firecracker microVM isolation adapter (`E2BSandbox`). |
| `vector` | `qdrant-client>=1.19.0` | Semantic long-term agent memory (`QdrantVectorStore`). |
| `llm` | `litellm>=1.100.0` | Multi-provider LLM gateway for the SuperAgent planner. |
| `inngest` | `inngest>=0.5.0` | Distributed event-driven workflow backend with step memoization. |
| `cloud` | `e2b`, `qdrant-client`, `litellm`, `inngest` | Meta-package bundling all cloud runtime dependencies. |
| `all` | All framework integrations (`langgraph`, `crewai`, `openhands`) + cloud extras. | Full kitchen-sink development installation. |

### 1.3 Recommended Developer Installation

For active local development across the full stack (including TUI, cloud server, workflows, and intelligence):

```bash
pip install -e ".[dev,tui,dashboard,sandbox,vector,llm,inngest]"
```

Or run the Makefile target:

```bash
make install
```

---

## 2. Codebase Architecture & Directory Map

```
agent-vcr/
├── pyproject.toml              # Project metadata, packaging config, ruff/mypy/pytest rules
├── Makefile                    # Standard developer automation targets
├── action.yml                  # Agent CI composite GitHub Action
├── schemas/                    # Pydantic-exported JSON Schemas (execution/v1, workflows, etc.)
│   ├── execution/              # session.json, frame.json
│   ├── workflows/              # workflow_def.json, run_record.json
│   ├── superagent/             # plan.json, run_report.json
│   ├── ci/                     # dataset.json, regression_report.json
│   ├── cloud/                  # project.json, api_token.json
│   └── intel/                  # failure_cluster.json, risk_score.json, diagnosis.json
├── src/
│   ├── agent_vcr/              # Core Agent VCR package
│   │   ├── models.py           # execution/v1 schema, StateDiff, Frame, Session
│   │   ├── recorder.py         # Sync append-only JSONL recorder, atomic flushes
│   │   ├── async_recorder.py   # Async non-blocking recorder (aiofiles)
│   │   ├── player.py           # Random-access player, compare_frames, verify_integrity
│   │   ├── async_player.py     # Async playback navigation
│   │   ├── golden_cache.py     # Ghost Replay engine, CostLedger, task fingerprinting
│   │   ├── bundle.py           # Portable Replay Bundle (.vcrb) export/import/reproduce
│   │   ├── datasets.py         # Replay datasets for regression testing (.vcr/datasets/)
│   │   ├── regression.py       # 9-dimension regression engine (PASS/DEGRADED/FAIL)
│   │   ├── cloud_store.py      # Project registry, HMAC scoped tokens, streaming upload gate
│   │   ├── cloud.py            # HTTP CloudClient for push/pull/status/replay
│   │   ├── server.py           # FastAPI server (HTTP API, WebSockets, static dashboard)
│   │   ├── cli.py              # CLI router (vcr run/inspect/ci/workflow/cloud/intel)
│   │   ├── tui.py              # Textual terminal time-travel debugger
│   │   ├── superagent/         # SuperAgent execution runtime
│   │   │   ├── agent.py        # SuperAgent execution pipeline
│   │   │   ├── planner.py      # Plan DAG builder, LiteLLM repair loop
│   │   │   ├── router.py       # ToolGateway schema validation & timeout enforcement
│   │   │   ├── policy.py       # Static Policy rules & BudgetTracker ledger
│   │   │   ├── verifier.py     # Declarative output assertions
│   │   │   ├── approval.py     # Approver protocol (AutoApprover, CallbackApprover)
│   │   │   ├── memory.py       # Working MemoryStore with deep secret redaction
│   │   │   └── vector_memory.py# Qdrant long-term vector memory with fallback
│   │   ├── workflows/          # Durable Workflow Engine
│   │   │   ├── engine.py       # WorkflowEngine executor, crash resume, saga rollback
│   │   │   ├── models.py       # WorkflowDef, WorkflowStep, RunRecord, CompensateSpec
│   │   │   ├── store.py        # WorkflowRegistry & atomic RunStore persistence
│   │   │   ├── backend.py      # LocalWorkflowBackend & InngestWorkflowBackend
│   │   │   └── schedules.py    # Zero-dependency 5-field cron/interval scheduler
│   │   ├── sandbox/            # Sandbox isolation runtime
│   │   │   ├── base.py         # BaseSandbox abstract contract
│   │   │   ├── local.py        # LocalSandbox (traversal guard, process-group kill)
│   │   │   └── e2b_adapter.py  # E2BSandbox (Firecracker microVM adapter)
│   │   ├── intelligence/       # Reliability intelligence tier
│   │   │   ├── failures.py     # Volatile-span normalization & error clustering
│   │   │   ├── risk.py         # Deterministic 0-100 risk scoring engine
│   │   │   ├── query.py        # diagnose() unified root-cause engine
│   │   │   ├── recovery.py     # Autonomous recovery planner & safety classifier
│   │   │   └── routing.py      # Empirical telemetry model routing
│   │   └── integrations/       # Framework bindings (LangGraph, CrewAI, OpenHands)
│   └── openhands_sentinel/     # Sentinel AST code quality guardian
│       ├── analyzer.py         # AST parser: cyclomatic complexity, length explosion
│       ├── sentinel.py         # Trajectory monitor & auto-attachment hooks
│       └── cli.py              # sentinel scan / watch commands
└── tests/
    ├── unit/                   # Unit tests organized by product capability
    ├── e2e/                    # End-to-end multi-step agent scenarios
    └── benchmarks/             # pytest-benchmark performance tests with hard budgets
```

---

## 3. Testing Strategy & Test Execution

Every architectural tier is covered by deterministic, hermetic tests that require zero external cloud credentials or network access.

### 3.1 Test Suite Organization

```
tests/unit/
├── test_schema.py                # Schema stability, StateDiff, JSONL resilience, header sync
├── test_bundle.py                # Replay bundles (.vcrb), manifest SHA-256, Ghost Replay modes
├── test_ci_regression.py         # Dataset management, regression engine, report generation
├── test_cloud.py                 # Projects, scoped HMAC tokens, chunked ingest, cloud client
├── test_superagent.py            # Plan DAG builder, LiteLLM repair, policy/budget, memory redaction
├── test_workflows.py             # WorkflowEngine, crash resume, LIFO saga compensation, cron
├── test_intelligence.py          # Error clustering, 0-100 risk scoring, diagnosis, model routing
├── test_cloud_store.py           # Sandboxes (Local/E2B), Qdrant vector memory, Inngest backend
├── test_acid_golden.py          # ACIDWorkspace git transactions & GoldenRunCache replay
├── test_sentinel.py             # Sentinel AST trajectory analyzer & complexity guards
├── test_server.py               # FastAPI ASGI endpoints & WebSocket live stream
├── test_async.py                # Async recorder and player non-blocking I/O
├── test_langgraph_integration.py# LangGraph node wrapping and checkpointer bindings
├── test_crewai_integration.py   # CrewAI callback interception and execution recording
├── test_edge_cases.py           # Corrupt JSONL handling, path traversal attacks, bounds
├── test_cli.py                  # CLI argument parsing and error formatting
├── test_tui.py                  # Textual timeline navigation
├── test_player.py               # Synchronous playback & random-access seeking
├── test_recorder.py             # Synchronous recording & auto-flush
└── test_models.py               # Model serializers and defaults
```

### 3.2 Running Unit Tests

To run the full unit test suite with coverage reporting:

```bash
# Using pytest directly
pytest tests/unit/ -v

# Using Makefile target
make test-unit
```

To run a specific capability or module:

```bash
# Run durable workflow engine tests
pytest tests/unit/test_workflows.py -v

# Run SuperAgent runtime tests
pytest tests/unit/test_superagent.py -v

# Run reliability intelligence tests
pytest tests/unit/test_intelligence.py -v
```

### 3.3 End-to-End Tests

```bash
pytest tests/e2e/ -v --no-cov
```

---

## 4. Performance Benchmarking & Local Regression Budgets

Agent VCR enforces hard local regression limits on latency, memory, and disk usage. They keep
changes comparable on CI runners; they are not universal production latency or throughput
guarantees for every filesystem, machine, state size, or workload.

### 4.1 Running Benchmarks

```bash
# Run benchmarks locally
python3 -m pytest tests/benchmarks/ -v --benchmark-only --no-cov

# Or via Makefile
make bench
```

### 4.2 Hard Regression Budgets (`tests/benchmarks/test_performance.py`)

| Benchmark Target | Metric | Hard Limit | Architecture Rationale |
|---|---|---|---|
| `test_record_frame_overhead` | Single frame serialization + write | **$< 5.0\text{ ms}$** | Agent execution loop must not be throttled by recording I/O. |
| `test_record_throughput_1000_frames` | Ingestion of 1,000 frames | **$< 2,000\text{ ms}$** | Sustains high-throughput parallel sub-agent swarms. |
| `test_player_load_1000_frames` | Parsing & loading a 1,000-frame `.vcr` | **$< 500\text{ ms}$** | Fast cold-start time for CLI, TUI, and CI test harnesses. |
| `test_player_goto_frame_random_access` | Random-access seek to arbitrary frame | **$< 1.0\text{ ms}$** | Instant time-travel seeking in the interactive debugger. |
| `test_player_compare_frames_diff` | State comparison between frames | **$< 2.0\text{ ms}$** | Rapid UI diff rendering and regression evaluations. |
| `test_golden_cache_save_latency` | Task fingerprinting & golden cache write | **$< 50.0\text{ ms}$** | Low overhead when promoting runs to golden fixtures. |
| `test_golden_cache_lookup_hit` | Fingerprint cache hit & mock injection | **$< 10.0\text{ ms}$** | Detects local replay lookup regressions. |
| `test_file_size_growth_linear` | Disk footprint per 1,000 frames | **$< 250\text{ KB}$** | Ensures million-frame archives remain storage-efficient. |
| `test_memory_usage_bounded` | Memory growth over 10,000 frames | **$< 50\text{ MB}$ delta** | Prevents memory leaks in continuous daemon recorders. |

---

## 5. Static Analysis, Quality Gates & Verification

Before submitting pull requests, run the static analysis suite to ensure compliance with project standards.

### 5.1 Linting with Ruff

Agent VCR adheres to strict Ruff rules configured in `pyproject.toml` (target line length: 100):

```bash
# Check code style and rules
ruff check .

# Automatically apply safe fixes
ruff check . --fix
```

### 5.2 Strict Type Checking with Mypy

All core modules (`src/agent_vcr/` and `src/openhands_sentinel/`) enforce type annotations:

```bash
mypy src/agent_vcr/ src/openhands_sentinel/
```

### 5.3 AST Sentinel Code Inspection

Run the built-in Sentinel guardian over local code:

```bash
sentinel scan src/
```

### 5.4 One-Command Local Verification

Run all checks, unit tests, e2e tests, benchmarks, and package distribution checks with a single command:

```bash
make verify
```

---

## 6. Replay Bundles & Deterministic Reproduction

The Replay Bundle (`.vcrb`) is the standard portable reproduction artifact (Phase 1, design doc §72). It bundles the execution recording, git commit hash, model identifiers, tool schemas, and environment fingerprints into an audited, SHA-256 verified package.

### 6.1 Exporting a Bundle

```bash
vcr bundle export <session_id> --output issue_404.vcrb
```

Inside `issue_404.vcrb` (gzipped tar archive):
```
├── execution.vcr    # Complete JSONL session frames (Source of Truth)
└── bundle.json      # BundleManifest: SHA-256 hash, git SHA, model/tool metadata
```

### 6.2 Verifying Bundle Integrity

Verifies that the `execution.vcr` contents match the embedded SHA-256 checksum without extracting:

```bash
vcr bundle verify issue_404.vcrb
```

### 6.3 One-Liner Reproduction

Reproduce and inspect the exact execution on another machine:

```bash
vcr reproduce issue_404.vcrb
```

---

## 7. Agent CI & GitHub Actions Regression Gate

Agent CI functions as the "GitHub Actions for AI agent reliability." It compares new agent runs against golden replay datasets across 9 dimensions:

1. **Output Correctness:** State output matches expected golden outputs.
2. **Tool Correctness:** Tools called match expected tool signatures.
3. **State Transitions:** State variables follow allowed mutations.
4. **Cost Delta:** Monetary spend ($) does not exceed baseline threshold.
5. **Latency Delta:** Execution duration (ms) does not degrade beyond budget.
6. **Token Usage:** Token consumption stays within allowed variance.
7. **Failure Rate:** Error-type frames do not increase.
8. **Policy Violations:** No forbidden tools or unapproved actions attempted.
9. **Checkpoint Integrity:** All critical checkpoints are reached.

### 7.1 Managing Datasets Locally

```bash
# Create a new test dataset
vcr dataset create payment_v1

# Add a verified golden session
vcr dataset add payment_v1 .vcr/golden_run.vcr --case happy_path

# Run regression check against a candidate run
vcr ci run payment_v1 --candidate .vcr/new_run.vcr --fail-on degraded
```

### 7.2 GitHub Action Workflow Configuration

Integrate Agent CI into PR checks via `.github/workflows/agent-ci.yml`:

```yaml
name: Agent CI Regression Gate

on:
  pull_request:
    branches: [main]

jobs:
  agent-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev,all]"

      - name: Run Agent CI Regression Gate
        uses: ./
        with:
          dataset: payment_v1
          candidate: .vcr/latest_run.vcr
          fail-on: degraded
```

---

## 8. Build Engineering & Release Pipeline

Agent VCR uses `hatchling` as its build backend. Builds exclude runtime caches, test recordings, and local development artifacts.

### 8.1 Building Artifacts

```bash
# Build source distribution (.tar.gz) and binary wheel (.whl)
python3 -m build --sdist --wheel
```

Artifacts are generated in `dist/`:
- `dist/ai_agent_vcr-1.5.4.tar.gz`
- `dist/ai_agent_vcr-1.5.4-py3-none-any.whl`

### 8.2 Distribution Integrity Verification

Verify package descriptions, license metadata, and markup compatibility before release:

```bash
python3 -m twine check dist/*
```

### 8.3 Workspace Cleanup

```bash
# Clean build, test, and cache artifacts
make clean

# Clean runtime .vcr recording directories
make clean-runtime
```
