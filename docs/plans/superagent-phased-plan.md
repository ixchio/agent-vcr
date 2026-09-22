# Agent VCR + SuperAgent — Phased Build Plan

> Status: historical planning artifact. Consult the current README and architecture docs for
> implemented behavior.
>
> Source: [`startup-system-design.md`](startup-system-design.md) (3112 lines, 90 sections)
> Codebase: `ai-agent-vcr 0.8.0` (`src/agent_vcr/`, `src/openhands_sentinel/`)
> Research date: 2026-09-08 — all versions verified live via PyPI JSON API + npm registry
> Verified with: `python3 urllib pypi.org/pypi/{pkg}/json` + `npm view {next,react,inngest}`

---

## 1. Executive Summary

Thesis: **Build the reliability, execution, recovery, and deployment layer for autonomous AI agents.**

Positioning: **Time-travel infrastructure for autonomous AI.**

Core loop:

```
Record → Checkpoint → Replay → Evaluate → Fork → Recover → Deploy
```

Product promise (design doc §2):

| Git | Agent VCR |
|---|---|
| commit | checkpoint |
| branch | fork / branch execution |
| diff | state diff |
| revert | rollback (state + filesystem) |
| test fixture | Ghost Replay |
| CI | Agent CI |
| deploy | hosted deployment + recovery |

> Observability tells you **what happened**. Agent VCR lets you **go back, change execution, and resume.**

---

## 2. Codebase Reality (v0.8.0) — EXISTS vs MISSING

### 2.1 EXISTS — reuse as-is (durability layer)

| Module | Purpose |
|---|---|
| `src/agent_vcr/models.py` | Pydantic `Frame` (node/tool/llm/error/checkpoint) + `Session` + `ResumeConfig(FORK/REPLAY/MOCK)` |
| `src/agent_vcr/recorder.py` | Sync append recorder, thread-safe, fsync flush, atomic `manifest.json` |
| `src/agent_vcr/async_recorder.py` | Same via `aiofiles` |
| `src/agent_vcr/player.py` | `load/goto_frame/goto_time/compare_frames/export_state/resume()` |
| `src/agent_vcr/async_player.py` | Async `load` |
| `src/agent_vcr/golden_cache.py` | Ghost/Golden replay: fingerprint `task+model/prompt-hash/commit/tool-hash` → zero-token replay; `CostLedger` for saved tokens/$/ms |
| `src/agent_vcr/integrations/openhands.py` | `ACIDWorkspace`: git-branch isolation, per-frame commits = savepoints, `reset --hard` rollback |
| `src/agent_vcr/integrations/langgraph.py` | `wrap_graph/wrap_node`, `LangGraphCallback`, `@vcr_record` |
| `src/agent_vcr/integrations/crewai.py` | `kickoff/wrap_task`, callbacks, `execute()` monkey-patch |
| `src/agent_vcr/server.py` | FastAPI: `GET /api/sessions`, `/{id}`, `/frames/{i}`, `POST /resume`, `/export`, `POST /api/push`, `WS /ws/live` + watchdog, token auth |
| `src/agent_vcr/tui.py` + `cli.py` | Textual time-travel debugger + `vcr`, `vcr-server`, `sentinel` CLIs |
| `src/agent_vcr/dashboard/` | Static React bundle served at `/dashboard` |
| `src/openhands_sentinel/` | AST guard: func-length, complexity, dup, wildcard imports; `check_and_warn/attach/watch` |
| Storage | JSONL `.vcr` — line 1 `{"type":"session"}`, then `{"type":"frame"}`; golden in `.vcr/golden/<fp16>.vcr` + `index.json` |

### 2.2 MISSING — must build for SuperAgent

> Verified: no `planner|workflow|gateway|sandbox|policy|recovery` modules in `src/`.

1. `planner/` — goal → executable DAG, sub-agent fan-out, budget-aware planning
2. `workflow engine` — durable DAG (triggers/schedules/approvals/retries/branching/compensation); current `resume()` is single-callable only
3. `tool gateway` — unified MCP/HTTP/shell registry, schema validation, rate-limit, approval gates
4. `sandbox` — isolated exec (containers/gVisor/Firecracker); git-branch ≠ isolation
5. `policy engine` — OPA-style allow/deny, secrets redaction, budget caps; Sentinel checks Python style only, no enforcement point
6. `recovery engine` — failure classifier + auto retry/escalate/compensate; only manual `rollback()/fork()` exists
7. Cloud control plane — deployments/envs/secrets ledger, eval/regression harness, analytics

**Architecture rule:** keep VCR core as durability layer (JSONL + git-ACID + Golden replay). Build SuperAgent as new `src/superagent/` **on top**, with policy/sandbox wrapping every tool call *before* frames are recorded.

```
External Services
  ↑
Filesystem | Shell | Browser | MCP | APIs
  ↑
Sandbox Runtime (NEW — E2B/Daytona)
  ↑
State Store | Event Recorder | Checkpoint Store (EXISTS — recorder/player/models)
  ↑
VCR Execution Kernel (EXISTS — harden)
  ↑
SuperAgent | Workflow Engine | Recovery Engine (NEW)
  ↑
Orchestrator → Control Plane API → CLI/SDK/Web Console
```

---

## 3. Verified Tech Stack (2026-09-08)

### 3.1 Orchestration

| Technology | Verified version | Relevance | Source |
|---|---|---|---|
| LangGraph (Python) | `1.2.11` (2026-08-11) | Stateful graphs, checkpointers, durable resume | https://pypi.org/project/langgraph/ |
| LangChain | `1.4.0` (2026-09-03) | Composable LLM components | https://pypi.org/project/langchain/ |
| langchain-core | `1.6.2` (2026-09-04) | Base abstractions | https://pypi.org/project/langchain-core/ |
| @langchain/langgraph (JS) | `1.4.14` (2026-09-04) | TS workers / console | https://www.npmjs.com/package/@langchain/langgraph |
| deepagents | `0.7.13` (2026-09-02) | Harness on LangGraph (subagents, fs, prompt cache) | https://pypi.org/project/deepagents/ |
| CrewAI | `1.15.20` (2026-09-04) | Multi-agent alternative; flows, guardrails | https://pypi.org/project/crewai/ |

### 3.2 Agents SDK + MCP

| Technology | Verified version | Relevance | Source |
|---|---|---|---|
| openai-agents (Python) | `0.22.1` (2026-09-08) | Handoffs, guardrails, sandbox agents | https://pypi.org/project/openai-agents/ |
| @openai/agents (TS) | `0.17.2` (2026-09-08) | TS console equivalent | https://www.npmjs.com/package/@openai/agents |
| MCP spec | `2026-07-28` current | Stateless core, MRTR, Tasks, MCP Apps | https://modelcontextprotocol.io/specification/2026-07-28/changelog |
| mcp (Python SDK) | `2.2.0` (2026-09-07) | Spec client/server | https://pypi.org/project/mcp/ |

### 3.3 API

| Technology | Verified version | Source |
|---|---|---|
| FastAPI | `0.141.1` (2026-07-29) | https://pypi.org/project/fastapi/ |
| Uvicorn | `0.52.4` (2026-08-19) | https://pypi.org/project/uvicorn/ |
| Pydantic v2 | `2.13.5` stable; `2.14.0b1` beta | https://github.com/pydantic/pydantic/releases |

### 3.4 TUI + Web Console

| Technology | Verified version | Source |
|---|---|---|
| Textual | `8.2.8` (2026-06-30) | https://pypi.org/project/textual/ |
| Rich | `15.0.0` (2026-04-12) | https://pypi.org/project/rich/ |
| Next.js | `16.3.4` Active LTS (15 EOL 2026-10-21) | https://www.npmjs.com/package/next |
| React | `19.2.8` (required by Next 16) | https://www.npmjs.com/package/react |
| Node + npm | `24.20.0` LTS Krypton + `npm 11.19.0` | https://nodejs.org/download/release/index.json |

> Local env is `node v20.20.2 + npm 11.6.0` — **upgrade to Node 24 LTS before web work.**

### 3.5 Sandbox

| Technology | Verified version | Verdict 2026 |
|---|---|---|
| E2B (`e2b` / `e2b` npm) | `2.46.4` / `2.46.1` | **Default** — Firecracker microVM, 5-30ms snapshot resume, MCP Catalog |
| Daytona | `0.211.2` | Persistent stateful workspaces, self-host AGPL |
| Modal | `1.5.5` | gVisor + GPU (T4/A10G/H100) |
| Raw Docker | — | Local dev only |

### 3.6 Durable Workflow

| Technology | Verified version | Verdict 2026 |
|---|---|---|
| temporalio (Python) | `1.32.0` | Polyglot/enterprise long-runs; needs worker fleet; determinism constraint fights LLMs |
| Inngest (npm / py) | `4.20.0` / `0.5.19` | **Zero-ops start** — step memoization tolerates non-deterministic LLM, native HITL/rate-limit |
| Trigger.dev | `4.5.16` | Checkpoint-resume TS, minimal boilerplate |

**Decision: Inngest-first, abstract `WorkflowBackend` so Temporal can swap in for enterprise.**

### 3.7 Model Routing + Memory + Eval

| Technology | Verified version |
|---|---|
| LiteLLM | `1.100.0` — single OpenAI-format gateway, 100+ LLMs, failover, spend tracking |
| pgvector | `0.5.0` — default if Postgres exists (<50M vectors, ACID memory) |
| qdrant-client | `1.19.0` — best dedicated VDB (filtered/HNSW, hybrid) |
| weaviate-client | `4.23.1` |
| pinecone | `10.0.0` |
| braintrust | `0.37.0` — **best for CI gates** (trace scorers, PR blocking) |
| langsmith | `0.12.2` — best if on LangGraph Cloud/Fleet |
| ragas | `0.4.3` — RAG faithfulness/recall |
| langfuse | `4.15.1` — OSS self-host tracing |

**Decisions:**
- All model calls via **LiteLLM gateway**.
- Memory: **pgvector if Postgres present, else Qdrant**.
- Eval: **Braintrust for gates**, LangSmith if Fleet; `ragas` for RAG.
- Tools: **MCP `2026-07-28` stateless + `server/discover`**, no sticky sessions.
- Pin: `langgraph>=1.2,<2`, `pydantic>=2.13`, `fastapi==0.141.1`, `uvicorn==0.52.4`.

---

## 4. Roadmap (from design doc §53 + Appendix A)

### Phase 0 — Core Stabilization (1–2 wks) — P0

- [ ] Freeze `Frame/Session/Event` schemas with `version: execution/v1`; export JSON Schema to `schemas/execution/`, `schemas/events/`
- [ ] Harden JSONL: atomic `mkstemp+os.replace`, fsync, corrupt-line skip + `manifest.json` rebuild
- [ ] `make verify` green: `pytest tests/unit/ -v`, `ruff check .`, `mypy src/agent_vcr/ src/openhands_sentinel/`, `tests/benchmarks/test_performance.py` limits
- [ ] Deduplicate `agent vcr ` vs `agent-vcr/`; adopt §54 monorepo: `packages/`, `services/`, `workers/`, `web/`, `schemas/`, `infra/`, `docs/`
- [ ] Exit: replay fidelity >99% on golden set; P0 (execution/checkpoint/rollback/replay/data-integrity) uncompromised

### Phase 1 — Agent VCR 1.0 (2–3 wks)

- [ ] Ghost Replay modes: `FULL_REPLAY / HYBRID_REPLAY / LIVE_REPLAY / BRANCH_REPLAY` on top of `GoldenRunCache`
- [ ] Replay Bundle: `.vcr` + `manifest.json` + git SHAs + model/tool hashes; `vcr reproduce <bundle>` one-liner (§72)
- [ ] Explorer polish: TUI timeline + `/dashboard` bundle
- [ ] Upgrade integrations: `langgraph 1.2.11`, `crewai 1.15.20`, `mcp 2.2.0`
- [ ] Exit: `pip install ai-agent-vcr` → record/fork/replay loop on LangGraph + CrewAI examples

### Phase 2 — Agent CI wedge (2 wks) — "GitHub Actions for agent reliability"

- [ ] `vcr dataset add/select` backed by `.vcr/golden/index.json`
- [ ] GitHub Action: build → deterministic replays → Braintrust (`0.37.0`) scorers → baseline compare → PR comment:
  `1000 replays: 973 pass, 19 degraded, 8 failed | $0.19→$0.27 | 1.3s→2.1s | risk LOW→HIGH`
- [ ] Regression dims: correctness, tool correctness, state transitions, cost, latency, tokens, failure rate, policy violations, escalation rate
- [ ] Exit: demo PR blocks on regression

### Phase 3 — VCR Cloud minimal (3–4 wks)

- [ ] `services/api` (FastAPI `0.141.1`): auth, projects, `POST /api/push`, S3-compatible storage, `WS /ws/live`
- [ ] `web/` on Next `16.3.4` + React `19.2.8` (Node 24): Dashboard, Execution Explorer, Replay Studio
- [ ] Local-first: SDK offline-capable, sync = thin push/pull (§56–57)
- [ ] Exit: `vcr push/pull/replay --cloud` + team share

### Phase 4 — SuperAgent Runtime (4–6 wks)

New `src/superagent/` **above** VCR kernel. Every tool call wrapped pre-record:

```
src/superagent/
  planner.py    # goal → executable DAG with checkpoints + policy bounds
  router.py     # Tool Gateway: MCP 2026-07-28 + HTTP/shell, validate, rate-limit, approval gate
  memory.py     # working + pgvector/Qdrant long-term + redaction
  verifier.py   # pass=commit / fail=recovery
  policy.py     # OPA-style allow/deny, budgets, secrets (Sentinel = one checker)
  budget.py     # token/$/latency caps
  approval.py   # HITL via Inngest waitForEvent / API callback
```

- [ ] Planner emits executable plan (not prose), e.g. `classify→lookup→verify→refund(requires_approval:true)`
- [ ] All LLM via LiteLLM `1.100.0`
- [ ] Flow: `SuperAgent → Intake → Planner → Researcher/Executor/Analyst → Verifier → Commit | Recovery(Retry|Rollback|Fork)`
- [ ] Exit: refund-agent demo with approval gate before payment + full VCR trace

### Phase 5 — Workflow Engine (3–4 wks)

- [ ] YAML DAG (§9): triggers, schedules, steps, agents, tools, conditions, branches, loops, parallel, retries, timeouts, checkpoints, approvals, compensations, joins
- [ ] `WorkflowBackend` abstraction: Inngest `4.20.0` now, Temporal `1.32.0` later
- [ ] Version every run: `workflow:v13 + agent + tool + policy` graph
- [ ] Exit: webhook → multi-agent → checkpoint → approve → compensate on failure

### Phase 6 — Reliability Intelligence (ongoing)

- [ ] Failure clustering, risk scoring (§14), recovery engine (§13), auto-routing (§28), cost opt
- [ ] "Why did my agent fail?" query engine (§49) + Reliability Graph (§48)
- [ ] SLOs (§60): availability, recovery success, replay fidelity, MTTR, regression rate
- [ ] Exit: auto-recovery + risk dashboards from accumulated execution dataset (moat #2/#3)

### Phase 7 — Agent Cloud (later)

- [ ] Hosted sandboxes (E2B/Daytona), deploys, autoscale, secrets/envs, RBAC, audit, billing (subscription + infra + intelligence)

---

## 5. MVP Definition (§79–80)

Credible first commercial cut — 8 items only:

1. VCR Core
2. Ghost Replay
3. Execution Explorer
4. Agent CI
5. Replay Datasets
6. Basic Cloud Sync
7. GitHub integration
8. SuperAgent runtime with checkpoints

User flow:

```
pip install ai-agent-vcr → vcr init → wrap agent → vcr run
→ execution recorded → agent fails → vcr inspect
→ vcr fork --frame 13 → modify prompt/model/tool → vcr replay
→ fix confirmed → vcr dataset add → GitHub Action blocks future regression
```

---

## 6. Technical Priorities (§82)

- **P0 (never sacrifice):** execution / checkpoint / rollback / replay correctness, data integrity
- **P1:** framework integrations, SDK ergonomics, viewer, Ghost Replay, Agent CI
- **P2:** cloud, workflow engine, SuperAgent, recovery engine
- **P3:** advanced intelligence, auto-optimization, enterprise, marketplace

---

## 7. Suggested Repo Structure (§54, adapted)

```
agent-vcr/
  packages/
    vcr-core/            # ← src/agent_vcr today
    vcr-sdk/
    vcr-cli/             # ← cli.py, tui.py
    vcr-integrations/    # ← langgraph/crewai/openhands
    superagent-runtime/  # ← NEW Phase 4
  services/
    api/ scheduler/ orchestrator/ evaluator/ recovery/ ingest/ billing/
  workers/
    execution/ replay/ evaluation/ sandbox/
  web/
    dashboard/ execution-studio/ replay-studio/ workflow-builder/  # Next 16.3.4
  schemas/
    execution/ events/ workflows/ policies/
  infra/
    docker/ kubernetes/ terraform/
  docs/
```

API boundaries (§55): `Agent code ↔ VCR SDK ↔ Execution Kernel ↔ Event Interface ↔ Storage Interface`. No cloud details leak into local SDK.

---

## 8. Next Actions

1. Confirm stack: Python kernel + Next console, Inngest-first, E2B-first, Braintrust-first, LiteLLM gateway — then scaffold `src/superagent/` + `schemas/` + pinned `pyproject` deps.
2. Start Phase 0 now: schema versioning + replay bundle + dep upgrades (`langgraph 1.2.11`, `crewai 1.15.20`, `mcp 2.2.0`, `fastapi 0.141.1`, `litellm 1.100.0`).
3. Upgrade local Node `20.20.2` → `24.20.0` LTS before any `web/` work.

---

## 9. Verification Log

- `langgraph 1.2.11 / langchain 1.4.0 / crewai 1.15.20 / mcp 2.2.0 / fastapi 0.141.1 / uvicorn 0.52.4 / pydantic 2.13.5 / litellm 1.100.0 / textual 8.2.8 / temporalio 1.32.0 / inngest(npm) 4.20.0 / qdrant-client 1.19.0 / langsmith 0.12.2 / braintrust 0.37.0 / e2b 2.46.4 / daytona 0.211.2 / openai-agents 0.22.1` — all via `https://pypi.org/pypi/{pkg}/json` on 2026-09-08.
- `next 16.3.4 / react 19.2.8 / inngest 4.20.0 / node v20.20.2 / npm 11.6.0` — via `npm view` + `node --version` on 2026-09-08.
