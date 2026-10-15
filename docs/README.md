# Agent VCR documentation

Agent VCR is the local-first intervention layer for coding and stateful AI agents. It records
execution as inspectable JSONL, joins agent state to filesystem savepoints, replays verified runs
without another model call, and turns recordings into regression gates.

```bash
pip install ai-agent-vcr
```

The core package needs no model key, hosted account, or agent framework. Python 3.9 through 3.13
is tested in CI.

## Start here

- [Record, inspect, resume, and Ghost Replay](recording-and-replay.md) — recorder/player APIs,
  identity-safe replay, bundles, and privacy boundaries.
- [Roll back state and files together](acid-workspace.md) — Git savepoints, content-aware escrow
  for `.env` and other local files, limits, and recovery semantics.
- [Gate agent changes in CI](agent-ci.md) — versioned golden datasets, candidate naming,
  expectations, exit thresholds, and the reusable GitHub Action.
- [Connect a framework](integrations/README.md) — raw Python, LangGraph, CrewAI, OpenHands,
  Claude Code hooks, and the generic integration boundary.

## Runtime guides

- [Workflow engine](workflow.md) — durable run state, schedules, crash resume, approvals, and saga
  compensation.
- [SuperAgent runtime](superagent.md) — validated plans, tool schemas, policy, budgets, approvals,
  verification, and recorded decisions.
- [Architecture](architecture.md) — package layers, data contracts, persistence, and subsystem
  boundaries.
- [Development and release verification](development.md) — local setup, tests, benchmark gates,
  packaging, and `make verify`.

## Operations and extensions

- [Live documentation site](https://ixchio.github.io/agent-vcr/) — product overview, demo, and
  developer documentation map.
- [Self-hosted server and dashboard](../README.md#self-hosted-store) — local HTTP/WebSocket access,
  scoped project tokens, and the production boundary.
- [Railway deployment](deployment/railway.md) — optional Supabase Auth, Stripe, Postgres, and
  volume-backed deployment. This is a foundation, not a finished managed SaaS.
- [n8n integration](integrations/n8n.md) — trigger workflows, wait for callbacks, and inspect
  failures through the HTTP API.
- [Sentinel](sentinel/index.html) — standalone and OpenHands-connected AST quality checks.

## Contracts and release evidence

- [`../schemas/`](../schemas/) contains versioned JSON contracts for execution, bundles, datasets,
  workflows, cloud APIs, intelligence, and SuperAgent.
- [`../examples/`](../examples/) contains credential-free examples for the core runtime, async
  APIs, integrations, Agent CI, ACID rollback, workflows, and Sentinel.
- [`dev/bench/`](dev/bench/) is the benchmark dashboard. Limits are local regression ceilings, not
  universal production latency promises.
- [`../CHANGELOG.md`](../CHANGELOG.md) is the release history.
- [`../SECURITY.md`](../SECURITY.md) explains private vulnerability reporting.

## Boundaries you should know

Agent VCR can restore a Git workspace and eligible pre-existing untracked/ignored files. It cannot
undo an email, payment, database write, or arbitrary third-party API call. Use approval gates before
consequential tools and workflow compensation for reversible external operations.

Recordings can contain prompts, tool inputs, outputs, and state. Redact secrets before recording or
keep `.vcr` storage private. The ACID local-file escrow is deliberately separate from Git and VCR
recordings, but it does not sanitize data your application explicitly records.

## Repository map

```text
src/                 distributable Python packages
tests/               unit, end-to-end, and benchmark suites
docs/                current guides, deployment notes, and historical plans
schemas/             versioned JSON contract artifacts
examples/            runnable, credential-free examples where possible
scripts/             repository maintenance and artifact-generation tools
.github/             CI workflows and repository automation
```

The [plans](plans/) directory is retained for traceability. Those files describe earlier design
intent and are not current release commitments. Current behavior is defined by the package, tests,
schemas, and the guides above.

Copyright © 2026 QueryXLabs and ixchio.
