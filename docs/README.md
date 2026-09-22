# Agent VCR Documentation

This directory contains technical documentation that complements the repository root:
`README.md` is the product and quick-start guide, `CHANGELOG.md` is the canonical release
history, and `SECURITY.md` describes vulnerability reporting.

## Repository map

```text
src/                 distributable Python packages
tests/               unit, end-to-end, and benchmark suites
docs/                current guides, deployment notes, integrations, and plans
schemas/             versioned JSON contract artifacts
examples/            runnable, credential-free examples where possible
scripts/             repository maintenance and artifact-generation tools
.github/             CI workflows and repository automation
```

Generated output does not belong in source control: `.vcr/` is local runtime data, and caches,
coverage, build output, and Python bytecode are removed with `make clean`. The clean target
intentionally preserves `.venv/` and `.vcr/`.

## Current engineering documentation

- [Architecture](architecture.md) — runtime tiers, data model, and subsystem boundaries.
- [Development](development.md) — local setup, quality gates, and repository conventions.
- [Workflows](workflow.md) — workflow definitions, execution, recovery, and operations.
- [Railway deployment](deployment/railway.md) — optional Supabase Auth and Stripe control-plane
  setup. It remains opt-in and does not alter local-first use.
- [Integrations](integrations/) — supported external integration guides and templates.

## Reference material

- [`../schemas/`](../schemas/) — versioned, published JSON contracts for execution, workflows,
  cloud, CI, intelligence, and SuperAgent APIs.
- [`dev/bench/`](dev/bench/) — benchmark history and static dashboard assets.

## Historical plans

The [plans](plans/) directory retains internal source plans for traceability. They are not
release commitments and must not override the current implementation or public documentation.
