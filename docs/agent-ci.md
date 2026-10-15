# Agent CI regression gates

Agent CI compares candidate recordings against reviewed golden recordings. It is deterministic and
does not call an LLM: candidate runs must already exist when the gate starts.

```text
.vcr/datasets/refunds.json       versioned cases and expectations
.vcr/golden/<fingerprint>.vcr    reviewed baseline recording
.vcr/candidates/<case-name>.vcr  candidate recording from your test harness
```

## Create a dataset

```bash
vcr dataset create refunds

vcr dataset add refunds \
  --task "refund a duplicate payment" \
  --case-name duplicate-payment \
  --vcr .vcr/reviewed/refund.vcr \
  --tags payments,critical \
  --max-cost 0.05 \
  --max-latency-ms 5000 \
  --max-tokens 4000 \
  --expect-state '{"status":"refunded"}' \
  --forbid-tools delete_customer \
  --require-checkpoints approval,completed
```

Adding a case copies the reviewed run into the golden cache and increments the dataset version. A
case requires no error frames by default and can enforce absolute budgets, final-state fields,
forbidden tools, required checkpoints, and relative cost/latency/token drift.

## Produce real candidates

Your test harness runs the code under review and writes one recording per case. The filename must
match the case name:

```text
.vcr/candidates/duplicate-payment.vcr
```

```bash
python examples/agent_ci_candidates.py
```

Do not copy the golden file into the candidates directory. That only proves a recording equals
itself; a useful gate records current behavior with controlled inputs and mocked or sandboxed
external dependencies.

## Run the gate

```bash
vcr ci run \
  --dataset refunds \
  --candidates .vcr/candidates \
  --fail-on degraded \
  --report agent-ci-report.json \
  --markdown agent-ci-report.md
```

`--fail-on degraded` rejects degraded and failed cases. `failed` allows degraded cases, and `never`
always exits successfully for observation-only adoption. A passing report means the declared
expectations passed; it is not a universal correctness proof.

## GitHub Actions

```yaml
name: Agent regression

on:
  pull_request:

jobs:
  agent-ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate candidate recordings
        run: python tests/generate_agent_candidates.py

      - name: Compare with reviewed baselines
        uses: ixchio/agent-vcr@v1.5.4
        with:
          dataset: refunds
          candidates-dir: .vcr/candidates
          fail-on: degraded
          package: ai-agent-vcr==1.5.4
```

Pin the action to a release tag or commit SHA according to your supply-chain policy. The composite
action installs the requested package, runs the regression engine, appends a step summary, and
uploads JSON and Markdown reports even on failure.

## Review discipline

A golden run is evidence, not permanent truth. Inspect candidate state and tool frames, verify
external effects independently, update expectations before replacing the baseline, record why the
baseline changed, and keep secret-bearing recordings out of public artifacts.

For portable sharing, use a [`vcrb` replay bundle](recording-and-replay.md#portable-bundles-and-privacy).
