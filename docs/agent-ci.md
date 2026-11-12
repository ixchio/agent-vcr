# ✅ Agent CI

Unit tests for agent behavior: pin a run you trust as the baseline, record the agent again on
every pull request, and fail the PR when it breaks a rule or drifts from the baseline.

**TL;DR**

- Golden run → **dataset** case with expectations → your test harness records a
  **candidate** → `vcr ci run` gives a ✅ pass / ⚠️ degraded / ❌ fail verdict and an exit code.
- Offline and deterministic: the gate compares recordings. It never calls an LLM.
- One step in GitHub Actions: `uses: ixchio/agent-vcr@v1.7.1`.

## Try it in 60 seconds

Save this stand-in agent as `refund_agent.py`. In real life it's your agent, run against
controlled inputs with mocked or sandboxed tools.

```python
"""A tiny stand-in agent. Swap the body for your real agent with mocked tools."""
import sys

from agent_vcr import FrameMetadata, VCRRecorder


def run(out_dir: str, session_id: str, *, careless: bool = False) -> None:
    rec = VCRRecorder(output_dir=out_dir)
    rec.start_session(session_id, if_exists="overwrite")
    state = {"order": "A-1", "charges": 2}
    rec.record_checkpoint(state, label="approval")
    rec.record_tool_call("lookup_payments", {"order": "A-1"}, {"duplicate": True}, latency_ms=40)
    if careless:
        rec.record_tool_call("delete_customer", {"order": "A-1"}, {"ok": True}, latency_ms=10)
    rec.record_step("refund", state, {**state, "status": "refunded"},
                    metadata=FrameMetadata(tokens_used=1200 if careless else 800,
                                           cost_usd=0.012 if careless else 0.008,
                                           latency_ms=900))
    rec.record_checkpoint({"status": "refunded"}, label="completed")
    rec.save()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], careless="--careless" in sys.argv)
```

Record a run you've reviewed, pin it as a case, record a candidate and gate it:

```bash
python refund_agent.py .vcr/reviewed refund-golden

vcr dataset create refunds
vcr dataset add refunds \
  --task "refund a duplicate payment" \
  --case-name duplicate-payment \
  --vcr .vcr/reviewed/refund-golden.vcr \
  --tags payments,critical \
  --max-cost 0.05 \
  --max-latency-ms 5000 \
  --max-tokens 4000 \
  --expect-state '{"status":"refunded"}' \
  --forbid-tools delete_customer \
  --require-checkpoints approval,completed

python refund_agent.py .vcr/candidates duplicate-payment
vcr ci run --dataset refunds --candidates .vcr/candidates --fail-on degraded
```

```text
| case | verdict | cost Δ | latency Δ | tokens Δ | notes |
|---|---|---|---|---|---|
| duplicate-payment | ✅ pass | +0.0000 | -0ms | +0 | — |
```

Now break it. The careless version calls a forbidden tool:

```bash
python refund_agent.py .vcr/candidates duplicate-payment --careless
vcr ci run --dataset refunds --candidates .vcr/candidates --fail-on degraded
echo "exit code: $?"
```

```text
| duplicate-payment | ❌ fail | +0.0040 | +10ms | +400 | forbidden tool(s) used: delete_customer; new tool(s) vs baseline: delete_customer |
exit code: 1
```

## How it works

```text
 reviewed run ──vcr dataset add──▶ .vcr/golden/<fingerprint>.vcr   (baseline, commit it)
                                   .vcr/datasets/refunds.json      (cases, commit it)
                                              │
 code under review ──your harness──▶ .vcr/candidates/<case>.vcr    (made fresh in CI)
                                              │
                              vcr ci run ─────┴──▶ verdict per case ─▶ report + exit code
```

- **A dataset** is a versioned JSON file of cases. Each case points at a golden run in the golden
  cache by fingerprint (task + identity) and carries its expectations. Adding or removing a case
  bumps the version. `vcr dataset add` copies the recording into the golden cache, so later edits
  to the original file don't change the baseline.
- **Candidates are matched by file name.** A candidate is `<case name>.vcr`, or the case name's
  *slug*: lowercased, with every run of characters other than `a-z 0-9 . _ -` turned into `-`.
  The case name defaults to the `--task` string, so task `Refund $10` expects `refund-10.vcr`.
  `vcr dataset add` prints the expected file name. Files that match no case get a warning.
- **Two kinds of problems.** Breaking a rule you wrote is a ❌ **fail**. Drifting from the
  baseline is ⚠️ **degraded**:

| ❌ Fail: a case rule is broken | ⚠️ Degraded: drift from the baseline |
|---|---|
| Any error frame (unless the case was added with `--allow-errors`) | Cost more than **+50%** over the baseline |
| Total cost, latency or tokens over `--max-cost` / `--max-latency-ms` / `--max-tokens` | Latency more than **+100%** |
| Final state doesn't contain `--expect-state` (partial match: nested keys, list prefixes) | Tokens more than **+50%** |
| A `--forbid-tools` tool was called | Any rise at all when the baseline value was 0 |
| A `--require-checkpoints` label is missing | A tool or model the baseline never used |
| The candidate or golden file can't be loaded | A baseline checkpoint is missing, or the final state differs |

- **Risk** sums it up: `HIGH` if anything failed, `MEDIUM` if anything degraded or a candidate is
  missing, else `LOW`. A missing candidate always breaks the gate. A green report means your
  declared expectations held. It doesn't prove the agent is correct.

## Everyday recipes

### Add a case from Python (custom thresholds, identity)

The CLI uses the default drift limits. From Python you can set any `Expectation` field, and pin
an identity so the case only matches a golden run made with that model and prompt:

```python
from agent_vcr import FrameMetadata, GoldenRunCache, VCRRecorder
from agent_vcr.datasets import Expectation, ReplayDataset

rec = VCRRecorder(output_dir=".vcr/reviewed")
rec.start_session("summarize-golden", if_exists="overwrite")
rec.record_step("summarize", {"doc": "q3.pdf"}, {"summary": "Revenue up"},
                metadata=FrameMetadata(tokens_used=500, cost_usd=0.005))
rec.save()

cache = GoldenRunCache(".vcr/golden")
ds = ReplayDataset.load("refunds")
ds.add_case(
    "summarize the q3 report",
    rec,
    name="q3-summary",
    identity=cache.build_identity(model="demo-model@2026-01", prompt_hash="sha256:abc123"),
    expect=Expectation(max_cost_usd=0.02, max_token_delta=1.0, max_latency_delta=None),
    golden_cache=cache,
)
print(ds.version, [c.name for c in ds.cases()])
```

`max_cost_delta`, `max_latency_delta` and `max_token_delta` are fractions of the baseline
(`1.0` = +100%). `None` turns that check off. Identity works the same as in
[Ghost Replay](recording-and-replay.md#-ghost-replay-re-use-a-run-you-trust).

### Produce real candidates

Your CI job runs the code under review and writes one recording per case into the candidates
folder. Use the same inputs as the golden run, and mock or sandbox anything with side effects.
**Don't copy golden files into the candidates folder**: a recording that equals itself proves
nothing.

The repo's `examples/agent_ci_candidates.py` is a demo helper that does exactly that copying, so
you can watch the gate work. `--break-case NAME` drops that case's last frame to fake a
regression:

```bash
python examples/agent_ci_candidates.py --dataset refunds --candidates-dir demo-candidates \
  --break-case duplicate-payment
vcr ci run --dataset refunds --candidates demo-candidates
```

### Gate only some cases

```bash
vcr ci run --dataset refunds --candidates .vcr/candidates --tags critical
```

A case runs only when it has **all** the listed tags.

### Run it in GitHub Actions

Commit `.vcr/datasets/` and `.vcr/golden/`, then add a workflow:

```yaml
name: Agent regression

on:
  pull_request:

jobs:
  agent-ci:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write    # only needed for the comment step
    steps:
      - uses: actions/checkout@v4

      - name: Record candidates with the code under review
        run: python tests/generate_agent_candidates.py   # your harness

      - name: Compare with reviewed baselines
        uses: ixchio/agent-vcr@v1.7.1
        with:
          dataset: refunds
          candidates-dir: .vcr/candidates
          fail-on: degraded
          package: ai-agent-vcr==1.7.1

      - name: Comment the report on the PR
        if: github.event_name == 'pull_request' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: fs.readFileSync('agent-ci-report.md', 'utf8'),
            });
```

The action sets up Python, installs `package`, runs `vcr ci run`, appends the Markdown report to
the job summary, adds an error or warning annotation per failed or degraded case, and uploads
`agent-ci-report.json` + `agent-ci-report.md` as the `agent-ci-report` artifact, even when the
gate fails. Its `report` output is the JSON path. It doesn't comment on the PR itself: that's the
last step above. Pin the action and the package to a tag or commit SHA according to your
supply-chain policy. `examples/agent-ci.yml` has a full copy.

| Input | Default | Meaning |
|---|---|---|
| `dataset` | (required) | Dataset name |
| `candidates-dir` | `.vcr/candidates` | Folder of `<case>.vcr` candidates |
| `datasets-dir` | `.vcr/datasets` | Where dataset JSON lives |
| `golden-dir` | `.vcr/golden` | Golden-run cache |
| `fail-on` | `failed` | `degraded`, `failed` or `never` |
| `package` | `ai-agent-vcr` (latest) | Pip spec to install |
| `python-version` | `3.12` | Python for the check |

### Read the reports

- `--markdown FILE` writes the PR-comment table you saw above. Without `--quiet`, it's also
  printed to the terminal.
- `--report FILE` writes the full report as strict JSON (any `NaN`/`Infinity` becomes `null`):
  counts, `missing_cases`, `risk`, total deltas, and per case the verdict, reasons and the full
  baseline diff (frames, tools and models added or removed, deltas, errors, missing checkpoints).
- Inside GitHub Actions, `$GITHUB_STEP_SUMMARY` gets the Markdown (skip it with `--no-summary`),
  and each failed or degraded case becomes an annotation. Case names and reasons come from
  recordings, so they're escaped and can't inject workflow commands.

### Update a baseline on purpose

A golden run is evidence, not permanent truth. When behavior changes for a good reason, inspect
the new candidate's state and tool frames, check its external effects some other way, update the
expectations, then swap the case:

```bash
vcr dataset show refunds
vcr dataset remove refunds duplicate-payment
vcr dataset add refunds --task "refund a duplicate payment" --case-name duplicate-payment \
  --vcr .vcr/candidates/duplicate-payment.vcr --expect-state '{"status":"refunded"}'
```

Say why the baseline changed in the commit message, so the dataset's history explains itself.

## Exit codes

| `vcr ci run` exits | When |
|---|---|
| `0` | Every case passed, or only degraded cases with `--fail-on failed` (the default), or anything at all with `--fail-on never` |
| `1` | A case failed, a candidate is missing, or a case degraded under `--fail-on degraded` |
| `1` | It couldn't run: the `--candidates` folder or the dataset doesn't exist |
| `2` | Bad arguments (for example, a missing `--dataset`) |

`vcr dataset remove` exits `2` when the case doesn't exist. Other `vcr dataset` errors exit `1`.

## What it can't do

- **It doesn't run your agent.** Candidates must already exist when the gate starts. Making them
  well (same inputs, mocked externals) is your harness's job, and it's most of the value.
- **It checks what you declared, plus drift.** Final-state checks look at the *last* frame's
  output, and only at the keys you list. Tools are seen through `tool:<name>` frames, so a call
  your agent didn't record is invisible.
- **Latency is noisy.** CI runners vary, which is why the default drift limit is +100%. Use
  `--max-latency-ms` for a hard ceiling instead of trusting the delta.
- **Cost and token checks use what the recordings say.** If your harness records no tokens or
  cost, those checks have nothing to compare.
- **Recordings are artifacts.** The JSON report, the Markdown and the uploaded artifact contain
  case names, tool names and error text. Keep secrets out of recordings, and keep secret-bearing
  recordings out of public repos and artifacts.
- **The CLI can't set drift thresholds or identity.** Use the Python API for those.

## Cheat sheet

| Command | What it does |
|---|---|
| `vcr dataset create NAME` | Create an empty dataset (`v1`) |
| `vcr dataset add NAME --task T --vcr FILE [--case-name N] [--tags a,b] [--max-cost USD] [--max-latency-ms MS] [--max-tokens N] [--expect-state JSON] [--forbid-tools a,b] [--require-checkpoints a,b] [--allow-errors]` | Pin a recording as a golden case |
| `vcr dataset list` / `show NAME` / `remove NAME CASE` | Inspect and edit datasets |
| `vcr ci run --dataset NAME --candidates DIR [--tags a,b] [--fail-on degraded\|failed\|never] [--report JSON] [--markdown MD] [--quiet] [--no-summary]` | Run the gate |
| `--datasets-dir DIR` / `--golden-dir DIR` | On every `dataset` and `ci` command; default `.vcr/datasets` and `.vcr/golden` |
| `ReplayDataset.create / load / add_case / remove_case / select(tags)` | Python API (`agent_vcr.datasets`) |
| `run_regression(dataset, {case: path_or_player}, GoldenRunCache(...))` | Python API (`agent_vcr.regression`); returns a `RegressionReport` with `.to_markdown()` and `.save(path)` |

For sharing a single run outside CI, export a
[`.vcrb` bundle](recording-and-replay.md#ship-a-bug-report-as-a-bundle).

**Next:** [Recording & Ghost Replay](recording-and-replay.md) (how golden runs are made) ·
[Sentinel](sentinel.md) (gate the code your agent writes, too) ·
[ACID workspace](acid-workspace.md) (roll back a failed attempt before it reaches CI)
