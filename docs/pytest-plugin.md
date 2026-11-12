# 🧪 pytest plugin: record once, replay forever

Test your agent the way you already test the rest of your code. The first run calls the model
and keeps the recording. Every run after that replays it, so CI spends zero tokens.

**TL;DR**

- Install `ai-agent-vcr` (1.7.1 or later) and pytest (6.2 or later) picks the plugin up. You get
  two fixtures, `agent_vcr` and `vcr_recorder`, and one marker, `@pytest.mark.agent_vcr`. Tests
  that use none of them run exactly as before.
- `agent_vcr.run(task, agent)` runs your agent live once and saves that run as a **golden**.
  After that it returns the recorded outputs through Ghost Replay and doesn't call your agent.
- `agent_vcr.assert_matches_golden()` applies the [Agent CI](agent-ci.md) rules. A fresh run that
  regresses on behavior, cost, tokens or latency fails the test, and the assertion message shows
  the diff.
- Run CI with `AGENT_VCR_RECORD=none` (or `--agent-vcr-record=none`). A missing golden then
  fails the test and tells you how to record it. The model is never called.

## Try it in 60 seconds

Save this as `test_triage.py`. The agent is a stand-in that makes up its "LLM call". Your real
agent calls a model and records what it did on the recorder it's given.

```python
import os

import pytest


def triage(rec):
    """A stand-in agent. Yours calls a model; it records what it did on `rec`."""
    careless = os.environ.get("CARELESS") == "1"
    rec.record_llm_call(
        "demo-model",
        [{"role": "user", "content": "I was charged twice"}],
        {"content": "billing"},
        tokens_input=500,
        tokens_output=900 if careless else 300,
        latency_ms=400.0,
        cost_usd=0.008,
    )
    rec.record_tool_call("lookup_payments", {"order": "A-1"}, {"duplicate": True}, latency_ms=40)
    if careless:
        rec.record_tool_call("delete_customer", {"order": "A-1"}, {"ok": True}, latency_ms=10)
    return {"queue": "billing", "status": "refunded"}


@pytest.mark.agent_vcr(task="triage ticket t_123", identity={"model": "demo-model"})
def test_triage(agent_vcr):
    run = agent_vcr.run(triage)
    assert run.output == {"queue": "billing", "status": "refunded"}
    agent_vcr.assert_matches_golden(max_cost_usd=0.05)
```

Run it twice:

```bash
pytest test_triage.py -q     # 1st run: calls the agent, saves the golden
pytest test_triage.py -q     # every run after: replays it, the agent isn't called
```

```text
agent-vcr: 1 run(s) recorded, 1 saved as golden
1 passed in 0.16s

agent-vcr: 1 run(s) replayed (agent not called, 800 tokens / $0.0080 not spent)
1 passed in 0.16s
```

Now make the agent careless and re-record it against the golden:

```bash
CARELESS=1 pytest test_triage.py -q --vcr-record=all
```

```text
E       AssertionError: agent_vcr: DEGRADED vs golden (fail_on=degraded)
E         test:      test_triage.py::test_triage
E         task:      'triage ticket t_123'
E         golden:    .vcr/pytest/golden/test_triage.py--test_triage-b903b9e789/935739b6f2eecdde.vcr
E         candidate: /tmp/pytest-of-you/pytest-131/agent_vcr-test_triage.py--test_triage-b903b9e7890/candidate-0.vcr
E         reasons:
E           - tokens +75% vs baseline
E           - new tool(s) vs baseline: delete_customer
E         golden -> this run:
E           frames   3 -> 4  (+33%)
E           tokens   800 -> 1,400  (+75%)
E           cost     $0.0080 -> $0.0080
E           latency  440ms -> 450ms  (+2%)
E           tools    +delete_customer
E       To accept this run as the new golden:
E         pytest --vcr-record=all --vcr-fail-on=never 'test_triage.py::test_triage'

agent-vcr: 1 run(s) recorded, 0 saved as golden; 1 not saved because the test failed
1 failed in 0.17s
```

The failed run did **not** replace the golden. Commit `.vcr/pytest/golden/` and gate CI with:

```bash
AGENT_VCR_RECORD=none pytest -q
```

The environment variable is the CI-safe spelling: it works even when another plugin, such as
pytest-vcr, owns `--vcr-record`. `--agent-vcr-record=none` and `--vcr-record=none` do the same.

Delete the golden folder and the same command fails without calling anything:

```text
agent_vcr: no golden recording for this test, and recording is off (--vcr-record=none), so the agent was not called.
  test:      test_triage.py::test_triage
  task:      'triage ticket t_123'
  identity:  {"model": "demo-model"}
  looked in: .vcr/pytest/golden/test_triage.py--test_triage-b903b9e789
Record it once where the agent can run, review it, and commit the folder:
  pytest --vcr-record=once 'test_triage.py::test_triage'
```

## How it works

```text
 agent_vcr.run(task, agent, identity=)
        │
        ├─ golden for task + identity, and mode is none/once ──▶ Ghost Replay (FULL)
        │      the agent is not called · run.output = recorded value · run.ledger = savings
        │
        └─ no golden (once), or mode is all ──▶ agent(recorder) runs live, in memory
               │
        assert_matches_golden() ── Agent CI rules vs the previous golden ──▶ pass / fail
               │
        teardown: test passed? ──yes──▶ .vcr/pytest/golden/<test>/<fingerprint>.vcr
                                ──no───▶ golden left as it was
```

| `--vcr-record` | Golden exists | No golden |
|---|---|---|
| `none` | Replay | **Fail** with instructions. The agent isn't called |
| `once` (default) | Replay | Run live, save the golden if the test passes |
| `all` | Run live, compare with the golden, replace it if the test passes | Run live, save the golden if the test passes |

- **A golden is keyed by test, task and identity.** Each test gets its own folder,
  `<vcr-dir>/golden/<file>--<test>-<hash>/`: up to 60 readable characters of the node id plus a
  hash of the exact id, so `[a/b]` and `[a-b]`, or `[A]` and `[a]` on a case-insensitive disk,
  never share a folder. It holds a [golden-run cache](recording-and-replay.md#-ghost-replay-re-use-a-run-you-trust):
  `index.json` plus one `<fingerprint>.vcr` per task + identity. The fingerprint is the same as
  in `GoldenRunCache`: the task is trimmed and lowercased, and the identity is normalized with
  sorted keys.
- **A golden is written at teardown, and only when the test passed.** A run that breaks your
  assertions or `assert_matches_golden` never becomes the baseline. The summary line says how
  many runs were not saved.
- **`run.output` is always the recorded value.** It goes through the recording's serializer
  even on the first run, so the first run and every replay see the same thing. Return dicts,
  lists, strings and numbers. A Pydantic model or a dataclass comes back as a plain dict.
- **What gets compared.** On a live run, the baseline is the golden for the same task and
  identity. If your identity changed (a new model or prompt hash), it's the newest golden for
  the same task, and the message says which identity keys changed. On a replay the candidate
  *is* the golden, so only your expectations (budgets, forbidden tools, expected state) are
  checked. There's no drift to measure.
- **Verdicts follow Agent CI.** Breaking one of your expectations is ❌ **fail**. Drifting from
  the baseline (cost or tokens +50%, latency +100%, a new tool or model, a dropped checkpoint,
  a different final state) is ⚠️ **degraded**. `--vcr-fail-on` picks what fails the test.
  The default here is `degraded`, because drift is the regression you're testing for.
  `vcr ci run` defaults to `failed`.
- **Latency drift below 50 ms is ignored.** When the whole golden run took under 50 ms, a
  "+100%" is timer noise, so the plugin skips the latency drift check by default. Pass
  `max_latency_delta=` to force it. A `max_latency_ms` ceiling always applies.
- **Which setting wins:** the command line (`--vcr-record` / `--agent-vcr-record`), then
  `$AGENT_VCR_RECORD`, then the `agent_vcr_record` ini option, then the marker's `record=`, then
  `once`. A global `none` always beats a marker, so CI stays replay-only.
- **Inert unless used.** At startup the plugin registers options and a marker, and that's all.
  Agent VCR's modules load only when a test asks for one of the fixtures. Folders are created
  only when something is recorded.

## Everyday recipes

### Pass inputs to your agent and pin what makes a golden valid

`agent` receives only the recorder, so bind your inputs with a `lambda`. Put everything that
should invalidate the golden into `identity`. When the prompt changes, `once` records a fresh
golden and `none` fails in CI until you do:

```python
import hashlib

from agent_vcr import GoldenRunCache

PROMPT = "You triage support tickets. Answer with one queue name."
IDENTITY = GoldenRunCache.build_identity(
    model="demo-model@2026-01",
    prompt_hash="sha256:" + hashlib.sha256(PROMPT.encode()).hexdigest()[:12],
)


def triage(ticket, rec):
    rec.record_llm_call("demo-model@2026-01",
                        [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": ticket["text"]}],
                        {"content": "billing"}, tokens_input=60, tokens_output=2,
                        latency_ms=300.0, cost_usd=0.0006)
    return {"queue": "billing"}


def test_triage_billing(agent_vcr):
    ticket = {"id": "t_123", "text": "I was charged twice"}
    run = agent_vcr.run("triage t_123", lambda rec: triage(ticket, rec), identity=IDENTITY)
    assert run.output == {"queue": "billing"}
```

Replay has no idea your code changed. A `code_commit=` or a hash of your tool schemas in the
identity is what makes a stale golden miss instead of quietly replaying.

### Test a step pipeline

Instead of a callable, pass a list of `(name, step)` pairs and an `input=` state. Each step gets
the previous step's output and returns the next state, or `(state, usage)` where usage is a dict
of `FrameMetadata` fields. Each step becomes one frame. If a step doesn't report `latency_ms`,
its wall time is recorded. A hash of `input` is added to the identity for you
(`input_sha256`), so changing the input never replays a stale golden.

```python
def classify(state):
    return {**state, "queue": "billing"}, {"tokens_used": 300, "cost_usd": 0.003}


def draft(state):
    return {**state, "reply": "Refund on its way"}, {"tokens_used": 900, "cost_usd": 0.009}


def test_reply_pipeline(agent_vcr):
    run = agent_vcr.run(
        "draft a reply",
        [("classify", classify), ("draft", draft)],
        input={"text": "I was charged twice"},
    )
    assert run.output["reply"] == "Refund on its way"
    assert [o.get("queue") for o in run.outputs] == ["billing", "billing"]
```

`run.output` is the final state. `run.outputs` holds every frame's output.

### Set budgets and rules

`assert_matches_golden()` takes any [`Expectation`](agent-ci.md#add-a-case-from-python-custom-thresholds-identity)
field as a keyword. An unknown keyword raises `TypeError`, so a typo can't switch a check off.

```python
import pytest


def refund_agent(rec):
    rec.record_checkpoint({"order": "A-1"}, label="approval")
    rec.record_tool_call("lookup_payments", {"order": "A-1"}, {"duplicate": True}, latency_ms=40)
    rec.record_step("refund", {"order": "A-1"}, {"status": "refunded"})
    return {"status": "refunded"}


@pytest.mark.agent_vcr(task="refund a duplicate payment")
def test_refund_rules(agent_vcr):
    agent_vcr.run(refund_agent)
    agent_vcr.assert_matches_golden(
        max_cost_usd=0.05,
        max_tokens=4000,
        expect_state={"status": "refunded"},
        forbid_tools=["delete_customer"],
        require_checkpoints=["approval"],
        max_token_delta=0.2,
    )
```

| Keyword | Breaks it → | Default |
|---|---|---|
| `max_cost_usd`, `max_latency_ms`, `max_tokens` | ❌ fail | off |
| `require_no_errors` | ❌ fail on any error frame | `True` |
| `expect_state` | ❌ fail unless the final state contains it (partial match) | off |
| `forbid_tools`, `require_checkpoints` | ❌ fail | off |
| `max_cost_delta`, `max_token_delta` | ⚠️ degraded above this fraction of the baseline | `0.5` |
| `max_latency_delta` | ⚠️ degraded above this fraction of the baseline | `1.0` (skipped under 50 ms) |
| `fail_on=` | Overrides `--vcr-fail-on` for this call | |
| `expect=` | A whole `Expectation` object; keywords are applied on top | |

It returns the `CaseResult` (`.verdict`, `.reasons`, `.diff`). A verdict that `fail_on` lets
through is raised as an `AgentVCRWarning` instead, so it still shows up in the warnings summary.

### Re-record after a change you meant to make

```bash
pytest test_recipes.py -q --vcr-record=all -k refund                      # compare, keep golden on failure
pytest test_recipes.py -q --vcr-record=all --vcr-fail-on=never -k refund  # accept the new run as golden
```

Look at what changed (the candidate path in the failure message opens in `vcr <file>`), then
commit the updated golden with a message that says why. Goldens for an old identity stay in
the test's folder. Delete them once nothing uses them.

### See what replay saved

```python
def test_ledger(agent_vcr):
    run = agent_vcr.run("draft a reply", [("classify", classify), ("draft", draft)],
                        input={"text": "I was charged twice"})
    if run.replayed:
        print("\n", run.ledger.summary()["saved"])
    else:
        print("\n recorded", run.tokens, "tokens")
```

```text
$ pytest test_recipes.py -q -s --vcr-record=none -k ledger
 {'tokens': 1200, 'cost_usd': 0.012, 'time_ms': -0.2, 'percent': 100.0}
```

An `AgentRun` also has `.recording` (a `VCRPlayer` of the fresh run, or of the golden on a
replay), `.golden_path`, `.identity`, `.mode`, `.tokens` and `.cost_usd`.

### Just record, no golden

`vcr_recorder` is a started `VCRRecorder`. The session is named after the test node id and saved
at teardown, even when the test fails, to `<vcr-dir>/sessions/<test-slug>.vcr`. Re-running moves
the previous file to `.vcr.prev`.

```python
def test_plain_recording(vcr_recorder):
    vcr_recorder.record_step("plan", {"goal": "refund"}, {"steps": ["lookup", "refund"]})
    vcr_recorder.record_checkpoint({"plan": "ok"}, label="planned")
```

### Keep goldens next to your tests

```bash
pytest test_recipes.py -q --vcr-dir tests/goldens -k billing
```

A relative `--vcr-dir` is resolved against the rootdir (where your pytest config lives), so
goldens are found wherever you run pytest from. To set it once:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--vcr-dir tests/goldens"
```

### Run it in CI

Commit `.vcr/pytest/golden/` (and keep `sessions/` out unless you want those recordings), then:

```yaml
      - name: Agent tests (replay only, zero tokens)
        run: AGENT_VCR_RECORD=none pytest -q
```

It works under pytest-xdist (`-n auto`): each test owns its golden folder, so workers never
write the same file.

## What it can't do

- **It doesn't intercept HTTP or your model client.** Your agent must record its steps on the
  recorder it's given (`record_llm_call`, `record_tool_call`, `record_step`, ...), or run as a
  step pipeline. A call it doesn't record isn't in the golden, so its cost, tokens and tools are
  invisible to the checks.
- **On replay your agent doesn't run at all.** Side effects it would have (files, messages,
  database writes) don't happen. The test checks the recorded outputs.
- **Replay can't tell your code changed.** Only the task and identity select a golden. Put a
  model, prompt hash or code commit into `identity`, or re-record.
- **Drift is checked only on live runs.** A replay compares the golden with itself, so only your
  expectations can fail it. Catching drift needs `--vcr-record=all` (or a new identity), which
  calls the model and costs real tokens.
- **An agent that records nothing and returns `None` fails the test.** There's nothing to replay.
- **If pytest-vcr is installed, it owns `--vcr-record`.** Use `--agent-vcr-record`,
  `$AGENT_VCR_RECORD` or the `agent_vcr_record` ini option. A `--vcr-record` that went to
  pytest-vcr triggers an `AgentVCRWarning` and a line in the summary. `none` is still honoured
  (it can only prevent live calls); any other value is ignored. `--vcr-dir` and `--vcr-fail-on`
  are unaffected.
- **Identity and pipeline input must have a stable JSON form:** strings, numbers, lists, dicts
  and sets (sets are sorted). Anything else, such as an object or a `Path`, raises `TypeError`;
  pass a string or hash of it instead.
- **No summary line under xdist.** Workers don't print the `agent-vcr:` line. Recording and
  replay still work.
- **Recordings hold what your agent saw:** prompts, tool arguments, outputs. Keep secrets out
  of them before you commit goldens.

## Cheat sheet

| Do this | With |
|---|---|
| Run or replay an agent | `agent_vcr.run(task, agent, identity=None)` → `AgentRun`, where `agent(recorder) -> result` |
| Run a step pipeline | `agent_vcr.run(task, [(name, step), ...], input={...})`, where `step(state) -> state` or `(state, usage)` |
| Gate a run | `agent_vcr.assert_matches_golden(run=None, fail_on=None, expect=None, **Expectation fields)` |
| Per-test defaults | `@pytest.mark.agent_vcr(task=, identity=, record="none"\|"once"\|"all")`, then `agent_vcr.run(agent)` |
| Plain recording | `vcr_recorder` fixture: a started `VCRRecorder`, saved at teardown |
| Record mode | `--vcr-record=none\|once\|all` (default `once`; alias `--agent-vcr-record`), `$AGENT_VCR_RECORD`, ini `agent_vcr_record` |
| What fails a test | `--vcr-fail-on=degraded\|failed\|never` (default `degraded`) |
| Where goldens live | `--vcr-dir DIR` (default `.vcr/pytest`): `golden/<test>/`, `sessions/` |
| Inspect a result | `run.output`, `run.outputs`, `run.replayed`, `run.ledger`, `run.recording`, `run.golden_path`, `run.tokens`, `run.cost_usd` |
| Accept new behavior | `pytest --vcr-record=all --vcr-fail-on=never <test id>` |

**Next:** [Agent CI](agent-ci.md) (the same rules as a dataset gate) ·
[Recording & Ghost Replay](recording-and-replay.md) (what a golden run is) ·
[ACID workspace](acid-workspace.md) (roll back files your agent touched in a test)
