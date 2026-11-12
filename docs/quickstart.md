# Agent VCR in 5 minutes

Six superpowers, one snippet each. Everything below runs offline: no API key, no account.

**TL;DR**

- `pip install ai-agent-vcr` gives you the `vcr` and `sentinel` commands and the Python API.
- Coding-agent user? Section 1 is all you need. Building your own agent? Sections 2–4.
- Every section links to the one page that covers it in full.

## Install

```bash
pipx install ai-agent-vcr     # the `vcr` command, isolated (Python 3.9–3.13)
pip install ai-agent-vcr      # …or into your environment, for the Python API below
vcr --version                 # vcr 1.7.0
```

Work in an empty folder for the Python snippets below: they write a `.vcr/` directory next to
where you run them.

## 1. 🛟 Undo your coding agent

```bash
cd your-project                # any git repository
vcr airbag check "rm -rf src"
# 🛟 DESTRUCTIVE: destroys project data — snapshot first, undo with `vcr undo`
#    • rm deletes files

bash examples/airbag_demo.sh   # from a clone of the repo: an agent wipes a project, `vcr undo` brings it back
```

Then arm it once for your user, not per project:

```bash
vcr airbag install   # every coding agent on your PATH, user level: nothing lands in your repos
vcr doctor           # everything armed? prints the exact fix if not
```

From then on, in every git repository, every tool call your agent makes (shell commands
included) gets a snapshot first, so `vcr undo` can bring files, `.env` and even `.git` back.
Outside a repository nothing is snapshotted, so destructive commands are stopped for you to
confirm (`mkdir .vcr` opts a plain folder in). Do the "Required once" steps it prints (Codex:
`/hooks`; Gemini: `/trust`). → [Airbag](airbag.md)

## 2. ⏪ Record and time-travel a run

```python
from agent_vcr import VCRPlayer, VCRRecorder

with VCRRecorder() as rec:
    rec.start_session(session_id="hello")
    rec.record_step("plan", {"task": "refund"}, {"plan": ["look up", "refund"]})
    rec.record_step("act", {"plan": ["look up", "refund"]}, {"status": "refunded"})

player = VCRPlayer.load(".vcr/hello.vcr")
print(player.list_nodes())        # ['plan', 'act']
print(player.goto_frame(0))       # {'plan': ['look up', 'refund']}
```

Every step lands in an append-only `.vcr` file you can jump around in, frame by frame.
→ [Recording and replay](recording-and-replay.md)

## 3. 🔁 Roll back files *and* state in your own agent

```python
import pathlib, subprocess, tempfile
from agent_vcr.integrations.openhands import ACIDWorkspace

repo = pathlib.Path(tempfile.mkdtemp())
subprocess.run(["git", "init", "-q", str(repo)], check=True)
(repo / "app.py").write_text("print('v1')\n")
subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
subprocess.run(["git", "-C", str(repo), "-c", "user.name=me", "-c", "user.email=me@x",
                "commit", "-qm", "v1"], check=True)

ws = ACIDWorkspace(str(repo))
ws.begin(session_id="demo")
ws.savepoint({"step": "before agent"}, node_name="start")   # frame 0
(repo / "app.py").write_text("raise SystemExit('agent broke it')\n")
ws.savepoint({"step": "agent edit"}, node_name="edit")      # frame 1
ws.rollback(to_frame_index=0)
print((repo / "app.py").read_text())                        # print('v1')
ws.abort()                                                  # back to where you started
```

Each savepoint is a git commit plus a state frame, so a rollback rewinds the files on disk
together with the agent's memory. → [ACID workspace](acid-workspace.md)

## 4. 👻 Ghost Replay a verified run

```python
from agent_vcr import GoldenRunCache, VCRRecorder

rec = VCRRecorder()
rec.start_session(session_id="good-run")
rec.record_llm_call("gpt-4o", [{"role": "user", "content": "refund #42"}],
                    {"content": "Refunded."}, tokens_input=900, tokens_output=100,
                    latency_ms=1800, cost_usd=0.012)
rec.save()

cache = GoldenRunCache()
cache.save_golden_run("refund order 42", rec)          # pin the verified run
outputs, ledger = cache.replay("refund order 42")      # zero model calls
print(ledger.tokens_saved, ledger.cost_saved_usd)      # 1000 0.012
```

Save a run you trust once, then replay it from cache as often as you like, at zero tokens.
→ [Recording and replay](recording-and-replay.md)

## 5. ✅ Gate it in CI

```bash
vcr dataset create refunds
vcr dataset add refunds --task "refund order 42" --case-name refund-42 \
  --vcr .vcr/hello.vcr --expect-state '{"status":"refunded"}'

# your test harness writes a fresh recording per case: .vcr/candidates/refund-42.vcr
vcr ci run --dataset refunds --candidates .vcr/candidates --fail-on degraded
# Dataset `refunds` (v2) · 1 cases: **1 passed**, 0 degraded, 0 failed
```

Pin reviewed runs as golden cases, and CI fails the build when a new recording drifts on
state, cost, latency or tools, without calling an LLM. → [Agent CI](agent-ci.md)

## 6. 🔍 Sentinel scan

```bash
sentinel scan src/
# example output, for a file with an 8-argument function:
#   ✗ x.py
#      WARNING  Line 1: `f()` has 8 parameters (max 7). Use a config object.
#   PASSED  No critical issues in 1 file(s).
```

Sentinel checks Python files for oversized, overly complex or duplicated functions, and saves
an audit trail to `src/.vcr/sentinel-scan.vcr`. → [Sentinel](sentinel.md)

## What it can't do (in 5 minutes)

- The airbag only protects project folders (git repositories, or folders with `.vcr/`);
  remote databases, cloud resources and pushes are stopped for a human, not undone. Details in
  [Airbag](airbag.md#what-it-cant-do).
- Ghost Replay replays what was recorded. It doesn't make a new task free; change the task or
  its identity and you need a fresh run.
- `vcr ci` compares recordings. It doesn't run your agent; your harness produces candidates.

## Where to next

| You are… | Read this |
|---|---|
| **Using a coding agent** (Claude Code, Codex, Cursor, …) | [Airbag](airbag.md): install, undo, the gotchas per agent |
| **Building an agent** (LangGraph, CrewAI, plain Python) | [Recording and replay](recording-and-replay.md) → [ACID workspace](acid-workspace.md) → [Integrations](integrations/README.md) |
| **Owning CI** | [Agent CI](agent-ci.md) → [Sentinel](sentinel.md) |

Want the big picture? [Architecture](architecture.md) · [FAQ](faq.md) · [All docs](README.md)
