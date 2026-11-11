# Sentinel — a code-quality guard that watches your agent write 🛡️

Sentinel reads every Python file your agent writes, catches copy-paste, monster functions and
broken syntax with the AST, and hands the agent a "fix this first" note. It records every check to
`.vcr`.

**TL;DR**

- `sentinel scan ./src` checks a folder and exits `0`/`1`/`2` — easy to drop into CI or an agent
  loop.
- It has **memory across files**: the same function pasted into a second file is flagged, which a
  one-file linter never sees.
- Pure stdlib `ast`, no extra dependencies, no network, no LLM.

## Try it in 60 seconds

```bash
pip install ai-agent-vcr
mkdir -p demo/app && cd demo

cat > app/auth.py <<'EOF'
def check_password(user, password):
    return user.password == password
EOF

cat > app/billing.py <<'EOF'
from os.path import *


def check_password(user, password):
    return user.password == password
EOF

sentinel scan app; echo "exit code: $?"
```

You'll see:

```text
  ✓ auth.py
  ✗ billing.py
    CRITICAL  Line 4: `check_password()` already exists in auth.py:1 with an identical body. ...
     WARNING  Line 1: Wildcard import `from os.path import *` — import specific names instead.
  ...
   CRITICAL  1 critical issue(s) found.
exit code: 1
```

Now break the syntax (`echo 'def broken(:' >> app/billing.py`) and scan again: `BLOCKED`, exit
code `2`. Every scan leaves an audit trail in `app/.vcr/sentinel-scan.vcr`.

## How it works

```text
 agent writes file ──▶ Sentinel.check_file(path, content)
                          │
                          ├─ ast.parse ── fails? ──▶ BLOCKER (syntax_error)
                          ├─ size, growth, functions, classes, imports
                          └─ trajectory memory: defs seen in other files
                          ▼
                 AnalysisResult ──▶ warning text for the agent
                          │
                          └──▶ VCR frame "sentinel:analysis" (.vcr audit trail)
```

- Only `.py` files are analyzed. The CLI skips hidden directories, `node_modules`, `__pycache__`,
  `.venv` and `venv`.
- "Passed" means no **critical** and no **blocker** findings. Warnings alone still pass.
- Memory lives as long as the `Sentinel` object: across files in one `scan`, or across edits in
  `watch` / an attached agent session. When a file is rewritten, its old definitions are
  forgotten, so fixing a duplicate clears it.
- When a file that had findings comes back clean, it counts as a **self-correction**.

### What it catches

| Rule | Severity | Fires when (defaults) |
| --- | --- | --- |
| `syntax_error` | 🛑 blocker | the file does not parse |
| `function_too_long` | warning → critical → blocker | over 50 lines → over 2× → over 4× (over 200 lines) |
| `high_complexity` | warning → critical | cyclomatic complexity over 10 → 20 or more |
| `too_many_params` | warning | more than 7 parameters |
| `file_too_large` | warning → critical | over 500 lines → 1000 or more |
| `rapid_growth` | critical | one edit adds 50+ lines **and** grows the file by more than 200% |
| `duplicate_function` | critical / warning | same top-level function in another file: identical body / name only |
| `duplicate_function` (method) | critical | identical `Class.method` body in another file |
| `class_too_large` | warning | more than 15 methods |
| `duplicate_class` | warning | a class with the same name in another file |
| `wildcard_import` | warning | `from x import *` |

## Everyday recipes

### Gate CI or an agent loop on the exit code

```bash
sentinel scan ./src
case $? in
  0) echo "clean" ;;
  1) echo "critical issues — ask the agent to fix them" ;;
  2) echo "blocked — do not continue" ; exit 1 ;;
  3) echo "no Python files found — nothing was checked" ;;
  4) echo "folder missing" ; exit 1 ;;
esac
```

### Tune the thresholds

```bash
sentinel scan . --max-lines 80 --max-complexity 5 --max-file-lines 300 --max-params 4
```

The same four flags work on `watch`. The class-method limit (15) is only configurable from Python
(`SentinelConfig(max_class_methods=...)`).

### Watch a folder while the agent works

```bash
sentinel watch ./src                 # rescans changed files every second; Ctrl+C to stop
sentinel watch ./src --interval 0.5
sentinel watch ./src --once          # one pass, then exit (handy in scripts)
```

`watch` is stricter than `scan`: it exits `2` if it saw a blocker, `1` if it saw **any** other
finding (warnings included, even when the file line says "clean"), `0` if it saw nothing, and `4`
for a missing folder. It saves `sentinel-watch.vcr` on exit.

### Use it from Python

```python
from agent_vcr import VCRRecorder
from openhands_sentinel import Sentinel, SentinelConfig

sentinel = Sentinel(
    config=SentinelConfig(max_function_lines=30, max_complexity=8),
    recorder=VCRRecorder(output_dir=".vcr"),
    on_violation=lambda v: print("caught:", v.rule, v.severity.value),
)
sentinel.start_session("my-agent-quality")

agent_wrote = "def save(user):\n    return db.write(user)\n"
result, warning = sentinel.check_and_warn("app/users.py", agent_wrote)
print("passed:", result.passed)

result, warning = sentinel.check_and_warn("app/admin.py", agent_wrote)  # same code, new file
print("passed:", result.passed, "blocked:", result.blocked)
if warning:
    print(warning)  # feed this back to your agent as its next message

print(sentinel.stats.to_dict())
print(sentinel.save())  # .vcr/my-agent-quality.vcr
```

`check_file()` does the same without building the warning text. `sentinel.get_report()` prints the
session summary that `scan` shows.

### Attach to OpenHands

> Needs `pip install "ai-agent-vcr[openhands]"` (the `openhands-ai` event-stream API).

```python
from openhands_sentinel import Sentinel, SentinelConfig

sentinel = Sentinel(SentinelConfig(block_on_blocker=True)).attach(runtime.event_stream)
# ... run your OpenHands task ...
sentinel.save()
```

Once attached, Sentinel:

- records **every** event on the stream as a VCR frame, and shrinks frames over 50 KB to a summary
  (`max_frame_bytes`, `truncate_large_frames`);
- analyzes each `FileWriteAction` and `FileEditAction`. Patch-style edits are analyzed when their
  `FileEditObservation` arrives, reading the file from disk if needed;
- injects the warning back into the stream as a user `MessageAction`, so the agent reads it as
  feedback (`warn_agent=False` turns that off; `auto_analyze=False` records events only).

With `block_on_blocker=True` (the default), a blocker marks the result `blocked`, counts it in
`stats.blocked_actions`, and starts the warning with "🛑 SENTINEL BLOCKED". The agent is told to
stop. Sentinel does **not** cancel the action — it already ran.

Try the whole flow without OpenHands: `python examples/sentinel_demo.py` from a repo checkout
simulates an agent session.

## What it can't do

- **Python only.** Other files are ignored.
- **It reviews, it doesn't undo.** A bad write has already happened. To roll it back, use the
  [airbag](airbag.md) or [ACIDWorkspace](acid-workspace.md).
- **It doesn't block actions.** "Blocked" is a label and a strong message to the agent, not a
  veto.
- **Structure, not correctness.** No type checking, no tests, no security scan. Run your linter
  and test suite too.
- **Duplicate memory is per session.** A fresh `scan` only compares files within that scan.
- **The OpenHands hook targets the legacy event stream** (`openhands-ai`). For the newer
  `openhands-sdk`, call `check_and_warn()` yourself.

## Cheat sheet

| Command / API | What it does |
| --- | --- |
| `sentinel scan [DIR]` | One-shot scan (default `.`) and report |
| `sentinel watch [DIR] [--interval S] [--once]` | Rescan changed files continuously |
| `--max-lines` / `--max-complexity` / `--max-file-lines` / `--max-params` | Thresholds (50 / 10 / 500 / 7) |
| `Sentinel(config, recorder, on_violation)` | Python entry point |
| `.check_file(path, content)` → `AnalysisResult` | Analyze and record |
| `.check_and_warn(path, content)` → `(result, warning \| None)` | Plus agent-facing text |
| `.attach(event_stream)` | Hook into OpenHands |
| `.stats`, `.get_report()`, `.save()` | Counters, summary, write `.vcr` |

| Exit code | `scan` | `watch` |
| --- | --- | --- |
| `0` | passed (warnings allowed) | no findings at all |
| `1` | critical issue(s) | any non-blocker finding, warnings included |
| `2` | blocker(s) — stop | blocker(s) seen |
| `3` | no Python files found | — |
| `4` | folder missing / not a folder | folder missing / not a folder |

## Next

- [airbag.md](airbag.md) — undo what the agent did, including Bash.
- [integrations/README.md](integrations/README.md) — OpenHands and other frameworks.
- [recording-and-replay.md](recording-and-replay.md) — explore the `.vcr` audit trail.
