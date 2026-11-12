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

> Needs the OpenHands SDK 1.x: `pip install openhands-sdk openhands-tools` (verified against
> 1.7.3). `agent` and `ws` are yours.

```python
from openhands.sdk import Conversation

from openhands_sentinel import Sentinel, SentinelConfig

sentinel = Sentinel(SentinelConfig(block_on_blocker=True))
conversation = Conversation(agent=agent, workspace=ws, callbacks=[sentinel])
sentinel.bind(conversation)          # lets Sentinel reject actions

conversation.send_message("Add a login endpoint")
conversation.run()
sentinel.deliver_warnings()          # hand non-blocking findings to the agent...
conversation.run()                   # ...and let it fix them
print(sentinel.save())               # .vcr/sentinel-openhands.vcr
```

Once it is a callback, Sentinel:

- analyzes the file after every successful `file_editor` `create`, `str_replace`, `insert` and
  `undo_edit`, using the observation's `new_content` (or the file on disk). `view` and failed
  edits are skipped;
- on a blocker **this edit introduced** (with `block_on_blocker=True`, the default), rejects
  the agent's **next** action through the SDK's own mechanism, `state.block_action`. The agent
  gets a `UserRejectObservation` that reads "Action rejected: Sentinel rejected this action: your
  edit of ... introduced blocker issues ..." with the findings. Tool calls from the same model
  response that have not run yet are rejected too;
- compares with the file **before** the edit (the observation's `old_content`, or the file read
  just before the edit ran). A blocker that was already there — say a 260-line legacy function
  next to the line the agent changed — is reported as a warning ("already had blocker-level
  issues"), never a rejection;
- keeps rejecting until the file no longer has blockers the edit added. Allowed meanwhile:
  `file_editor` on the broken file (any command, so the agent can view and fix it) and `think`.
  A clean re-analysis lifts the gate and counts as a self-correction;
- never wedges the agent. Before each rejection it re-reads blocked files from disk and drops
  any that were fixed or deleted some other way (you, `git checkout`, `vcr undo`). A new user
  message (`conversation.send_message(...)`) clears the gate. After
  `max_rejections_per_file` rejections (default 3) for one file it lets actions through, logs a
  warning, and queues a "still has blocker issues" note for the agent;
- queues non-blocking warnings in `sentinel.pending_warnings`. `deliver_warnings()` sends them as
  one user message — call it **between** `run()` calls, never from a callback (a user message
  in the middle of a batch of tool calls breaks the order the model expects);
- records each analysis (`sentinel:analysis`), rejection (`sentinel:reject`) and gate release
  (`sentinel:release`) as a VCR frame. `stats.rejected_actions` counts real vetoes; `stats.blocked_actions` counts blocker
  results.

You decide a blocker is fine? `sentinel.blocked_files` shows the gate (path → reason);
`sentinel.unblock("app/x.py")` releases one file and `sentinel.unblock()` all of them.

```python
print(sentinel.blocked_files)   # {'/ws/app/x.py': 'Sentinel rejected this action: ...'}
sentinel.unblock()              # ['/ws/app/x.py'] — the agent can carry on
```

Forgot `bind()`? Sentinel still analyzes, logs one warning, and rejects nothing.
`auto_analyze=False` turns the callback off. To record every tool call too, add a
[`VCRRuntime`](integrations/README.md#openhands) to `callbacks`; pass both the same `VCRRecorder`
for a single `.vcr` file.

**Legacy V0 (`openhands-ai`, `pip install "ai-agent-vcr[openhands-legacy]"`):**
`Sentinel().attach(runtime.event_stream)` subscribes to the event stream, records every event,
analyzes each `FileWriteAction` / `FileEditAction` (patch-style edits when their
`FileEditObservation` arrives) and injects warnings back as a user `MessageAction`. V0 cannot
veto an action: "blocked" there is a label and a strong message.

Try the whole flow without OpenHands: `python examples/sentinel_demo.py` from a repo checkout
simulates an agent session.

## What it can't do

- **Python only.** Other files are ignored.
- **It reviews, it doesn't undo.** A bad write has already happened. To roll it back, use the
  [airbag](airbag.md) or [ACIDWorkspace](acid-workspace.md).
- **It can't stop the write that broke the file.** With the OpenHands SDK it rejects the *next*
  action; the bad write itself already happened (the SDK has no "after write, before commit"
  hook). Shell commands that edit files (`sed -i`, `cat >`) are not analyzed, and a fix made from
  the terminal is rejected too — the agent must use `file_editor`. Fixes made outside the agent
  are picked up only when the file is on this machine (not in a remote workspace). Elsewhere
  (V0, `scan`, `check_file`) "blocked" is a label, not a veto.
- **Structure, not correctness.** No type checking, no tests, no security scan. Run your linter
  and test suite too.
- **Duplicate memory is per session.** A fresh `scan` only compares files within that scan.
- **OpenHands SDK: local conversations only.** Verified with a local `Conversation`; a
  `RemoteConversation` against an agent server is untested.

## Cheat sheet

| Command / API | What it does |
| --- | --- |
| `sentinel scan [DIR]` | One-shot scan (default `.`) and report |
| `sentinel watch [DIR] [--interval S] [--once]` | Rescan changed files continuously |
| `--max-lines` / `--max-complexity` / `--max-file-lines` / `--max-params` | Thresholds (50 / 10 / 500 / 7) |
| `Sentinel(config, recorder, on_violation)` | Python entry point |
| `.check_file(path, content)` → `AnalysisResult` | Analyze and record |
| `.check_and_warn(path, content)` → `(result, warning \| None)` | Plus agent-facing text |
| `Conversation(callbacks=[sentinel])` + `.bind(conversation)` | Hook into OpenHands SDK 1.x (rejects after blockers) |
| `.deliver_warnings()` | Send queued warnings between `run()` calls |
| `.blocked_files` / `.unblock([path])` | See / release the OpenHands gate |
| `.check_file(path, content, previous_content=old)` | Only blockers new since `old` mark it `blocked` |
| `.attach(event_stream)` | Hook into legacy OpenHands V0 |
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
