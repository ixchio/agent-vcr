# Recording, time travel, and Ghost Replay

Agent VCR stores an agent run as append-friendly JSONL: one session header followed by ordered
frames. A frame can represent a node, model call, tool call, error, or checkpoint and can carry
input/output state, latency, token use, cost, and caller-defined metadata.

## Record a run

```python
from agent_vcr import VCRPlayer, VCRRecorder
from agent_vcr.models import FrameMetadata

recorder = VCRRecorder(output_dir=".vcr")
recorder.start_session("support-triage", tags=["support", "v1"])

before = {"ticket_id": "t_123", "status": "new"}
after = {"ticket_id": "t_123", "status": "classified", "priority": "high"}
recorder.record_step(
    node_name="classify",
    input_state=before,
    output_state=after,
    metadata=FrameMetadata(model="provider/model", tokens_used=312, cost_usd=0.0018),
)
recorder.record_checkpoint(after, label="classification-approved")
recording = recorder.save()

player = VCRPlayer.load(recording, strict=True)
print(player.verify_integrity())
```

Use `record_llm_call`, `record_tool_call`, and `record_error` for typed frames and `record_step` as
the generic framework boundary. `diff_mode=True` can store a state delta when the input duplicates
the prior output; the player reconstructs it during navigation.

## Inspect, compare, and resume

```python
from agent_vcr import VCRPlayer

player = VCRPlayer.load(".vcr/support-triage.vcr")
print(player.list_nodes())
print(player.list_checkpoints())
print(player.get_errors())
print(player.goto_frame(0))
print(player.goto_checkpoint("classification-approved"))
print(player.compare_frames(0, 1))
```

Load with `strict=True` when malformed lines should be fatal. The default reader isolates malformed
or truncated lines and reports them in `player.skipped_lines` so a valid prefix can still be read.

```bash
pip install "ai-agent-vcr[tui]"
vcr tui .vcr/support-triage.vcr
```

`VCRPlayer.resume()` creates a child session and invokes your callable from a selected boundary. It
does not restore files; use [`ACIDWorkspace`](acid-workspace.md) for filesystem rollback.

```python
from agent_vcr import VCRPlayer
from agent_vcr.models import ResumeConfig, ResumeMode


def continue_agent(state: dict) -> dict:
    return {**state, "status": "resolved"}


player = VCRPlayer.load(".vcr/support-triage.vcr")
child_id = player.resume(
    continue_agent,
    ResumeConfig(
        from_frame=0,
        mode=ResumeMode.FORK,
        state_overrides={"priority": "urgent"},
    ),
)
print(child_id)
```

`FORK` runs fresh from the boundary, `REPLAY` copies prior frames then continues, and `MOCK` adds
caller-provided values under `__mocks__`. None of these modes reverses external side effects.

## Ghost Replay

Ghost Replay promotes a reviewed run into a local golden cache. Full replay returns recorded outputs
without invoking model or tool code. Use an identity covering every input that should invalidate
reuse:

```python
from agent_vcr.golden_cache import GoldenRunCache, GhostReplayMode

cache = GoldenRunCache(".vcr/golden")
identity = cache.build_identity(
    model="provider/model@revision",
    prompt_hash="sha256:prompt-content",
    code_commit="0123456789abcdef",
    tool_schema_hash="sha256:tool-schema",
    dependency_lock_hash="sha256:lockfile",
    environment="staging",
)
cache.save_golden_run("classify support ticket", recorder, identity=identity)
outputs, ledger = cache.replay_with_mode(
    "classify support ticket",
    GhostReplayMode.FULL,
    identity=identity,
)
print(outputs, ledger.summary())
```

Identity JSON is normalized recursively with sorted keys before SHA-256 fingerprinting, so
dictionary key order in nested tool schemas does not create false cache misses. Real content drift
still changes the key. If identity is omitted, only the normalized task string is matched; that is
convenient for demos but too weak for most production reuse.

- `FULL` reuses every output and makes no model/tool calls.
- `HYBRID` reruns changed frames and all downstream frames.
- `LIVE` reruns only selected frames; use it only when downstream outputs are independent.
- `BRANCH` replays history before a fork point, then reruns the fork and everything after it.

“Zero-cost” refers to skipped provider execution on a full hit. Local CPU/disk overhead still
exists, and partial modes can incur provider cost.

## Portable bundles and privacy

```bash
vcr bundle export .vcr/support-triage.vcr --output support-triage.vcrb
vcr bundle verify support-triage.vcrb
vcr reproduce support-triage.vcrb
```

A bundle includes the recording and a SHA-256 manifest with execution identity and summary
metadata. Integrity proves byte consistency, not semantic correctness.

Recordings may contain prompts, state, model responses, tool arguments/results, and errors. Redact
secrets before recording, restrict `.vcr` and CI artifact access, use synthetic test fixtures, and
review a run before promoting it to a golden baseline. ACID escrow bytes are separate and never
serialized into recordings, but that does not redact secrets explicitly present in recorded state.
