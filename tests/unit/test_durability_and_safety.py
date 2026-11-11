"""Recorder durability and security boundaries.

Recorder data loss (diff mode, truncation on reused ids, unserializable state),
lazy imports, the Claude Code recorder hook, bundle path traversal, sandbox
secret leakage, fail-closed approvals, Qdrant search and Ghost Replay paths.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from agent_vcr.bundle import export_bundle, import_bundle
from agent_vcr.cli import CLAUDE_CODE_EVENTS, write_claude_code_hooks
from agent_vcr.hooks import claude_code
from agent_vcr.models import FrameMetadata, StateSerializer
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder
from agent_vcr.sandbox import LocalSandbox, SandboxConfig
from agent_vcr.superagent.approval import ApprovalRequest, AutoApprover

# ── Recorder ─────────────────────────────────────────────────────────────


def test_diff_mode_keeps_inputs_that_differ_from_previous_output(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), diff_mode=True)
    rec.start_session("diff")
    rec.record_step("plan", {"q": "hi"}, {"plan": "x"})
    rec.record_llm_call("gpt", [{"role": "user", "content": "PROMPT"}], {"text": "ok"}, 1, 1, 5.0)
    rec.record_step("next", {"text": "ok", "extra": 1}, {"done": True})
    rec.save()

    player = VCRPlayer.load(tmp_path / "diff.vcr")
    assert player.get_input_state(1) == {"messages": [{"role": "user", "content": "PROMPT"}]}
    assert player.get_input_state(2) == {"text": "ok", "extra": 1}


def test_diff_mode_still_elides_duplicate_inputs(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), diff_mode=True)
    rec.start_session("dup")
    rec.record_step("a", {"v": 0}, {"v": 1})
    frame = rec.record_step("b", {"v": 1}, {"v": 2})
    assert frame.input_state == {}  # stored compactly...
    rec.save()
    assert VCRPlayer.load(tmp_path / "dup.vcr").get_input_state(1) == {"v": 1}  # ...but restorable


def test_restarting_same_session_id_preserves_previous_recording(tmp_path: Path) -> None:
    first = VCRRecorder(output_dir=str(tmp_path))
    first.start_session("run")
    first.record_step("a", {}, {"n": 1})
    first.save()

    second = VCRRecorder(output_dir=str(tmp_path))
    second.start_session("run")
    backup = tmp_path / "run.vcr.prev"
    assert backup.exists()
    assert len(VCRPlayer.load(backup).frames) == 1

    with pytest.raises(FileExistsError):
        VCRRecorder(output_dir=str(tmp_path)).start_session("run", if_exists="error")


def test_session_id_cannot_escape_output_dir(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        rec.start_session("../escape")


def test_unserializable_state_never_crashes_or_duplicates(tmp_path: Path) -> None:
    class Opaque:
        pass

    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=2)
    rec.start_session("weird")
    rec.record_step("a", {}, {"ok": 1})
    rec.record_step("b", {}, {"tags": {"x", "y"}, "blob": b"\x00\x01", "obj": Opaque()})
    rec.record_step("c", {(1, 2): "tuple-key"}, {"p": Path("/tmp/x")})
    rec.save()

    player = VCRPlayer.load(tmp_path / "weird.vcr", strict=True)
    assert [f.node_name for f in player.frames] == ["a", "b", "c"]  # no duplicates
    out = player.get_output_state(1)
    assert out["tags"] == {"x", "y"}
    assert out["blob"] == b"\x00\x01"


def test_record_step_does_not_mutate_caller_metadata(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False, diff_mode=True)
    rec.start_session()
    meta = FrameMetadata()
    rec.record_step("a", {}, {"x": 1}, metadata=meta)
    rec.record_step("b", {"y": 2}, {"x": 2}, metadata=meta)
    assert meta.latency_ms == 0.0 and meta.custom == {}


def test_serializer_round_trips_dataclass_and_set() -> None:
    from dataclasses import dataclass

    @dataclass
    class Point:
        x: int
        y: int

    data = StateSerializer.serialize({"p": Point(1, 2), "s": frozenset({3})})
    json.dumps(data)  # JSON-safe
    back = StateSerializer.deserialize(data)
    assert back == {"p": {"x": 1, "y": 2}, "s": {3}}


# ── Import cost ──────────────────────────────────────────────────────────


def test_import_agent_vcr_is_lazy() -> None:
    code = (
        "import sys, agent_vcr, agent_vcr.hooks.claude_code;"
        "heavy=[m for m in ('qdrant_client','agent_vcr.superagent','agent_vcr.workflows',"
        "'agent_vcr.cloud','agent_vcr.sandbox') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_lazy_exports_resolve() -> None:
    import agent_vcr

    for name in agent_vcr.__all__:
        assert getattr(agent_vcr, name) is not None


# ── Claude Code hook ─────────────────────────────────────────────────────


def _real_payload(session: str = "abc-123", event: str = "PostToolUse") -> dict:
    return {
        "session_id": session,
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/repo",
        "hook_event_name": event,
        "tool_name": "Write",
        "tool_input": {"file_path": "/repo/src/app.py", "content": "x" * 50_000},
        "tool_response": {"success": True},
    }


def test_hook_uses_real_claude_payload_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_VCR_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_VCR_CLAUDE_SESSION_ID", raising=False)
    path = claude_code.record_hook_event("Unknown", _real_payload())
    claude_code.record_hook_event("Unknown", _real_payload(event="PreToolUse"))
    claude_code.record_hook_event("Unknown", _real_payload(session="other"))

    assert path.name == "claude-abc-123.vcr"
    player = VCRPlayer.load(path, strict=True)
    assert [f.node_name for f in player.frames] == ["claude:PostToolUse", "claude:PreToolUse"]
    summary = player.frames[0].output_state
    assert summary["file_path"] == "/repo/src/app.py"
    content = player.frames[0].input_state["payload"]["tool_input"]["content"]
    assert len(content) < 10_000 and "truncated" in content
    assert (tmp_path / "claude-other.vcr").exists()


def test_hook_main_never_fails_and_keeps_raw_input(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AGENT_VCR_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_VCR_CLAUDE_SESSION_ID", "fixed")
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json {"))
    assert claude_code.main(["Stop"]) == 0
    frame = VCRPlayer.load(tmp_path / "fixed.vcr").frames[0]
    assert frame.input_state["payload"]["raw"] == "not json {"

    monkeypatch.setenv("AGENT_VCR_DIR", str(tmp_path / "file-not-dir"))
    (tmp_path / "file-not-dir").write_text("blocks mkdir")
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    assert claude_code.main(["Stop"]) == 0  # recording failed, hook still succeeds


def test_init_uses_current_interpreter_and_upgrades_legacy_hooks(tmp_path: Path) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    legacy = "python -m agent_vcr.hooks.claude_code PreToolUse"
    settings.write_text(json.dumps({"hooks": {"PreToolUse": [
        {"matcher": "*", "hooks": [{"type": "command", "command": legacy}]},
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "my-own-hook"}]},
    ]}}))

    write_claude_code_hooks(tmp_path)
    data = json.loads(settings.read_text())
    commands = [h["command"] for e in data["hooks"]["PreToolUse"] for h in e["hooks"]]
    assert legacy not in commands
    assert "my-own-hook" in commands  # user hooks untouched
    ours = [c for c in commands if "agent_vcr.hooks.claude_code" in c]
    assert len(ours) == 1 and sys.executable in ours[0]
    for event in CLAUDE_CODE_EVENTS:
        assert event in data["hooks"]


# ── Bundle import ────────────────────────────────────────────────────────


def _tamper_session_id(bundle: Path, new_id: str) -> None:
    with tarfile.open(bundle, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}  # type: ignore[union-attr]
    manifest = json.loads(members["bundle.json"])
    manifest["session_id"] = new_id
    members["bundle.json"] = json.dumps(manifest).encode()
    with tarfile.open(bundle, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("evil", ["../escaped", "/tmp/abs-escape", "a/../../b"])
def test_bundle_import_rejects_path_traversal(tmp_path: Path, evil: str) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path / "src"))
    rec.start_session("ok")
    rec.record_step("a", {}, {"x": 1})
    vcr_path = rec.save()
    bundle = export_bundle(vcr_path, tmp_path / "b.vcrb")
    _tamper_session_id(bundle, evil)

    dest = tmp_path / "dest"
    with pytest.raises(ValueError, match="unsafe|escapes"):
        import_bundle(bundle, dest)
    assert not (tmp_path / "escaped.vcr").exists()


# ── Sandbox / approvals ──────────────────────────────────────────────────


def test_local_sandbox_does_not_leak_host_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_OPENAI_API_KEY", "sk-secret")
    with LocalSandbox(workdir=tmp_path / "sb") as sb:
        res = sb.execute_command('echo "key=${FAKE_OPENAI_API_KEY}" home=$HOME')
        assert res.success
        assert "sk-secret" not in res.stdout
        assert str(tmp_path / "sb") in res.stdout
    with LocalSandbox(config=SandboxConfig(inherit_env=True), workdir=tmp_path / "sb2") as sb:
        assert "sk-secret" in sb.execute_command("echo $FAKE_OPENAI_API_KEY").stdout


def test_default_auto_approver_fails_closed() -> None:
    req = ApprovalRequest("s", "pay", "tool", "irreversible")
    assert not AutoApprover().decide(req).approved
    assert AutoApprover(allow_all=True, max_auto_cost_usd=1.0).decide(
        ApprovalRequest("s", "pay", "tool", "r", cost_estimate_usd=0.5)
    ).approved
    # allow_all still refuses unknown cost unless the ref is named explicitly
    assert not AutoApprover(allow_all=True).decide(req).approved
    assert AutoApprover(allow_refs=["pay"]).decide(req).approved


def test_qdrant_search_uses_supported_api() -> None:
    qdrant_client = pytest.importorskip("qdrant_client")
    from agent_vcr.superagent.vector_memory import QdrantVectorStore

    store = QdrantVectorStore(
        collection_name="t", client=qdrant_client.QdrantClient(location=":memory:")
    )
    store.add("refund the duplicate payment", namespace="ops")
    store.add("unrelated weather report", namespace="ops")
    hits = store.search("refund duplicate payment", limit=1, namespace="ops")
    assert hits and "refund" in hits[0].text


# ── Ghost Replay / golden cache ──────────────────────────────────────────


def _golden_run(cache_dir: Path, vcr_dir: Path) -> tuple:
    from agent_vcr.golden_cache import GoldenRunCache
    from agent_vcr.models import FrameType

    cache = GoldenRunCache(str(cache_dir))
    rec = VCRRecorder(output_dir=str(vcr_dir))
    rec.start_session()
    rec.record_step("plan", {"q": 1}, {"plan": "p"})
    rec.record_llm_call("m", [{"role": "user", "content": "x"}], {"t": "y"}, 0, 0, 1.0)
    cache.save_golden_run("refund task", rec)
    return cache, FrameType


def test_golden_cache_works_from_any_working_directory(tmp_path: Path, monkeypatch) -> None:
    from agent_vcr.golden_cache import GoldenRunCache

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    _golden_run(Path(".vcr/golden"), Path(".vcr"))

    monkeypatch.chdir(tmp_path)  # e.g. CI running from the repo root's parent
    cache = GoldenRunCache(str(project / ".vcr" / "golden"))
    outputs, ledger = cache.replay("refund task", recorder=VCRRecorder(str(tmp_path / "out")))
    assert len(outputs) == 2 and ledger.steps_replayed == 2
    assert Path(cache.get_golden_info("refund task")["file"]).is_absolute()


def test_golden_cache_reads_legacy_relative_index_entries(tmp_path: Path) -> None:
    from agent_vcr.golden_cache import GoldenRunCache

    cache_dir = tmp_path / "golden"
    _golden_run(cache_dir, tmp_path / "vcr")
    index_path = cache_dir / "index.json"
    index = json.loads(index_path.read_text())
    for info in index.values():
        info["file"] = ".vcr/golden/" + info["file"]  # how <=1.5.2 stored it
    index_path.write_text(json.dumps(index))

    outputs, _ = GoldenRunCache(str(cache_dir)).replay(
        "refund task", recorder=VCRRecorder(str(tmp_path / "out"))
    )
    assert len(outputs) == 2


def test_replay_preserves_frame_types_and_honest_savings(tmp_path: Path) -> None:
    cache, frame_type = _golden_run(tmp_path / "golden", tmp_path / "vcr")
    rec = VCRRecorder(str(tmp_path / "out"))
    _, ledger = cache.replay(
        "refund task",
        step_executor=lambda node, state: {"rerun": node},
        changed_steps={0},
        recorder=rec,
    )
    assert [f.frame_type for f in rec.get_frames()] == [
        frame_type.NODE_EXECUTION, frame_type.LLM_CALL,
    ]
    # Golden run had no cost/token data and every step re-ran: 0% saved, not 100%.
    assert ledger.steps_rerun == 2
    assert ledger.savings_percent == 0.0
    assert ledger.summary()["rerun_cost_estimated"] is True
