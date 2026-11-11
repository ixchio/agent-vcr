"""Regression tests for medium-severity core fixes (recorder / player / async)."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_vcr.async_player import AsyncVCRPlayer
from agent_vcr.async_recorder import AsyncVCRRecorder
from agent_vcr.models import FrameType, ResumeConfig, ResumeMode
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


def test_player_keeps_file_order_and_reports_clock_skew(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path))
    rec.start_session("skew")
    first = rec.record_step("first", {}, {"i": 0})
    second = rec.record_step("second", {}, {"i": 1})
    # Simulate a clock step backwards between the two frames.
    second.timestamp = first.timestamp - timedelta(seconds=5)
    rec.save()

    player = VCRPlayer.load(tmp_path / "skew.vcr")
    assert [f.node_name for f in player.frames] == ["first", "second"]
    report = player.verify_integrity()
    assert not report["ok"] and any("backwards" in issue for issue in report["issues"])


def test_goto_time_accepts_z_suffix(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path))
    rec.start_session("z")
    rec.record_step("a", {}, {"v": 1})
    rec.save()
    player = VCRPlayer.load(tmp_path / "z.vcr")
    stamp = player.frames[0].timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    assert player.goto_time(stamp) == {"v": 1}


def test_replay_mode_keeps_frame_types_and_writes_next_to_source(tmp_path: Path) -> None:
    src_dir = tmp_path / "recordings"
    rec = VCRRecorder(output_dir=str(src_dir))
    rec.start_session("orig")
    rec.record_step("plan", {"q": 1}, {"plan": "p"})
    rec.record_llm_call("m", [{"role": "user", "content": "x"}], {"t": "y"}, 3, 4, 1.0)
    rec.record_step("act", {"t": "y"}, {"done": True})
    rec.save()

    player = VCRPlayer.load(src_dir / "orig.vcr")
    new_id = player.resume(
        lambda state: {"resumed": True},
        ResumeConfig(from_frame=2, mode=ResumeMode.REPLAY, new_session_id="branch"),
    )
    branch = VCRPlayer.load(src_dir / f"{new_id}.vcr")  # same directory as the source
    assert [f.frame_type for f in branch.frames[:2]] == [
        FrameType.NODE_EXECUTION, FrameType.LLM_CALL,
    ]
    assert branch.frames[1].metadata.custom["replayed_from_frame_id"] == player.frames[1].frame_id
    assert player.frames[1].metadata.custom == {}  # source frame metadata untouched


def test_fork_in_place_keeps_recorder_identity(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path))
    rec.start_session("main")
    rec.record_step("a", {}, {"n": 1})
    rec.record_step("b", {}, {"n": 2})
    same = rec.fork(from_frame=0, in_place=True)
    assert same is rec
    rec.record_step("c", {}, {"n": 3})
    session = rec.get_session()
    assert session is not None and session.parent_session_id == "main"
    assert [f.node_name for f in rec.get_frames()] == ["fork:a", "c"]
    assert len(VCRPlayer.load(tmp_path / "main.vcr").frames) == 2  # parent was saved


def test_async_save_fixes_header_totals(tmp_path: Path) -> None:
    async def run() -> None:
        rec = AsyncVCRRecorder(output_dir=tmp_path)
        await rec.start_session("hdr")
        await rec.record_llm_call("m", [], {"t": 1}, tokens_input=5, tokens_output=7)
        await rec.record_step("x", {}, {"y": 1})
        await rec.save()

    asyncio.run(run())
    header = json.loads((tmp_path / "hdr.vcr").read_text().splitlines()[0])["data"]
    assert header["frame_count"] == 2 and header["total_tokens"] == 12
    player = asyncio.run(AsyncVCRPlayer.load(tmp_path / "hdr.vcr"))
    assert player.verify_integrity()["ok"]


def test_async_callback_can_record_without_deadlock(tmp_path: Path) -> None:
    async def run() -> int:
        rec: AsyncVCRRecorder

        async def on_frame(frame) -> None:  # noqa: ANN001
            if frame.node_name == "outer":
                await rec.record_step("inner", {}, {})

        rec = AsyncVCRRecorder(output_dir=tmp_path, on_frame_recorded=on_frame)
        await rec.start_session()
        await asyncio.wait_for(rec.record_step("outer", {}, {}), timeout=5)
        return len(rec.get_frames())

    assert asyncio.run(run()) == 2


def _record_many(args: tuple[str, int]) -> None:
    out_dir, worker = args
    for i in range(10):
        rec = VCRRecorder(output_dir=out_dir)
        rec.start_session(f"w{worker}-{i}")
        rec.record_step("s", {}, {"i": i})
        rec.save()


def test_manifest_survives_parallel_processes(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_record_many, [(str(tmp_path), w) for w in range(4)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    ids = {s["session_id"] for s in manifest["sessions"]}
    assert len(ids) == 40


def test_manifest_update_keeps_entry_position(tmp_path: Path) -> None:
    rec_a = VCRRecorder(output_dir=str(tmp_path))
    rec_a.start_session("a")
    rec_a.save()
    rec_b = VCRRecorder(output_dir=str(tmp_path))
    rec_b.start_session("b")
    rec_b.save()
    rec_a.record_step("later", {}, {})
    rec_a.save()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [s["session_id"] for s in manifest["sessions"]] == ["a", "b"]
    assert manifest["sessions"][0]["frame_count"] == 1
    assert datetime.fromisoformat(manifest["updated_at"])
