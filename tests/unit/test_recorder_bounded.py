"""Bounded-memory recording: ``max_cached_frames`` evicts written frames from RAM.

Evicted frames are read back from the ``.vcr`` file by get_frames / fork / golden cache,
and the header rewrite on save streams from disk instead of the in-memory cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

import agent_vcr.recorder as recorder_module
from agent_vcr.async_recorder import AsyncVCRRecorder
from agent_vcr.golden_cache import GoldenRunCache
from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder

N_FRAMES = 20_000
CACHE = 50
BUFFER = 100


def _record(rec: VCRRecorder, i: int) -> None:
    rec.record_step(f"step_{i}", {"i": i}, {"i": i, "out": i * 2},
                    metadata=FrameMetadata(tokens_used=1, latency_ms=1.0))


def test_20k_frames_with_a_small_cache_stay_bounded_and_correct(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=BUFFER, max_cached_frames=CACHE)
    rec.start_session("long-run")

    tracemalloc.start()
    try:
        peak_cache = 0
        baseline = 0
        for i in range(N_FRAMES):
            _record(rec, i)
            peak_cache = max(peak_cache, rec.cached_frame_count)
            if i == 2_000:
                baseline = tracemalloc.get_traced_memory()[0]
        grown = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()

    # The cache never holds more than the bound plus one unflushed buffer ...
    assert peak_cache <= CACHE + BUFFER
    # ... so 18k more frames add (almost) nothing. Unbounded, they cost ~20 MB.
    assert grown < 1_000_000, f"memory grew by {grown} bytes"
    assert rec.get_session() is not None
    assert rec.get_session().frame_count == N_FRAMES  # type: ignore[union-attr]

    path = rec.save()
    player = VCRPlayer.load(path)
    assert len(player.frames) == N_FRAMES
    assert [f.node_name for f in player.frames[:3]] == ["step_0", "step_1", "step_2"]
    assert player.frames[-1].output_state == {"i": N_FRAMES - 1, "out": 2 * (N_FRAMES - 1)}
    assert player.verify_integrity()["ok"]
    assert player.session.frame_count == N_FRAMES
    assert player.session.total_tokens == N_FRAMES
    assert rec.cached_frame_count <= CACHE

    # Evicted frames come back from disk, in order, identical to what the player loads.
    frames = rec.get_frames()
    assert [f.frame_id for f in frames] == [f.frame_id for f in player.frames]


def test_fork_and_golden_cache_read_evicted_frames(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path / "vcr"), buffer_size=10, max_cached_frames=5)
    rec.start_session("src")
    for i in range(200):
        _record(rec, i)
    assert rec.cached_frame_count <= 15

    fork = rec.fork(3, new_session_id="forked", state_overrides={"patched": True})
    fork_frame = fork.get_frames()[0]
    assert fork_frame.output_state == {"i": 3, "out": 6, "patched": True}
    assert fork.max_cached_frames == 5
    with pytest.raises(ValueError):
        rec.fork(200)

    cache = GoldenRunCache(cache_dir=str(tmp_path / "golden"))
    key = cache.save_golden_run("task", rec)
    golden = VCRPlayer.load(tmp_path / "golden" / f"{key}.vcr")
    assert [f.node_name for f in golden.frames] == [f"step_{i}" for i in range(200)]


def test_save_rewrites_the_header_from_disk_not_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_cache_rewrite(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("header rewrite must stream from disk")

    monkeypatch.setattr(recorder_module, "rewrite_session_file", _no_cache_rewrite)
    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=7, max_cached_frames=3)
    rec.start_session("hdr")
    for i in range(50):
        _record(rec, i)
    path = rec.save()
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "session"
    assert header["data"]["frame_count"] == 50
    assert sum(json.loads(line)["type"] == "frame" for line in lines) == 50
    assert VCRPlayer.load(path).verify_integrity()["ok"]


def test_unbounded_default_keeps_everything_and_repairs_a_truncated_file(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=10)
    assert rec.max_cached_frames is None
    rec.start_session("default")
    for i in range(25):
        _record(rec, i)
    assert rec.cached_frame_count == 25
    path = tmp_path / "default.vcr"
    # Lose flushed frames underneath the recorder: with every frame still cached, the
    # header rewrite falls back to the cache instead of keeping the damaged file.
    path.write_text("\n".join(path.read_text(
        encoding="utf-8").splitlines()[:6]) + "\n", encoding="utf-8")
    rec.save()
    player = VCRPlayer.load(path)
    assert [f.node_name for f in player.frames] == [f"step_{i}" for i in range(25)]
    assert player.verify_integrity()["ok"]


def test_attach_session_keeps_only_the_tail(tmp_path: Path) -> None:
    first = VCRRecorder(output_dir=str(tmp_path), buffer_size=10)
    first.start_session("resume")
    for i in range(40):
        _record(first, i)
    first.save()

    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=10, max_cached_frames=8)
    rec.attach_session("resume")
    assert rec.cached_frame_count == 8
    for i in range(40, 60):
        _record(rec, i)
    path = rec.save()
    assert [f.node_name for f in rec.get_frames()] == [f"step_{i}" for i in range(60)]
    player = VCRPlayer.load(path)
    assert len(player.frames) == 60
    assert player.verify_integrity()["ok"]


def test_a_new_session_drops_the_previous_sessions_frames(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path))
    rec.start_session("one")
    for i in range(5):
        _record(rec, i)
    rec.start_session("two")
    assert rec._cache.get_frames("one") == []
    assert rec.cached_frame_count == 0


def test_negative_bound_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        VCRRecorder(output_dir=str(tmp_path), max_cached_frames=-1)
    with pytest.raises(ValueError):
        AsyncVCRRecorder(output_dir=str(tmp_path), max_cached_frames=-1)


@pytest.mark.asyncio
async def test_async_recorder_bounded(tmp_path: Path) -> None:
    rec = AsyncVCRRecorder(output_dir=tmp_path, buffer_size=BUFFER, max_cached_frames=CACHE)
    await rec.start_session("async-long")
    peak = 0
    for i in range(3_000):
        await rec.record_step(f"step_{i}", {"i": i}, {"i": i})
        peak = max(peak, rec.cached_frame_count)
    assert peak <= CACHE + BUFFER

    path = await rec.save()
    player = VCRPlayer.load(path)
    assert len(player.frames) == 3_000
    assert player.verify_integrity()["ok"]
    loaded = await rec.load_frames()
    assert [f.frame_id for f in loaded] == [f.frame_id for f in player.frames]
    assert [f.frame_id for f in rec.get_frames()] == [f.frame_id for f in player.frames]

    fork = await rec.fork(7, new_session_id="async-fork")
    assert fork.get_frames()[0].output_state == {"i": 7}
    assert fork.max_cached_frames == CACHE


# -- review fixes --------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_async_save_keeps_the_lock_until_the_rewrite_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled save() used to release the lock while its rewrite thread ran on; the
    thread's os.replace then dropped frames appended meanwhile (unrecoverable when bounded)."""
    rewriting = threading.Event()
    real_fsync = os.fsync

    def slow_fsync(fd: int) -> None:
        real_fsync(fd)
        if sys._getframe(1).f_code.co_name == "_stream_rewrite":
            rewriting.set()
            time.sleep(0.4)  # a slow disk, between reading the file and os.replace

    monkeypatch.setattr(os, "fsync", slow_fsync)
    rec = AsyncVCRRecorder(output_dir=tmp_path, buffer_size=10, max_cached_frames=5)
    await rec.start_session("cancel")
    for i in range(100):
        await rec.record_step(f"s{i}", {}, {"i": i})
    await rec.save()
    rewriting.clear()
    for i in range(5):
        await rec.record_step(f"t{i}", {}, {"i": i})

    save = asyncio.create_task(rec.save())
    assert await asyncio.to_thread(rewriting.wait, 10)
    save.cancel()
    with pytest.raises(asyncio.CancelledError):
        await save
    for i in range(20):  # two auto-flushes; they must not race the old rewrite
        await rec.record_step(f"u{i}", {}, {"i": i})
    await asyncio.sleep(0.6)  # (before the fix: let the orphaned rewrite's os.replace land)
    player = VCRPlayer.load(await rec.save())
    assert len(player.frames) == 125
    assert player.session.frame_count == 125
    assert player.verify_integrity()["ok"]


@pytest.mark.asyncio
async def test_cancelled_async_flush_does_not_append_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    syncing = threading.Event()
    real = AsyncVCRRecorder._fsync_path

    def slow(path: Path) -> None:
        syncing.set()
        time.sleep(0.3)
        real(path)

    # Bounded, so the header rewrite cannot quietly repair the file from the cache.
    rec = AsyncVCRRecorder(output_dir=tmp_path, buffer_size=5, max_cached_frames=2)
    await rec.start_session("flush")
    for i in range(4):
        await rec.record_step(f"s{i}", {}, {"i": i})
    monkeypatch.setattr(AsyncVCRRecorder, "_fsync_path", staticmethod(slow))
    step = asyncio.create_task(rec.record_step("s4", {}, {"i": 4}))  # 5th frame flushes
    assert await asyncio.to_thread(syncing.wait, 10)
    step.cancel()
    with pytest.raises(asyncio.CancelledError):
        await step
    monkeypatch.setattr(AsyncVCRRecorder, "_fsync_path", staticmethod(real))
    player = VCRPlayer.load(await rec.save())
    assert [f.node_name for f in player.frames] == [f"s{i}" for i in range(5)]


def test_bounded_get_frames_returns_a_snapshot(tmp_path: Path) -> None:
    rec = VCRRecorder(output_dir=str(tmp_path), buffer_size=5, max_cached_frames=3)
    rec.start_session("snap")
    for i in range(4):
        _record(rec, i)
    before = rec.get_frames()
    ids = [f.frame_id for f in before]
    for i in range(4, 30):
        _record(rec, i)  # evicts in place
    assert [f.frame_id for f in before] == ids


@pytest.mark.parametrize("cls", ["sync", "async"])
def test_cached_and_evicted_frames_look_the_same(tmp_path: Path, cls: str) -> None:
    """JSON object keys are strings on disk; bounded recorders store cached frames that way
    too, so get_frames() never mixes ``{1: ..}`` (cache) with ``{"1": ..}`` (disk)."""
    async def _run() -> list[Any]:
        rec = AsyncVCRRecorder(output_dir=tmp_path, buffer_size=4, max_cached_frames=3)
        await rec.start_session("keys")
        for i in range(12):
            await rec.record_step("n", {i: "in"}, {i: ("out", i)},
                                  metadata=FrameMetadata(custom={"k": {i: "c"}}))
        await rec.save()
        return rec.get_frames()

    if cls == "async":
        frames = asyncio.run(_run())
    else:
        sync = VCRRecorder(output_dir=str(tmp_path), buffer_size=4, max_cached_frames=3)
        sync.start_session("keys")
        for i in range(12):
            sync.record_step("n", {i: "in"}, {i: ("out", i)},
                             metadata=FrameMetadata(custom={"k": {i: "c"}}))
        sync.save()
        frames = sync.get_frames()
    disk = VCRPlayer.load(tmp_path / "keys.vcr").frames
    assert [f.model_dump() for f in frames] == [f.model_dump() for f in disk]
    assert frames[-1].output_state == {"11": ["out", 11]}
    assert frames[-1].metadata.custom == {"k": {"11": "c"}}
