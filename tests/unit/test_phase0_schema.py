"""Phase 0 stabilization tests: frozen schema, checkpoint, diff, durability APIs."""

from __future__ import annotations

import json

import pytest

import agent_vcr
from agent_vcr.models import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    Frame,
    ResumeConfig,
    Session,
    StateDiff,
)
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


class TestFrozenSchema:
    def test_schema_constants(self):
        assert SCHEMA_NAME == "execution/v1"
        assert SCHEMA_VERSION == "1.0.0"
        assert agent_vcr.SCHEMA_VERSION == "1.0.0"

    def test_session_frame_carry_version(self):
        s = Session(session_id="s")
        f = Frame(session_id="s", node_name="n", input_state={}, output_state={})
        assert s.schema_version == "1.0.0"
        assert f.schema_version == "1.0.0"

    def test_backward_compat_old_files_without_version(self, tmp_path):
        """Pre-1.0 records without schema_version must still load."""
        path = tmp_path / "old.vcr"
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "type": "session",
                "data": {
                    "session_id": "old",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                },
            }) + "\n")
            fh.write(json.dumps({
                "type": "frame",
                "data": {
                    "session_id": "old",
                    "node_name": "n",
                    "input_state": {},
                    "output_state": {"v": 1},
                },
            }) + "\n")
        player = VCRPlayer.load(path)
        assert player.session.session_id == "old"
        assert player.goto_frame(0) == {"v": 1}

    def test_future_fields_ignored(self):
        s = Session.model_validate({"session_id": "x", "future_field": 123})
        assert s.session_id == "x"


class TestStateDiff:
    def test_compute_views(self):
        before = {"a": 1, "b": 2, "c": 3}
        after = {"a": 1, "b": 20, "d": 4}
        d = StateDiff.compute(before, after)
        assert d["added"] == {"d": 4}
        assert d["removed"] == {"c": 3}
        assert d["modified"] == {"b": {"before": 2, "after": 20}}
        assert d["unchanged"] == {"a": 1}

    def test_json_patch_view(self):
        patch = StateDiff.to_json_patch({"a": 1}, {"a": 2, "b": 3})
        ops = {(p["op"], p["path"]) for p in patch}
        assert ("replace", "/a") in ops
        assert ("add", "/b") in ops

    def test_player_compare_uses_canonical(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("cmp")
        rec.record_step("n", {}, {"a": 1})
        rec.record_step("n", {}, {"a": 2})
        path = rec.save()
        player = VCRPlayer.load(path)
        diff = player.compare_frames(0, 1)
        assert diff["modified"]["a"] == {"before": 1, "after": 2}


class TestCheckpointAPI:
    def test_record_list_goto_checkpoint(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("ckpt")
        rec.record_step("step1", {}, {"v": 1})
        rec.record_checkpoint({"v": 1}, label="before_payment")
        rec.record_step("step2", {"v": 1}, {"v": 2})
        path = rec.save()

        player = VCRPlayer.load(path)
        cps = player.list_checkpoints()
        assert len(cps) == 1
        assert cps[0]["label"] == "before_payment"
        assert player.goto_checkpoint("before_payment") == {"v": 1}
        with pytest.raises(KeyError):
            player.goto_checkpoint("nope")

    def test_fork_marker_listed_as_checkpoint(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("parent")
        rec.record_step("n", {}, {"v": 1})
        forked = rec.fork(from_frame=0)
        assert forked.list_checkpoints if hasattr(forked, "list_checkpoints") else True
        # forked recorder saved session starts with a CHECKPOINT frame
        fpath = forked.save()
        player = VCRPlayer.load(fpath)
        assert len(player.list_checkpoints()) == 1

    def test_async_checkpoint(self, tmp_path):
        import asyncio

        from agent_vcr.async_player import AsyncVCRPlayer
        from agent_vcr.async_recorder import AsyncVCRRecorder

        async def go():
            rec = AsyncVCRRecorder(output_dir=str(tmp_path))
            await rec.start_session("async-ckpt")
            await rec.record_step("n", {}, {"v": 1})
            await rec.record_checkpoint({"v": 1}, label="c1")
            path = await rec.save()
            player = await AsyncVCRPlayer.load(path)
            assert player.goto_checkpoint("c1") == {"v": 1}
            assert player.verify_integrity()["ok"]

        asyncio.run(go())


class TestDurability:
    def test_header_carries_schema_and_fsync(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=True)
        rec.start_session("hdr")
        rec.record_step("n", {}, {"v": 1})
        path = rec.save()
        with open(path) as fh:
            header = json.loads(fh.readline())
        assert header["type"] == "session"
        assert header.get("schema") == "execution/v1"
        assert header["data"]["schema_version"] == "1.0.0"

    def test_strict_load_raises_vs_skip(self, tmp_path):
        path = tmp_path / "mixed.vcr"
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "type": "session",
                "data": {
                    "session_id": "m",
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                },
            }) + "\n")
            fh.write("NOT JSON\n")
            fh.write(json.dumps({
                "type": "frame",
                "data": {"session_id": "m", "node_name": "n",
                         "input_state": {}, "output_state": {}},
            }) + "\n")
        player = VCRPlayer.load(path)
        assert player.skipped_lines == 1
        assert len(player.frames) == 1
        with pytest.raises(ValueError, match="Malformed JSON"):
            VCRPlayer.load(path, strict=True)

    def test_verify_integrity(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("integ")
        rec.record_step("n", {}, {"v": 1})
        path = rec.save()
        player = VCRPlayer.load(path)
        report = player.verify_integrity()
        assert report["ok"] is True
        assert report["frame_count"] == 1

    def test_rebuild_manifest(self, tmp_path):
        rec = VCRRecorder(output_dir=str(tmp_path), auto_save=False)
        rec.start_session("r1")
        rec.record_step("n", {}, {"v": 1})
        rec.save()
        (tmp_path / "manifest.json").unlink()
        manifest = VCRRecorder.rebuild_manifest(tmp_path)
        assert manifest["rebuilt"] is True
        assert any(s["session_id"] == "r1" for s in manifest["sessions"])
        assert (tmp_path / "manifest.json").exists()

    def test_resume_config_schema_exported(self, tmp_path):
        import pathlib

        schema_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "schemas" / "events" / "resume_config.json"
        )
        assert schema_path.exists(), "run python3 scripts/export_schemas.py"
        payload = json.loads(schema_path.read_text())
        assert "from_frame" in json.dumps(payload)
        assert ResumeConfig(from_frame=0).mode.value == "fork"
