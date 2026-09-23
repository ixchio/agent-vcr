"""Shared test fixtures for Agent VCR tests."""

import asyncio
import tempfile
from collections.abc import Generator

import pytest

from agent_vcr.models import FrameMetadata
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


@pytest.fixture(autouse=True)
def main_thread_event_loop() -> Generator[None, None, None]:
    """Provide a fresh default loop for synchronous ASGI tests on Python 3.9."""
    created_loop = None
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        created_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(created_loop)

    yield

    if created_loop is not None:
        try:
            if asyncio.get_event_loop() is created_loop:
                asyncio.set_event_loop(None)
        except RuntimeError:
            pass
        created_loop.close()


@pytest.fixture
def tmp_vcr_dir():
    """Create a temporary directory for VCR files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_recorder(tmp_vcr_dir):
    """Create a recorder with a started session."""
    recorder = VCRRecorder(output_dir=tmp_vcr_dir, auto_save=False)
    recorder.start_session(session_id="test-session")
    return recorder


@pytest.fixture
def sample_vcr_file(tmp_vcr_dir):
    """Create a VCR file with 3 recorded frames."""
    recorder = VCRRecorder(output_dir=tmp_vcr_dir, auto_save=False)
    recorder.start_session(session_id="test-session")

    recorder.record_step(
        node_name="node1",
        input_state={"step": 0, "value": 0},
        output_state={"step": 1, "value": 10},
        metadata=FrameMetadata(latency_ms=100),
    )
    recorder.record_step(
        node_name="node2",
        input_state={"step": 1, "value": 10},
        output_state={"step": 2, "value": 20},
        metadata=FrameMetadata(latency_ms=200),
    )
    recorder.record_step(
        node_name="node3",
        input_state={"step": 2, "value": 20},
        output_state={"step": 3, "value": 30},
        metadata=FrameMetadata(latency_ms=300),
    )

    return recorder.save()


@pytest.fixture
def sample_player(sample_vcr_file):
    """Load a player from the sample VCR file."""
    return VCRPlayer.load(sample_vcr_file)


@pytest.fixture
def error_vcr_file(tmp_vcr_dir):
    """Create a VCR file with an error frame."""
    recorder = VCRRecorder(output_dir=tmp_vcr_dir, auto_save=False)
    recorder.start_session(session_id="error-session")

    recorder.record_step("ok_node", {}, {"status": "ok"})
    recorder.record_error(
        "error_node", {}, ValueError("test error"), latency_ms=50
    )

    return recorder.save()
