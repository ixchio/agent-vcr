"""Unit tests for VCR Server API."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder
from agent_vcr.server import VCRServer


@pytest.fixture
def vcr_server():
    """Create a VCR server with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test recording
        recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
        recorder.start_session(session_id="test-session")

        recorder.record_step(
            node_name="step_1",
            input_state={"a": 1},
            output_state={"b": 2},
            metadata=FrameMetadata(latency_ms=100, tokens_used=50),
        )
        recorder.record_step(
            node_name="step_2",
            input_state={"b": 2},
            output_state={"c": 3},
            metadata=FrameMetadata(latency_ms=200, tokens_used=100),
        )
        recorder.save()

        server = VCRServer(vcr_dir=tmpdir)
        yield server, tmpdir


@pytest.fixture
def client(vcr_server):
    """Create a test client."""
    server, _ = vcr_server
    return TestClient(server.app, raise_server_exceptions=False)


class TestRootEndpoint:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Agent VCR API"
        assert "endpoints" in data


class TestSessionsEndpoint:
    def test_list_sessions(self, client):
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_get_session_detail(self, client):
        response = client.get("/api/sessions/test-session")
        assert response.status_code == 200
        data = response.json()
        assert data["session"]["session_id"] == "test-session"
        assert len(data["frames"]) == 2
        assert "statistics" in data
        assert data["statistics"]["total_frames"] == 2

    def test_get_session_not_found(self, client):
        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    def test_session_statistics(self, client):
        response = client.get("/api/sessions/test-session")
        data = response.json()
        stats = data["statistics"]
        assert stats["total_frames"] == 2
        assert stats["total_tokens"] == 150
        assert stats["nodes"] == ["step_1", "step_2"]


class TestFramesEndpoint:
    def test_get_frame(self, client):
        response = client.get("/api/sessions/test-session/frames/0")
        assert response.status_code == 200
        data = response.json()
        assert "frame" in data
        assert "input_state" in data
        assert "output_state" in data

    def test_get_frame_not_found(self, client):
        response = client.get("/api/sessions/test-session/frames/999")
        assert response.status_code == 404

    def test_get_frame_session_not_found(self, client):
        response = client.get("/api/sessions/nonexistent/frames/0")
        assert response.status_code == 404


class TestResumeEndpoint:
    def test_resume_session(self, client):
        response = client.post(
            "/api/sessions/test-session/resume",
            json={"from_frame": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["from_frame"] == 0
        assert "state" in data

    def test_resume_with_state_overrides(self, client):
        response = client.post(
            "/api/sessions/test-session/resume",
            json={"from_frame": 0, "state_overrides": {"new_key": "new_value"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state"]["new_key"] == "new_value"

    def test_resume_session_not_found(self, client):
        response = client.post(
            "/api/sessions/nonexistent/resume",
            json={"from_frame": 0},
        )
        assert response.status_code == 404


class TestExportEndpoint:
    def test_export_json(self, client):
        response = client.get("/api/sessions/test-session/export?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "session" in data
        assert "frames" in data
        assert "statistics" in data

    def test_export_mermaid(self, client):
        response = client.get("/api/sessions/test-session/export?format=mermaid")
        assert response.status_code == 200
        data = response.json()
        assert "mermaid" in data
        assert "graph TD" in data["mermaid"]

    def test_export_unsupported_format(self, client):
        response = client.get("/api/sessions/test-session/export?format=csv")
        assert response.status_code == 500  # Wrapped in generic exception handler

    def test_export_session_not_found(self, client):
        response = client.get("/api/sessions/nonexistent/export?format=json")
        assert response.status_code == 404


class TestWebSocket:
    def test_websocket_connect(self, client):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"action": "ping"})
            response = ws.receive_json()
            assert response["type"] == "pong"

    def test_websocket_subscribe(self, client):
        with client.websocket_connect("/ws/live") as ws:
            ws.send_json({"action": "subscribe", "session_id": "test-session"})
            response = ws.receive_json()
            assert response["type"] == "subscribed"
            assert response["session_id"] == "test-session"
