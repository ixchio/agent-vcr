"""Unit tests for VCR models."""

from datetime import datetime, timezone

from agent_vcr.models import (
    Frame,
    FrameMetadata,
    FrameType,
    ResumeConfig,
    ResumeMode,
    Session,
    StateSerializer,
)


class TestFrame:
    def test_frame_creation(self):
        frame = Frame(
            session_id="test-session",
            node_name="test_node",
            input_state={"key": "value"},
            output_state={"result": "success"},
        )

        assert frame.session_id == "test-session"
        assert frame.node_name == "test_node"
        assert frame.frame_type == FrameType.NODE_EXECUTION
        assert frame.frame_id is not None
        assert frame.timestamp is not None

    def test_frame_serialization(self):
        frame = Frame(
            session_id="test-session",
            node_name="test_node",
            input_state={"key": "value"},
            output_state={"result": "success"},
            metadata=FrameMetadata(latency_ms=100, tokens_used=50),
        )

        data = frame.model_dump()
        assert data["session_id"] == "test-session"
        assert data["node_name"] == "test_node"
        assert data["metadata"]["latency_ms"] == 100
        assert data["metadata"]["tokens_used"] == 50


class TestSession:
    def test_session_creation(self):
        session = Session(
            session_id="test-session",
            tags=["test", "debug"],
        )

        assert session.session_id == "test-session"
        assert session.tags == ["test", "debug"]
        assert session.frame_count == 0
        assert session.created_at is not None

    def test_session_fork(self):
        parent = Session(session_id="parent")
        child = Session(
            session_id="child",
            parent_session_id=parent.session_id,
            forked_from_frame=5,
        )

        assert child.parent_session_id == parent.session_id
        assert child.forked_from_frame == 5


class TestStateSerializer:
    def test_serialize_primitive(self):
        assert StateSerializer.serialize("hello") == "hello"
        assert StateSerializer.serialize(42) == 42
        assert StateSerializer.serialize(None) is None

    def test_serialize_dict(self):
        data = {"key": "value", "number": 42}
        result = StateSerializer.serialize(data)
        assert result == data

    def test_serialize_list(self):
        data = [1, 2, 3, "hello"]
        result = StateSerializer.serialize(data)
        assert result == data

    def test_serialize_nested(self):
        data = {"outer": {"inner": [1, 2, 3]}}
        result = StateSerializer.serialize(data)
        assert result == data

    def test_serialize_datetime(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = StateSerializer.serialize(dt)
        assert result["__type__"] == "datetime"
        assert result["data"] == dt.isoformat()

    def test_roundtrip(self):
        original = {
            "string": "hello",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }

        serialized = StateSerializer.serialize(original)
        deserialized = StateSerializer.deserialize(serialized)

        assert deserialized == original


class TestResumeConfig:
    def test_default_config(self):
        config = ResumeConfig(from_frame=5)

        assert config.from_frame == 5
        assert config.mode == ResumeMode.FORK
        assert config.state_overrides == {}
        assert config.skip_nodes == []

    def test_custom_config(self):
        config = ResumeConfig(
            from_frame=10,
            mode=ResumeMode.MOCK,
            state_overrides={"key": "value"},
            skip_nodes=["node1", "node2"],
        )

        assert config.from_frame == 10
        assert config.mode == ResumeMode.MOCK
        assert config.state_overrides == {"key": "value"}
        assert config.skip_nodes == ["node1", "node2"]
