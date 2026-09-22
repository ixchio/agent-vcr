"""Core data models for Agent VCR.

Phase 0 frozen schema: ``execution/v1`` (schema version ``1.0.0``).

Backward-compatibility contract:
- Every persisted record carries ``schema_version`` with a default, so
  pre-1.0 ``.vcr`` files (no version field) still load.
- ``model_config(extra="ignore")`` on Session/Frame so future fields
  never break old readers; old files never break new readers.
- Canonical diff helpers live in :class:`StateDiff` — single source of
  truth for recorder (JSON-patch list) and player (added/removed dict).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

SCHEMA_NAME = "execution/v1"
SCHEMA_VERSION = "1.0.0"


class FrameType(str, Enum):
    """Types of frames that can be recorded."""

    NODE_EXECUTION = "node"
    TOOL_CALL = "tool"
    LLM_CALL = "llm"
    ERROR = "error"
    CHECKPOINT = "checkpoint"


class ResumeMode(str, Enum):
    """Modes for resuming execution."""

    FORK = "fork"
    REPLAY = "replay"
    MOCK = "mock"


class FrameMetadata(BaseModel):
    """Metadata associated with a frame."""

    model: str | None = None
    latency_ms: float = 0.0
    tokens_used: int | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    custom: dict[str, Any] = Field(default_factory=dict)


class Frame(BaseModel):
    """A single frame in a VCR session recording."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION)
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    parent_frame_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_type: FrameType = FrameType.NODE_EXECUTION
    node_name: str
    input_state: dict[str, Any]
    output_state: dict[str, Any]
    metadata: FrameMetadata = Field(default_factory=FrameMetadata)
    state_diff: list[dict] | None = None

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()


class Session(BaseModel):
    """A VCR recording session."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default=SCHEMA_VERSION)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_session_id: str | None = None
    forked_from_frame: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    frame_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


class ResumeConfig(BaseModel):
    """Configuration for resuming execution from a frame."""

    from_frame: int
    new_session_id: str | None = None
    state_overrides: dict[str, Any] = Field(default_factory=dict)
    mode: ResumeMode = ResumeMode.FORK
    skip_nodes: list[str] = Field(default_factory=list)
    inject_mocks: dict[str, Any] = Field(default_factory=dict)


class StateDiff:
    """Canonical state-diff helpers (Phase 0 single source of truth).

    Two views of the same diff:
    - :meth:`compute` — ``{"added","removed","modified","unchanged"}`` dict
      used by ``VCRPlayer.compare_frames``.
    - :meth:`to_json_patch` — RFC-6902-style ``add/remove/replace`` list
      stored on ``Frame.state_diff`` in diff mode.
    """

    @staticmethod
    def compute(
        before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        diff: dict[str, dict[str, Any]] = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {},
        }
        for key in set(before) | set(after):
            if key not in before:
                diff["added"][key] = after[key]
            elif key not in after:
                diff["removed"][key] = before[key]
            elif before[key] != after[key]:
                diff["modified"][key] = {"before": before[key], "after": after[key]}
            else:
                diff["unchanged"][key] = before[key]
        return diff

    @staticmethod
    def to_json_patch(
        before: dict[str, Any], after: dict[str, Any]
    ) -> list[dict[str, Any]]:
        patch: list[dict[str, Any]] = []
        for key in set(before) | set(after):
            if key not in before:
                patch.append({"op": "add", "path": f"/{key}", "value": after[key]})
            elif key not in after:
                patch.append({"op": "remove", "path": f"/{key}"})
            elif before[key] != after[key]:
                patch.append({"op": "replace", "path": f"/{key}", "value": after[key]})
        return patch


class StateSerializer:
    """Handles serialization of complex state objects with type preservation."""

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, type_name: str, type_class: type) -> None:
        """Register a type for deserialization."""
        cls._registry[type_name] = type_class

    @classmethod
    def serialize(cls, obj: Any) -> Any:
        """Serialize an object to a JSON-compatible format with type info."""
        if obj is None:
            return None
        if isinstance(obj, BaseModel):
            return {
                "__type__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "data": obj.model_dump(mode="json"),
            }
        if isinstance(obj, list):
            return [cls.serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: cls.serialize(v) for k, v in obj.items()}
        if isinstance(obj, datetime):
            return {"__type__": "datetime", "data": obj.isoformat()}
        if isinstance(obj, Enum):
            return {"__type__": "Enum", "__class__": obj.__class__.__name__, "data": obj.value}
        return obj

    @classmethod
    def deserialize(cls, obj: Any) -> Any:
        """Deserialize an object from the serialized format."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            if "__type__" in obj:
                type_name = obj["__type__"]
                if type_name == "datetime":
                    return datetime.fromisoformat(obj["data"])
                if type_name == "Enum":
                    return obj["data"]
                if type_name in cls._registry:
                    return cls._registry[type_name](**obj["data"])
                return obj["data"]
            return {k: cls.deserialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.deserialize(item) for item in obj]
        return obj


class VCRCache:
    """In-memory cache for VCR sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._frames: dict[str, list[Frame]] = {}

    def add_session(self, session: Session) -> None:
        self._sessions[session.session_id] = session
        self._frames[session.session_id] = []

    def add_frame(self, session_id: str, frame: Frame) -> None:
        if session_id not in self._frames:
            raise ValueError(f"Session {session_id} not found")
        self._frames[session_id].append(frame)

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_frames(self, session_id: str) -> list[Frame]:
        return self._frames.get(session_id, [])
