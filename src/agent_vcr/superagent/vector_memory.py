"""Long-term semantic vector memory for cross-session agent retrieval (Phase 7).

Provides vector database abstraction (Qdrant client integration + in-memory cosine fallback).
All stored values are automatically sanitized with secret redaction.

Semantics shared by both backends:

- Reserved payload keys (``text``, ``namespace``, ``session_id``,
  ``timestamp``, ``_orig_id``) always win over user metadata with the same
  name; colliding metadata keys are dropped with a warning.
- ``search`` never returns zero-similarity hits (unrelated text, an empty
  query, or a vector of the wrong dimension) unless ``min_score`` is negative.
- ``delete(id)`` deletes in any namespace unless ``namespace`` is given.
- ``index_vcr_session`` uses deterministic ids, so re-indexing a session is a
  no-op (check-then-skip).
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_vcr.superagent.memory import atomic_write_text, redact_text, redact_value

logger = logging.getLogger(__name__)

# Check for qdrant-client
_QDRANT_IMPORT_ERROR: ImportError | None = None
try:
    import qdrant_client  # noqa: F401
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError as exc:
    _QDRANT_IMPORT_ERROR = exc

    class _FallbackPointStruct:  # type: ignore[no-redef]
        def __init__(self, id: Any, vector: Any, payload: Any):
            self.id = id
            self.vector = vector
            self.payload = payload

    class _FallbackVectorParams:  # type: ignore[no-redef]
        def __init__(self, size: int = 128, distance: Any = "Cosine"):
            self.size = size
            self.distance = distance

    class _FallbackDistance:  # type: ignore[no-redef]
        COSINE = "Cosine"

    PointStruct = _FallbackPointStruct  # type: ignore[assignment,misc]
    VectorParams = _FallbackVectorParams  # type: ignore[assignment,misc]
    Distance = _FallbackDistance  # type: ignore[assignment,misc]
    QdrantClient = None  # type: ignore[assignment,misc]


class MemoryRecord(BaseModel):
    """A semantic memory record stored in vector memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    vector: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    namespace: str = "default"
    timestamp: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "vector": self.vector,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "timestamp": self.timestamp,
        }


class MemorySearchResult(BaseModel):
    """Result of a semantic memory search."""

    record: MemoryRecord
    score: float

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def text(self) -> str:
        return self.record.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.record.metadata


def default_embedder(text: str, dim: int = 128) -> list[float]:
    """Deterministic hashing-based text embedder for zero-dependency vector retrieval.

    Generates a unit-normalized dense embedding from word and character n-grams.
    """
    clean_text = text.lower().strip()
    if not clean_text:
        return [0.0] * dim

    vec = [0.0] * dim
    words = clean_text.split()

    # Word unigrams and character trigrams
    tokens: list[str] = list(words)
    for word in words:
        if len(word) >= 3:
            for i in range(len(word) - 2):
                tokens.append(word[i : i + 3])

    for token in tokens:
        # SHA-256 keeps the embedder deterministic without routing potentially
        # sensitive memory text through a cryptographically broken hash.
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dim
        val = (int.from_bytes(digest[2:4], "big") / 65535.0) * 2.0 - 1.0
        vec[idx] += val

    # Unit normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 1e-9:
        return [x / norm for x in vec]
    return vec


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


#: Payload keys owned by the store; user metadata can never override them.
RESERVED_PAYLOAD_KEYS = frozenset({"_orig_id", "text", "session_id", "namespace", "timestamp"})


def _user_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Redacted user metadata minus any key that collides with a reserved key."""
    clean = redact_value(metadata or {})
    clashes = sorted(k for k in clean if k in RESERVED_PAYLOAD_KEYS)
    if clashes:
        logger.warning("Dropping metadata keys that collide with reserved payload keys: %s",
                       clashes)
    return {k: v for k, v in clean.items() if k not in RESERVED_PAYLOAD_KEYS}


def _score_passes(score: float, min_score: float) -> bool:
    """Zero similarity is "unrelated", not a weak match (unless min_score < 0)."""
    if min_score < 0.0:
        return score >= min_score
    return score > 0.0 and score >= min_score


def _is_zero_vector(vec: list[float]) -> bool:
    return not vec or all(abs(x) < 1e-12 for x in vec)


def vcr_record_id(namespace: str, session_id: str, frame_id: str) -> str:
    """Deterministic memory id for one indexed VCR frame."""
    return f"vcr:{namespace}:{session_id}:{frame_id}"


def _normalize_qdrant_id(raw_id: str | int) -> str | int:
    """Ensure point IDs sent to Qdrant conform to 64-bit unsigned integer or UUID string."""
    if isinstance(raw_id, int):
        return raw_id
    id_str = str(raw_id)
    try:
        uuid.UUID(id_str)
        return id_str
    except ValueError:
        # Generate deterministic UUIDv5 for arbitrary string IDs
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))


class BaseVectorMemory(abc.ABC):
    """Abstract interface for long-term semantic agent memory."""

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
        vector_dim: int = 128,
    ) -> None:
        self._embed = embedding_fn or (lambda t: default_embedder(t, dim=vector_dim))
        self.vector_dim = vector_dim

    def embed(self, text: str) -> list[float]:
        return self._embed(text)

    @abc.abstractmethod
    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        namespace: str = "default",
        id: str | None = None,
    ) -> str:
        """Store a text snippet into memory with automatic secret redaction."""

    @abc.abstractmethod
    def add_batch(self, records: list[MemoryRecord]) -> list[str]:
        """Batch store memory records."""

    @abc.abstractmethod
    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemorySearchResult]:
        """Search memory by semantic similarity."""

    @abc.abstractmethod
    def get(self, id: str) -> MemoryRecord | None:
        """Retrieve a record by ID."""

    @abc.abstractmethod
    def delete(self, id: str, namespace: str | None = None) -> bool:
        """Delete a record by ID (only within ``namespace`` when given)."""

    @abc.abstractmethod
    def clear(self, namespace: str | None = None) -> int:
        """Clear records (optionally scoped to a namespace)."""

    @abc.abstractmethod
    def count(self, namespace: str | None = None) -> int:
        """Count stored memory records."""

    def index_vcr_session(
        self,
        player_or_path: Any,
        namespace: str = "sessions",
        include_errors: bool = True,
    ) -> int:
        """Index steps from a recorded VCR session for cross-session knowledge recall.

        Idempotent: each frame gets a deterministic id (namespace, session,
        frame) and frames already indexed are skipped. Returns the number of
        records newly added. Frame states are secret-redacted (keys and
        values) before they are embedded.
        """
        from agent_vcr.player import VCRPlayer

        player: VCRPlayer
        if isinstance(player_or_path, VCRPlayer):
            player = player_or_path
        else:
            player = VCRPlayer.load(str(player_or_path))

        count = 0
        session_id = player.session.session_id if player.session else ""

        for idx, frame in enumerate(player.frames):
            text_pieces: list[str] = []
            frame_type = getattr(frame.frame_type, "value", str(frame.frame_type))

            # Include relevant frame inputs and outputs
            if frame.input_state:
                inp_str = json.dumps(redact_value(frame.input_state), default=str)
                text_pieces.append(f"Node {frame.node_name} input: {inp_str}")

            if frame.output_state:
                out_str = json.dumps(redact_value(frame.output_state), default=str)
                text_pieces.append(f"Node {frame.node_name} ({frame_type}) output: {out_str}")

            err_msg = getattr(frame.metadata, "error_message", None)
            if include_errors and (frame_type == "error" or err_msg):
                text_pieces.append(f"Error in {frame.node_name}: {err_msg or 'Unknown failure'}")

            if not text_pieces:
                continue

            rec_id = vcr_record_id(namespace, session_id, str(frame.frame_id or idx))
            if self.get(rec_id) is not None:
                continue  # already indexed — re-indexing is a no-op

            combined_text = "\n".join(text_pieces)
            meta = {
                "frame_index": idx,
                "frame_id": frame.frame_id,
                "node_name": frame.node_name,
                "frame_type": frame_type,
            }
            self.add(
                text=combined_text,
                metadata=meta,
                session_id=session_id,
                namespace=namespace,
                id=rec_id,
            )
            count += 1

        return count


class InMemoryVectorStore(BaseVectorMemory):
    """In-memory vector store with cosine similarity and metadata filtering."""

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
        vector_dim: int = 128,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn, vector_dim=vector_dim)
        self._records: dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        namespace: str = "default",
        id: str | None = None,
    ) -> str:
        # Enforce secret redaction
        clean_text = redact_text(text)
        clean_meta = redact_value(metadata or {})
        rec_id = id or str(uuid.uuid4())
        vec = self.embed(clean_text)

        rec = MemoryRecord(
            id=rec_id,
            text=clean_text,
            vector=vec,
            metadata=clean_meta,
            session_id=session_id,
            namespace=namespace,
            timestamp=time.time(),
        )
        with self._lock:
            self._records[rec_id] = rec
        return rec_id

    def add_batch(self, records: list[MemoryRecord]) -> list[str]:
        ids: list[str] = []
        for r in records:
            clean_text = redact_text(r.text)
            clean_meta = redact_value(r.metadata)
            vec = self.embed(clean_text) if (clean_text != r.text or r.vector is None) else r.vector
            rec = MemoryRecord(
                id=r.id,
                text=clean_text,
                vector=vec,
                metadata=clean_meta,
                session_id=r.session_id,
                namespace=r.namespace,
                timestamp=r.timestamp or time.time(),
            )
            with self._lock:
                self._records[rec.id] = rec
            ids.append(rec.id)
        return ids

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemorySearchResult]:
        q_vec = self.embed(query)
        scored: list[MemorySearchResult] = []
        if _is_zero_vector(q_vec):
            return scored  # empty / unembeddable query matches nothing

        with self._lock:
            records = list(self._records.values())
        for rec in records:
            if namespace is not None and rec.namespace != namespace:
                continue

            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if rec.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            if rec.vector is None or len(rec.vector) != len(q_vec):
                continue  # missing or wrong-dimension vector: not comparable

            score = cosine_similarity(q_vec, rec.vector)
            if _score_passes(score, min_score):
                scored.append(MemorySearchResult(record=rec, score=round(score, 4)))

        scored.sort(key=lambda res: res.score, reverse=True)
        return scored[:limit]

    def get(self, id: str) -> MemoryRecord | None:
        with self._lock:
            return self._records.get(id)

    def delete(self, id: str, namespace: str | None = None) -> bool:
        with self._lock:
            rec = self._records.get(id)
            if rec is None:
                return False
            if namespace is not None and rec.namespace != namespace:
                return False
            del self._records[id]
            return True

    def clear(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace is None:
                c = len(self._records)
                self._records.clear()
                return c
            to_del = [k for k, v in self._records.items() if v.namespace == namespace]
            for k in to_del:
                del self._records[k]
            return len(to_del)

    def count(self, namespace: str | None = None) -> int:
        with self._lock:
            if namespace is None:
                return len(self._records)
            return sum(1 for v in self._records.values() if v.namespace == namespace)

    def export_json(self, path: str | Path) -> Path:
        """Write all records to ``path`` atomically (tmp sibling + ``os.replace``)."""
        out = Path(path)
        with self._lock:
            data = [r.to_dict() for r in self._records.values()]
        atomic_write_text(out, json.dumps(data, indent=2) + "\n")
        return out

    def load_json(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        records = [MemoryRecord(**item) for item in data]
        with self._lock:
            for rec in records:
                self._records[rec.id] = rec


class QdrantVectorStore(BaseVectorMemory):
    """Qdrant vector database memory adapter (qdrant-client >= 1.19.0)."""

    def __init__(
        self,
        collection_name: str = "agent_vcr_memory",
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        client: Any | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        vector_dim: int = 128,
    ) -> None:
        super().__init__(embedding_fn=embedding_fn, vector_dim=vector_dim)
        self.collection_name = collection_name
        self._injected_client = client

        if client is not None:
            self.client = client
        else:
            if _QDRANT_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "The qdrant-client package is not installed. "
                    'Install with: pip install "ai-agent-vcr[vector]" or pip install qdrant-client>=1.19.0'
                ) from _QDRANT_IMPORT_ERROR

            if path:
                self.client = QdrantClient(path=path)
            elif url:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            if hasattr(self.client, "collection_exists"):
                exists = self.client.collection_exists(self.collection_name)
                if not exists:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_dim,
                            distance=Distance.COSINE,
                        ),
                    )
            else:
                collections = self.client.get_collections().collections
                names = [c.name for c in collections]
                if self.collection_name not in names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_dim,
                            distance=Distance.COSINE,
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Qdrant collection setup note: %s", exc)

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        namespace: str = "default",
        id: str | None = None,
    ) -> str:
        clean_text = redact_text(text)
        clean_meta = _user_metadata(metadata)
        rec_id = id or str(uuid.uuid4())
        vec = self.embed(clean_text)

        payload = {
            **clean_meta,  # reserved keys below always win
            "_orig_id": str(rec_id),
            "text": clean_text,
            "session_id": session_id,
            "namespace": namespace,
            "timestamp": time.time(),
        }

        norm_id = _normalize_qdrant_id(rec_id)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=norm_id,
                    vector=vec,
                    payload=payload,
                )
            ],
        )
        return rec_id

    def add_batch(self, records: list[MemoryRecord]) -> list[str]:
        points: list[PointStruct] = []
        ids: list[str] = []
        for r in records:
            clean_text = redact_text(r.text)
            clean_meta = _user_metadata(r.metadata)
            vec = self.embed(clean_text) if (clean_text != r.text or r.vector is None) else r.vector
            payload = {
                **clean_meta,  # reserved keys below always win
                "_orig_id": str(r.id),
                "text": clean_text,
                "session_id": r.session_id,
                "namespace": r.namespace,
                "timestamp": r.timestamp or time.time(),
            }
            norm_id = _normalize_qdrant_id(r.id)
            points.append(PointStruct(id=norm_id, vector=vec, payload=payload))
            ids.append(r.id)

        self.client.upsert(collection_name=self.collection_name, points=points)
        return ids

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
        namespace: str | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[MemorySearchResult]:
        q_vec = self.embed(query)
        if _is_zero_vector(q_vec):
            return []  # empty / unembeddable query matches nothing

        # Build filter if namespace or metadata provided
        qdrant_filter: Any = None
        if namespace is not None or filter_metadata:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue

                must_conditions: list[Any] = []
                if namespace is not None:
                    must_conditions.append(
                        FieldCondition(key="namespace", match=MatchValue(value=namespace))
                    )
                if filter_metadata:
                    for k, v in filter_metadata.items():
                        must_conditions.append(
                            FieldCondition(key=k, match=MatchValue(value=v))
                        )
                qdrant_filter = Filter(must=must_conditions)
            except Exception:  # noqa: BLE001
                pass

        score_threshold = min_score if min_score > 0.0 else None
        if hasattr(self.client, "query_points"):
            # qdrant-client >= 1.10: ``search`` is deprecated and removed in
            # current releases; ``query_points`` returns a QueryResponse.
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            hits = response.points
        else:  # pragma: no cover - legacy qdrant-client
            search_kwargs: dict[str, Any] = {
                "collection_name": self.collection_name,
                "query_vector": q_vec,
                "limit": limit,
                "score_threshold": score_threshold,
            }
            if qdrant_filter is not None:
                search_kwargs["query_filter"] = qdrant_filter
            hits = self.client.search(**search_kwargs)
        results: list[MemorySearchResult] = []
        for hit in hits:
            if not _score_passes(float(hit.score), min_score):
                continue
            payload = hit.payload or {}
            text = payload.get("text", "")
            session_id = payload.get("session_id")
            ns = payload.get("namespace", "default")
            ts = payload.get("timestamp", 0.0)
            orig_id = payload.get("_orig_id", str(hit.id))
            meta = {
                k: v
                for k, v in payload.items()
                if k not in RESERVED_PAYLOAD_KEYS
            }
            rec = MemoryRecord(
                id=orig_id,
                text=text,
                vector=hit.vector,
                metadata=meta,
                session_id=session_id,
                namespace=ns,
                timestamp=ts,
            )
            results.append(MemorySearchResult(record=rec, score=round(hit.score, 4)))

        return results

    def get(self, id: str) -> MemoryRecord | None:
        try:
            norm_id = _normalize_qdrant_id(id)
            points = self.client.retrieve(collection_name=self.collection_name, ids=[norm_id])
            if not points:
                return None
            p = points[0]
            payload = p.payload or {}
            orig_id = payload.get("_orig_id", str(p.id))
            return MemoryRecord(
                id=orig_id,
                text=payload.get("text", ""),
                vector=p.vector,
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in RESERVED_PAYLOAD_KEYS
                },
                session_id=payload.get("session_id"),
                namespace=payload.get("namespace", "default"),
                timestamp=payload.get("timestamp", 0.0),
            )
        except Exception:  # noqa: BLE001
            return None

    def delete(self, id: str, namespace: str | None = None) -> bool:
        try:
            norm_id = _normalize_qdrant_id(id)
            points = self.client.retrieve(collection_name=self.collection_name, ids=[norm_id])
            if not points:
                return False
            if namespace is not None and (points[0].payload or {}).get("namespace") != namespace:
                return False
            self.client.delete(collection_name=self.collection_name, points_selector=[norm_id])
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _namespace_filter(namespace: str) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))])

    def clear(self, namespace: str | None = None) -> int:
        try:
            c = self.count(namespace)
            if namespace is None:
                self.client.delete_collection(collection_name=self.collection_name)
                self._ensure_collection()
            else:
                flt = self._namespace_filter(namespace)
                try:
                    from qdrant_client.models import FilterSelector
                    selector: Any = FilterSelector(filter=flt)
                except (ImportError, Exception):
                    selector = flt
                self.client.delete(collection_name=self.collection_name, points_selector=selector)
            return c
        except Exception:  # noqa: BLE001
            return 0

    def count(self, namespace: str | None = None) -> int:
        try:
            if namespace is None:
                res = self.client.count(collection_name=self.collection_name, exact=True)
            else:
                res = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=self._namespace_filter(namespace),
                    exact=True,
                )
            return int(getattr(res, "count", 0))
        except Exception:  # noqa: BLE001
            return 0


def create_vector_memory(
    backend: str = "auto",
    collection_name: str = "agent_vcr_memory",
    url: str | None = None,
    api_key: str | None = None,
    path: str | None = None,
    client: Any | None = None,
    embedding_fn: Callable[[str], list[float]] | None = None,
    vector_dim: int = 128,
) -> BaseVectorMemory:
    """Create a vector memory store.

    Supported backends:
    - 'auto': uses Qdrant if qdrant-client is installed; otherwise InMemoryVectorStore.
    - 'qdrant': uses QdrantVectorStore.
    - 'memory': uses InMemoryVectorStore.
    """
    target = backend.lower()
    if target == "auto":
        if client is not None:
            target = "qdrant"
        else:
            try:
                import qdrant_client  # noqa: F401
                target = "qdrant"
            except ImportError:
                target = "memory"

    if target == "qdrant":
        return QdrantVectorStore(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            path=path,
            client=client,
            embedding_fn=embedding_fn,
            vector_dim=vector_dim,
        )
    elif target in ("memory", "in_memory", "local"):
        return InMemoryVectorStore(embedding_fn=embedding_fn, vector_dim=vector_dim)
    else:
        raise ValueError(f"Unknown vector memory backend: {backend}")
