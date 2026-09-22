"""Failure clustering over recorded sessions (Phase 6, §27).

Error messages contain volatile spans (ids, counts, paths, timestamps).
:func:`normalize_message` replaces them with placeholders so repeated
failures collapse to one template; :func:`cluster_sessions` groups
error frames by ``(error_type, node_name, template)`` across a session
directory, ranked by occurrences.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from agent_vcr.intelligence.models import FailureCluster
from agent_vcr.models import FrameType
from agent_vcr.player import VCRPlayer

logger = logging.getLogger(__name__)

_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d:?\d{2})?\b"), "<ts>"),
    (re.compile(r"(?<=/)[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"), "<path>"),
    (re.compile(r"\b\d+[a-zA-Z%]+\b"), "<n>"),  # 3000ms, 50GB, 4o (before bare digits)
    (re.compile(r"\b\d+\b"), "<n>"),
]


def normalize_message(message: str) -> str:
    """Replace volatile spans so repeats share one template (pure)."""
    template = message.strip()
    for pattern, placeholder in _VOLATILE:
        template = pattern.sub(placeholder, template)
    return " ".join(template.split())


def cluster_key(error_type: str, node_name: str, template: str) -> str:
    digest = hashlib.sha256(f"{error_type}\x00{node_name}\x00{template}".encode()).hexdigest()
    return digest[:16]


def cluster_sessions(sessions_dir: str | Path, limit: int = 500) -> list[FailureCluster]:
    """Cluster error frames across ``*.vcr`` files, most frequent first."""
    directory = Path(sessions_dir)
    clusters: dict[str, FailureCluster] = {}
    scanned = 0
    for path in sorted(directory.glob("*.vcr")):
        if scanned >= limit:
            break
        scanned += 1
        try:
            player = VCRPlayer.load(path)
        except (ValueError, OSError) as err:
            logger.debug("Skipping unloadable session %s: %s", path.name, err)
            continue
        for frame in player.frames:
            message: str | None = None
            if frame.frame_type == FrameType.ERROR:
                message = (frame.metadata.error_message or "") or _output_error(frame)
            elif frame.metadata.error_type:
                message = frame.metadata.error_message or frame.metadata.error_type
            if not message:
                continue
            error_type = frame.metadata.error_type or "Error"
            template = normalize_message(message)
            key = cluster_key(error_type, frame.node_name, template)
            cluster = clusters.get(key)
            stamp = frame.timestamp.isoformat()
            if cluster is None:
                clusters[key] = FailureCluster(
                    key=key, error_type=error_type, node_name=frame.node_name,
                    template=template, occurrences=1,
                    sessions=[player.session.session_id],
                    first_seen=stamp, last_seen=stamp, sample_message=message,
                )
            else:
                cluster.occurrences += 1
                if player.session.session_id not in cluster.sessions:
                    cluster.sessions.append(player.session.session_id)
                cluster.first_seen = min(cluster.first_seen, stamp)
                cluster.last_seen = max(cluster.last_seen, stamp)
    return sorted(clusters.values(), key=lambda c: (-c.occurrences, c.key))


def _output_error(frame: Any) -> str:
    try:
        output = frame.output_state
        if isinstance(output, dict) and "error" in output:
            return str(output["error"])
    except Exception:  # noqa: BLE001 — best-effort extraction
        pass
    return ""
