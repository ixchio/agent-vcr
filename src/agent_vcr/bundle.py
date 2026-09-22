"""Replay Bundle — portable reproduction artifact (Phase 1).

A ``.vcrb`` bundle is a gzipped tar containing::

    execution.vcr   # Phase 0 JSONL recording (source of truth)
    bundle.json     # BundleManifest: identity, hashes, environment

``bundle.json`` pins everything needed to reproduce or audit a run:
session ID, schema version, frame count, sha256 of ``execution.vcr``,
git commit (best-effort), models used, tools used, and cost/tokens.

One-liner reproduction::

    vcr reproduce ./session.vcrb

Stdlib only (tarfile/hashlib/json) — no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_vcr.models import SCHEMA_VERSION

logger = logging.getLogger(__name__)

BUNDLE_VERSION = "1.0"
BUNDLE_SUFFIX = ".vcrb"
EXECUTION_NAME = "execution.vcr"
MANIFEST_NAME = "bundle.json"


class BundleManifest(BaseModel):
    """Identity + integrity metadata for a replay bundle."""

    bundle_version: str = BUNDLE_VERSION
    session_id: str
    schema_version: str = SCHEMA_VERSION
    frame_count: int
    execution_sha256: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    models: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_count: int = 0
    checkpoints: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: Path | None = None) -> str | None:
    """Best-effort git commit hash; None outside a repo (never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        logger.debug("git rev-parse unavailable", exc_info=True)
    return None


def _summarize_player(player: Any) -> dict[str, Any]:
    """Collect model/tool/cost/checkpoint summary from a loaded player."""
    from agent_vcr.models import FrameType

    models: list[str] = []
    tools: list[str] = []
    errors = 0
    for frame in player.frames:
        if frame.metadata and frame.metadata.model and frame.metadata.model not in models:
            models.append(frame.metadata.model)
        if frame.node_name.startswith("tool:"):
            tool = frame.node_name.split("tool:", 1)[1]
            if tool not in tools:
                tools.append(tool)
        if frame.frame_type == FrameType.ERROR:
            errors += 1
    checkpoints = [cp.get("label") or cp.get("node_name") for cp in player.list_checkpoints()]
    return {
        "models": models,
        "tools": tools,
        "error_count": errors,
        "checkpoints": [c for c in checkpoints if c],
        "total_tokens": player.get_total_tokens(),
        "total_cost_usd": player.get_total_cost(),
    }


def export_bundle(
    source_vcr: str | Path,
    output: str | Path | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Export a ``.vcr`` recording to a portable ``.vcrb`` bundle.

    Args:
        source_vcr: Path to the recorded ``.vcr`` file.
        output: Destination bundle path (default: ``<session_id>.vcrb``
            next to the source file).
        extra: Extra caller-supplied metadata stored verbatim.

    Returns:
        Path to the written bundle.
    """
    from agent_vcr.player import VCRPlayer

    source = Path(source_vcr)
    if not source.exists():
        raise FileNotFoundError(f"VCR file not found: {source}")

    player = VCRPlayer.load(source)
    integrity = player.verify_integrity()
    if not integrity["ok"]:
        raise ValueError(f"Refusing to bundle corrupt session: {integrity['issues']}")

    summary = _summarize_player(player)
    manifest = BundleManifest(
        session_id=player.session.session_id,
        schema_version=player.session.schema_version,
        frame_count=len(player.frames),
        execution_sha256=_sha256_file(source),
        git_commit=_git_commit(source.parent),
        models=summary["models"],
        tools=summary["tools"],
        total_tokens=summary["total_tokens"],
        total_cost_usd=summary["total_cost_usd"],
        error_count=summary["error_count"],
        checkpoints=summary["checkpoints"],
        extra=extra or {},
    )

    dest = Path(output) if output is not None else source.parent / f"{player.session.session_id}{BUNDLE_SUFFIX}"
    if dest.suffix != BUNDLE_SUFFIX:
        dest = dest.with_suffix(BUNDLE_SUFFIX)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest_path = tmp / MANIFEST_NAME
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(str(source), arcname=EXECUTION_NAME)
            tar.add(str(manifest_path), arcname=MANIFEST_NAME)

    logger.info("Exported bundle %s (%d frames)", dest, len(player.frames))
    return dest


def _extract_member(tar: tarfile.TarFile, member: str, path: str | Path) -> None:
    """Extract with data filter on 3.12+ (3.14 default), compat fallback below."""
    try:
        tar.extract(member, path=path, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tar.extract(member, path=path)


def _read_manifest_from_tar(tar: tarfile.TarFile) -> BundleManifest:
    try:
        member = tar.getmember(MANIFEST_NAME)
    except KeyError as err:
        raise ValueError(f"Invalid bundle: missing {MANIFEST_NAME}") from err
    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"Invalid bundle: cannot read {MANIFEST_NAME}")
    return BundleManifest(**json.loads(handle.read().decode("utf-8")))


def verify_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Verify bundle integrity without extracting (sha256 + frame count)."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle not found: {path}")

    try:
        with tarfile.open(path, "r:gz") as tar:
            names = tar.getnames()
            if EXECUTION_NAME not in names:
                return {"ok": False, "issues": [f"missing {EXECUTION_NAME}"], "bundle": str(path)}
            manifest = _read_manifest_from_tar(tar)
            with tempfile.TemporaryDirectory() as tmpdir:
                _extract_member(tar, EXECUTION_NAME, tmpdir)
                actual_sha = _sha256_file(Path(tmpdir) / EXECUTION_NAME)
    except (tarfile.TarError, EOFError, OSError, ValueError, KeyError) as err:
        return {"ok": False, "issues": [f"unreadable bundle: {err}"], "bundle": str(path)}

    issues: list[str] = []
    if actual_sha != manifest.execution_sha256:
        issues.append("execution.vcr sha256 mismatch (tampered bundle)")
    return {
        "ok": not issues,
        "issues": issues,
        "bundle": str(path),
        "session_id": manifest.session_id,
        "frame_count": manifest.frame_count,
        "schema_version": manifest.schema_version,
        "git_commit": manifest.git_commit,
        "models": manifest.models,
        "tools": manifest.tools,
    }


def import_bundle(bundle_path: str | Path, dest_dir: str | Path = ".") -> Path:
    """Extract + verify a bundle; returns the path to ``execution.vcr``.

    Raises:
        ValueError: If integrity verification fails.
    """
    report = verify_bundle(bundle_path)
    if not report["ok"]:
        raise ValueError(f"Bundle verification failed: {report['issues']}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{report['session_id']}.vcr"
    with tarfile.open(bundle_path, "r:gz") as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            _extract_member(tar, EXECUTION_NAME, tmpdir)
            data = (Path(tmpdir) / EXECUTION_NAME).read_bytes()
        out.write_bytes(data)
    logger.info("Imported bundle %s -> %s", bundle_path, out)
    return out


def reproduce(bundle_path: str | Path) -> dict[str, Any]:
    """One-liner reproduction: verify → load → integrity → summary.

    Returns a JSON-serializable report (design doc §72). Side-effect free:
    never executes agent code, only replays recorded state.
    """
    from agent_vcr.player import VCRPlayer

    report = verify_bundle(bundle_path)
    if not report["ok"]:
        return {**report, "reproduced": False}

    with tempfile.TemporaryDirectory() as tmpdir:
        execution = import_bundle(bundle_path, dest_dir=tmpdir)
        player = VCRPlayer.load(execution)
        integrity = player.verify_integrity()
        summary = _summarize_player(player)

    return {
        **report,
        "reproduced": integrity["ok"],
        "integrity": integrity,
        "total_tokens": summary["total_tokens"],
        "total_cost_usd": summary["total_cost_usd"],
        "error_count": summary["error_count"],
        "checkpoints": summary["checkpoints"],
    }
