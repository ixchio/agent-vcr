"""Replay Datasets — versioned regression corpora for Agent CI (Phase 2).

A dataset is a named, versioned collection of :class:`TestCase` entries.
Each case pins a golden run (by fingerprint in :class:`GoldenRunCache`)
plus **test annotations** (:class:`Expectation`): budgets, forbidden
tools, required checkpoints, and partial expected final state.

Storage: ``<datasets_dir>/<name>.json`` (atomic write, same pattern as
the session manifest). Golden payloads stay in the golden cache —
datasets store references, never copies.

Typical flow::

    dataset = ReplayDataset.create("refunds", datasets_dir=".vcr/datasets")
    dataset.add_case(task="refund $10", recorder=rec,
                     expect=Expectation(max_cost_usd=0.30, require_no_errors=True))
    run_regression(dataset, candidates={"refund $10": candidate_player}, ...)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATASET_VERSION = "1.0"


class Expectation(BaseModel):
    """Test annotations for one case: budgets, safety rails, expected state."""

    max_cost_usd: float | None = None
    max_latency_ms: float | None = None
    max_tokens: int | None = None
    require_no_errors: bool = True
    expect_state: dict[str, Any] = Field(default_factory=dict)
    forbid_tools: list[str] = Field(default_factory=list)
    require_checkpoints: list[str] = Field(default_factory=list)
    # Regression thresholds vs the golden baseline (fraction, e.g. 0.5 = +50%).
    max_cost_delta: float | None = 0.5
    max_latency_delta: float | None = 1.0
    max_token_delta: float | None = 0.5


class TestCase(BaseModel):
    """One annotated regression case: golden reference + expectations."""

    name: str
    task: str
    fingerprint: str
    identity: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    expect: Expectation = Field(default_factory=Expectation)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetFile(BaseModel):
    """On-disk dataset envelope (versioned for forward migration)."""

    dataset_version: str = DATASET_VERSION
    name: str
    version: int = 1
    cases: list[TestCase] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReplayDataset:
    """Versioned store of annotated golden-run references."""

    def __init__(self, name: str, datasets_dir: str | Path = ".vcr/datasets"):
        self.name = name
        self.dir = Path(datasets_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._doc = self._load()

    # -- construction ----------------------------------------------------

    @classmethod
    def create(cls, name: str, datasets_dir: str | Path = ".vcr/datasets") -> ReplayDataset:
        ds = cls(name, datasets_dir)
        if ds._doc.cases:
            raise ValueError(f"Dataset {name!r} already exists ({len(ds._doc.cases)} cases)")
        ds._save()
        return ds

    @classmethod
    def load(cls, name: str, datasets_dir: str | Path = ".vcr/datasets") -> ReplayDataset:
        ds = cls(name, datasets_dir)
        if not ds._path().exists() and not ds._doc.cases:
            raise FileNotFoundError(f"Dataset {name!r} not found in {ds.dir}")
        return ds

    @staticmethod
    def list_datasets(datasets_dir: str | Path = ".vcr/datasets") -> list[str]:
        d = Path(datasets_dir)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    # -- mutation --------------------------------------------------------

    def add_case(
        self,
        task: str,
        recorder: Any,
        *,
        name: str | None = None,
        identity: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        expect: Expectation | None = None,
        golden_cache: Any | None = None,
    ) -> TestCase:
        """Record a golden run from ``recorder`` and pin it as a case."""
        from agent_vcr.golden_cache import GoldenRunCache

        cache = golden_cache or GoldenRunCache()
        fingerprint = cache.save_golden_run(task, recorder, tags=tags, identity=identity)
        case = TestCase(
            name=name or task,
            task=task,
            fingerprint=fingerprint,
            identity=cache._normalize_identity(identity),
            tags=tags or [],
            expect=expect or Expectation(),
        )
        return self.add_case_ref(case)

    def add_case_ref(self, case: TestCase) -> TestCase:
        if any(c.name == case.name for c in self._doc.cases):
            raise ValueError(f"Case {case.name!r} already in dataset {self.name!r}")
        self._doc.cases.append(case)
        self._doc.version += 1
        self._save()
        logger.info("Dataset %s: added case %r (v%d)", self.name, case.name, self._doc.version)
        return case

    def remove_case(self, name: str) -> bool:
        before = len(self._doc.cases)
        self._doc.cases = [c for c in self._doc.cases if c.name != name]
        if len(self._doc.cases) == before:
            return False
        self._doc.version += 1
        self._save()
        return True

    # -- access ----------------------------------------------------------

    @property
    def version(self) -> int:
        return self._doc.version

    def cases(self) -> list[TestCase]:
        return list(self._doc.cases)

    def select(self, tags: list[str] | None = None) -> list[TestCase]:
        """Select cases matching ALL given tags (None/empty = all cases)."""
        if not tags:
            return self.cases()
        wanted = set(tags)
        return [c for c in self._doc.cases if wanted <= set(c.tags)]

    def get_case(self, name: str) -> TestCase:
        for case in self._doc.cases:
            if case.name == name:
                return case
        raise KeyError(f"Case {name!r} not in dataset {self.name!r}")

    # -- persistence -----------------------------------------------------

    def _path(self) -> Path:
        return self.dir / f"{self.name}.json"

    def _load(self) -> DatasetFile:
        path = self._path()
        if not path.exists():
            return DatasetFile(name=self.name)
        try:
            return DatasetFile(**json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as err:
            raise ValueError(f"Corrupt dataset file {path}: {err}") from err

    def _save(self) -> None:
        self._doc.updated_at = datetime.now(timezone.utc).isoformat()
        fd, tmp_path = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp", prefix="ds_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._doc.model_dump(), f, indent=2)
            os.replace(tmp_path, str(self._path()))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
