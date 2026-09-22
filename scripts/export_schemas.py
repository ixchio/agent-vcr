"""Export frozen Phase 0 JSON Schemas (execution/v1) to schemas/.

Run: python3 scripts/export_schemas.py
Output: schemas/execution/session.json, schemas/execution/frame.json,
        schemas/events/resume_config.json, schemas/VERSION
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_vcr.bundle import BUNDLE_VERSION, BundleManifest
from agent_vcr.cloud_store import ApiTokenPublic, Project
from agent_vcr.datasets import DatasetFile
from agent_vcr.intelligence.models import Diagnosis, FailureCluster, RiskScore
from agent_vcr.models import SCHEMA_VERSION, Frame, ResumeConfig, Session
from agent_vcr.regression import RegressionReport
from agent_vcr.superagent.models import Plan, RunReport
from agent_vcr.workflows.models import RunRecord, WorkflowDef

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    _write(SCHEMAS / "execution" / "session.json", Session.model_json_schema())
    _write(SCHEMAS / "execution" / "frame.json", Frame.model_json_schema())
    _write(SCHEMAS / "events" / "resume_config.json", ResumeConfig.model_json_schema())
    _write(SCHEMAS / "bundle" / "manifest.json", BundleManifest.model_json_schema())
    _write(SCHEMAS / "ci" / "dataset.json", DatasetFile.model_json_schema())
    _write(SCHEMAS / "ci" / "regression_report.json", RegressionReport.model_json_schema())
    _write(SCHEMAS / "cloud" / "project.json", Project.model_json_schema())
    _write(SCHEMAS / "cloud" / "token.json", ApiTokenPublic.model_json_schema())
    _write(SCHEMAS / "superagent" / "plan.json", Plan.model_json_schema())
    _write(SCHEMAS / "superagent" / "run_report.json", RunReport.model_json_schema())
    _write(SCHEMAS / "workflows" / "definition.json", WorkflowDef.model_json_schema())
    _write(SCHEMAS / "workflows" / "run_record.json", RunRecord.model_json_schema())
    _write(SCHEMAS / "intel" / "failure_cluster.json", FailureCluster.model_json_schema())
    _write(SCHEMAS / "intel" / "risk_score.json", RiskScore.model_json_schema())
    _write(SCHEMAS / "intel" / "diagnosis.json", Diagnosis.model_json_schema())
    _write(
        SCHEMAS / "bundle" / "README.json",
        {"bundle": BUNDLE_VERSION, "execution_schema": "execution/v1"},
    )
    _write(
        SCHEMAS / "execution" / "README.json",
        {"schema": "execution/v1", "version": SCHEMA_VERSION},
    )
    (SCHEMAS / "VERSION").write_text(SCHEMA_VERSION + "\n")
    print(f"schema execution/v1 version {SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
