"""Reporter routes — trigger evaluation runs directly from Postman / the dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.database import db
from services.alerting_engine import AlertManager
from services.reporter import (
    DataDriftResult,
    ModelEvaluationResult,
    ReporterConfig,
    ReporterService,
)


class DeletedRecordsResponse(BaseModel):
    deleted_records: int


class ModelHealthResponse(BaseModel):
    model_api: str
    url: str
    status_code: int | None = None
    error: str | None = None

router = APIRouter()

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "reporter.json"
_reporter: ReporterService | None = None


def _get_reporter() -> ReporterService:
    global _reporter
    if _reporter is None:
        if not _CONFIG_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Reporter config not found at {_CONFIG_PATH}",
            )
        config = ReporterConfig.from_json(_CONFIG_PATH)
        _reporter = ReporterService(config, db=db, alert_manager=AlertManager(db=db))
    return _reporter


# ---------------------------------------------------------------------------
# Pre-Production — Data Evaluation
# ---------------------------------------------------------------------------


@router.post("/reporter/evaluate-data", response_model=dict[str, Any])
def run_evaluate_data():
    """
    Profile every data stage and split in one request.

    Returns:
      balanced.train / balanced.test / balanced.val  — processed balanced splits
      raw.unsw_nb15 / raw.ciciot2023 / raw.combined  — original raw CSV sources

    Warning: raw stages load large CSV files and may take a minute or more.
    """
    reporter = _get_reporter()
    try:
        result = reporter.evaluate_all_data()
        return {
            stage: {
                split: (v.model_dump() if hasattr(v, "model_dump") else v)
                for split, v in splits.items()
            }
            for stage, splits in result.items()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Pre-Production — Model Evaluation
# ---------------------------------------------------------------------------


class ModelEvalRequest(BaseModel):
    records: list[dict[str, Any]] | None = None


@router.post("/reporter/evaluate-model", response_model=ModelEvaluationResult)
def run_evaluate_model(
    model_version: str | None = Query(None, description="Optional version tag stored with the report"),
    body: ModelEvalRequest | None = None,
):
    """
    Send the test split to the remote model API and compute classification metrics.
    Saves the result as the baseline for drift/decay tracking and persists to DB.

    When called without a body: loads test_nodup_hybrid.parquet from disk (falls back
    to the latest stored PRE_PROD report if the parquet is unavailable).
    When called with { "records": [{...}, ...] }: uses those labeled records directly —
    each record must include the target column (label).
    """
    reporter = _get_reporter()
    df = None
    if body and body.records:
        from services.data_loader import DataLoader
        df = DataLoader.from_records(body.records)
    try:
        return reporter.evaluate_model(df=df, model_version=model_version)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model API unreachable — check model_api.base_url in reporter.json. Error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Production — Data Drift
# ---------------------------------------------------------------------------


class DriftRequest(BaseModel):
    records: list[dict[str, Any]]


@router.post("/reporter/evaluate-drift", response_model=DataDriftResult)
def run_evaluate_drift(
    split: str | None = Query(
        None,
        description="Use an existing split as stand-in production data (train/test/val). "
                    "Omit to POST records in the request body instead.",
    ),
    model_version: str | None = Query(None, description="Model version tag included in drift alert metadata."),
    body: DriftRequest | None = None,
):
    """
    Compare a production batch against the training reference distribution.
    For testing: pass ?split=val to reuse the validation split as fake production data.
    For real production data: POST a JSON body with { "records": [{...}, ...] }.
    """
    reporter = _get_reporter()

    if split is not None:
        try:
            production_df = reporter._loader.load_split(split)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    elif body is not None and body.records:
        from services.data_loader import DataLoader
        production_df = DataLoader.from_records(body.records)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either ?split=<name> or a JSON body with 'records'.",
        )

    try:
        return reporter.evaluate_data_drift(production_df, model_version=model_version)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Production — Reset accumulated drift window
# ---------------------------------------------------------------------------


@router.delete("/reporter/production-log", response_model=DeletedRecordsResponse)
def reset_production_log():
    """
    Delete all accumulated production records from the drift window.
    Use this to start a fresh evaluation period (e.g. after a model retrain).
    Returns the number of rows deleted.
    """
    reporter = _get_reporter()
    try:
        n_deleted = reporter.reset_production_log()
        return {"deleted_records": n_deleted}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Utility — Export baseline feature stats (used by seed script)
# ---------------------------------------------------------------------------


class FeatureStatsRow(BaseModel):
    feature_name: str
    # PostgreSQL JSONB columns deserialize to dict via psycopg2.
    stats_json: dict[str, Any]


@router.get("/reporter/feature-stats", response_model=list[FeatureStatsRow])
def get_feature_stats(model_id: str | None = None):
    """Return the model's baseline as a list of {feature_name, stats_json} rows,
    sorted by feature name. Reads from models.baseline. When no model_id is
    given, falls back to the database's resolved default model."""
    try:
        from repositories import ModelRepository
        baseline = ModelRepository(db).get_baseline(model_id) or {}
        return [
            {"feature_name": name, "stats_json": stats}
            for name, stats in sorted(baseline.items())
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Utility — Model API health check
# ---------------------------------------------------------------------------


@router.get("/reporter/model-health", response_model=ModelHealthResponse)
def check_model_api_health():
    """Ping the configured model API health endpoint."""
    reporter = _get_reporter()
    url = reporter.config.model_api.health_url
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(url)
        return {"model_api": "ok", "url": url, "status_code": resp.status_code}
    except httpx.HTTPError as exc:
        return {"model_api": "unreachable", "url": url, "error": str(exc)}
