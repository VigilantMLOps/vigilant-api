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
    serving_model_version: str | None = None

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


# ---------------------------------------------------------------------------
# Ingest — Accept pre-computed metrics from vigilant-detect
# ---------------------------------------------------------------------------


class IngestMetricsRequest(BaseModel):
    y_true: list[int]
    y_pred_proba: list[float]
    model_version: str | None = None
    schema_hash: str | None = None
    display_name: str | None = None


class IngestMetricsResponse(BaseModel):
    report_id: str
    f1: float
    roc_auc: float
    avg_precision: float
    accuracy: float


@router.post("/reporter/ingest-metrics", response_model=IngestMetricsResponse)
def ingest_model_metrics(body: IngestMetricsRequest):
    """
    Accept pre-computed y_true / y_pred_proba from vigilant-detect and persist
    as a PRE_PROD evaluation report.

    If `display_name` is provided, the report is linked to a models row with that
    display_name. If no such row exists it is created automatically, so each
    distinct model type (NIDS, ATO Detector, …) gets its own entry in the models
    table rather than all defaulting to the NIDS sentinel model.
    """
    if len(body.y_true) != len(body.y_pred_proba):
        raise HTTPException(status_code=400, detail="y_true and y_pred_proba must have the same length.")
    if len(body.y_true) == 0:
        raise HTTPException(status_code=400, detail="y_true must not be empty.")

    # Resolve (or register) the model_id for this display_name.
    model_id: str | None = None
    if body.display_name:
        import uuid as _uuid
        row = db.fetchone(
            "SELECT model_id FROM models WHERE display_name = ?",
            [body.display_name],
        )
        if row:
            model_id = row["model_id"]
        else:
            model_id = str(_uuid.uuid4())
            db.execute(
                "INSERT INTO models (model_id, model_name, display_name, model_version)"
                " VALUES (?, ?, ?, ?)",
                [model_id, body.display_name, body.display_name, "v1"],
            )

    reporter = _get_reporter()
    result = reporter.ingest_precomputed_metrics(body.y_true, body.y_pred_proba)

    report_id = reporter.save_report_to_db(
        result,
        report_type="PRE_PROD",
        model_version=body.model_version,
        model_id=model_id,
    )
    return IngestMetricsResponse(
        report_id=report_id,
        f1=result.f1,
        roc_auc=result.roc_auc,
        avg_precision=result.avg_precision,
        accuracy=result.accuracy,
    )


# ---------------------------------------------------------------------------
# Ingest — Accept pre-computed DATA_EVAL stats from vigilant-detect
# ---------------------------------------------------------------------------


class IngestDataEvalRequest(BaseModel):
    split: str
    model_version: str | None = None
    n_rows: int
    n_features: int
    class_distribution: dict[str, int]
    imbalance_ratio: float
    duplicate_rows: int = 0
    missing_cells: int = 0
    features: list[dict[str, Any]] = []


class IngestDataEvalResponse(BaseModel):
    report_id: str
    split: str
    n_rows: int


@router.post("/reporter/ingest-data-eval", response_model=IngestDataEvalResponse)
def ingest_data_eval(body: IngestDataEvalRequest):
    """Accept pre-computed data-evaluation statistics from vigilant-detect and store
    as a DATA_EVAL report linked to the ATO Detector model."""
    import uuid as _uuid

    # Resolve the ATO Detector model_id so it's linked correctly.
    row = db.fetchone("SELECT model_id FROM models WHERE display_name = 'ATO Detector'")
    model_id: str | None = row["model_id"] if row else None

    report_id = str(_uuid.uuid4())
    content = {
        "split": body.split,
        "n_rows": body.n_rows,
        "n_features": body.n_features,
        "class_distribution": body.class_distribution,
        "imbalance_ratio": body.imbalance_ratio,
        "duplicate_rows": body.duplicate_rows,
        "missing_cells": body.missing_cells,
        "features": body.features,
    }
    reporter = _get_reporter()
    reporter._report_repo.insert_data_eval(
        report_id=report_id,
        content=content,
        model_version=body.model_version or body.split,
        model_id=model_id,
    )
    return IngestDataEvalResponse(report_id=report_id, split=body.split, n_rows=body.n_rows)


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
    Compare production data against the training reference distribution.

    Modes (in priority order):
      1. ?split=<name>  — load an existing parquet split as the incoming batch
      2. POST body { "records": [...] }  — explicit batch from the caller
      3. No args  — evaluate against records already accumulated in production_log
    """
    reporter = _get_reporter()

    if split is not None:
        try:
            production_df = reporter._loader.load_split(split)
        except (ValueError, Exception) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    elif body is not None and body.records:
        from services.data_loader import DataLoader
        production_df = DataLoader.from_records(body.records)
    else:
        production_df = None  # use accumulated production_log

    try:
        return reporter.evaluate_data_drift(production_df, model_version=model_version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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
        serving_version: str | None = None
        try:
            serving_version = resp.json().get("model_version")
        except Exception:
            pass
        return {
            "model_api": "ok",
            "url": url,
            "status_code": resp.status_code,
            "serving_model_version": serving_version,
        }
    except httpx.HTTPError as exc:
        return {"model_api": "unreachable", "url": url, "error": str(exc)}
