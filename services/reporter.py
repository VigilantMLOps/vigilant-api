"""
Reporter Service — VigilantMLOps

All evaluation and monitoring math lives here. API routes stay thin wrappers.

Evaluation types
----------------
Pre-Production
    evaluate_data(split)  — statistical profile of a dataset split
    evaluate_model()      — classification metrics via remote model API

Production
    evaluate_data_drift(production_df)             — PSI / KS / Chi² per feature
    evaluate_production_performance(preds, labels) — rolling metrics vs baseline
"""
from __future__ import annotations

import json
import math
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database

import httpx
import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report as sk_classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from core.logger import get_logger
from .alerting_engine import AlertManager
from .data_loader import DataLoader, DataPaths

_logger = get_logger("vigilant.reporter")


# ===========================================================================
# Configuration
# ===========================================================================


class ModelAPIConfig(BaseModel):
    base_url: str
    predict_endpoint: str = "/predict"
    health_endpoint: str = "/health"
    timeout_seconds: int = 30
    api_key: str | None = None
    model_version: str | None = None

    @property
    def predict_url(self) -> str:
        return self.base_url.rstrip("/") + self.predict_endpoint

    @property
    def health_url(self) -> str:
        return self.base_url.rstrip("/") + self.health_endpoint


class DriftThresholds(BaseModel):
    # PSI (Population Stability Index)
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    # Statistical tests — flag when p-value drops below threshold
    ks_pvalue_threshold: float = 0.05
    chi2_pvalue_threshold: float = 0.05
    # Absolute metric drop vs baseline to trigger alerts
    performance_decay_warning: float = 0.05
    performance_decay_critical: float = 0.10


def _apply_env_overrides(raw: dict) -> dict:
    """Overlay env vars onto a raw reporter config dict before validation."""
    import os
    env = os.environ.get

    raw.setdefault("data", {}).setdefault("raw", {})
    if v := env("DATA_RAW_UNSW_DIR"):
        raw["data"]["raw"]["unsw_nb15_dir"] = v
    if v := env("DATA_RAW_CICIOT_DIR"):
        raw["data"]["raw"]["ciciot2023_dir"] = v
    if v := env("DATA_BALANCED_DIR"):
        raw["data"]["balanced_dir"] = v
    if v := env("DATA_BLACKLIST_PATH"):
        raw["data"]["blacklist"] = v
    raw.setdefault("model_api", {})
    if v := env("MODEL_API_URL"):
        raw["model_api"]["base_url"] = v
    return raw


class ReporterConfig(BaseModel):
    data: DataPaths
    model_api: ModelAPIConfig
    thresholds: DriftThresholds = Field(default_factory=DriftThresholds)
    target_column: str = "label"
    feature_columns: list[str] | None = None    # None → infer (all non-target cols)
    categorical_columns: list[str] = Field(default_factory=list)
    psi_bins: int = 10
    production_window_size: int = 500           # rows per rolling evaluation window

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ReporterConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(_apply_env_overrides(raw))

    @classmethod
    def from_json(cls, path: Path | str) -> "ReporterConfig":
        import json
        with open(path) as fh:
            raw = json.load(fh)
        return cls.model_validate(_apply_env_overrides(raw))


# ===========================================================================
# Result schemas
# ===========================================================================


class DriftStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureStats(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None


class DataEvaluationResult(BaseModel):
    split: str
    n_rows: int
    n_features: int
    class_distribution: dict[str, int]
    imbalance_ratio: float                  # majority / minority class count
    duplicate_rows: int
    missing_cells: int
    features: list[FeatureStats]


class ModelEvaluationResult(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float                          # 0.0 when probabilities not returned by API
    avg_precision: float
    confusion_matrix: list[list[int]]
    roc_curve_fpr: list[float] = Field(default_factory=list)   # x-axis for ROC plot
    roc_curve_tpr: list[float] = Field(default_factory=list)   # y-axis for ROC plot
    report: str                             # full sklearn classification report


class FeatureDriftResult(BaseModel):
    feature: str
    method: str                             # "psi+ks" for numeric | "psi+chi2" for categorical
    statistic: float                        # PSI score
    pvalue: float | None                    # KS or Chi² p-value
    status: DriftStatus


class DataDriftResult(BaseModel):
    n_accumulated: int
    n_features_checked: int
    n_drifted: int
    drift_rate: float
    overall_status: DriftStatus
    features: list[FeatureDriftResult]


class PerformanceWindow(BaseModel):
    window_index: int
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    decay_accuracy: float                   # positive = degradation vs baseline
    decay_f1: float
    status: DriftStatus


class ProductionModelEvaluationResult(BaseModel):
    n_samples: int
    baseline_accuracy: float
    baseline_f1: float
    current_accuracy: float
    current_f1: float
    decay_accuracy: float
    decay_f1: float
    overall_status: DriftStatus
    windows: list[PerformanceWindow]


# ===========================================================================
# Private helpers
# ===========================================================================

_EPSILON = 1e-8
_STATUS_PRIORITY: dict[DriftStatus, int] = {
    DriftStatus.OK: 0,
    DriftStatus.WARNING: 1,
    DriftStatus.CRITICAL: 2,
}


def _max_status(*statuses: DriftStatus) -> DriftStatus:
    return max(statuses, key=lambda s: _STATUS_PRIORITY[s])


def _status_from_psi(psi: float, t: DriftThresholds) -> DriftStatus:
    if psi >= t.psi_critical:
        return DriftStatus.CRITICAL
    if psi >= t.psi_warning:
        return DriftStatus.WARNING
    return DriftStatus.OK


def _status_from_pvalue(pvalue: float, threshold: float) -> DriftStatus:
    return DriftStatus.CRITICAL if pvalue < threshold else DriftStatus.OK


def _status_from_decay(decay_acc: float, decay_f1: float, t: DriftThresholds) -> DriftStatus:
    worst = max(abs(decay_acc), abs(decay_f1))
    if worst >= t.performance_decay_critical:
        return DriftStatus.CRITICAL
    if worst >= t.performance_decay_warning:
        return DriftStatus.WARNING
    return DriftStatus.OK


def _resolve_feature_cols(df: pl.DataFrame, cfg: ReporterConfig) -> list[str]:
    if cfg.feature_columns:
        return cfg.feature_columns
    return [c for c in df.columns if c != cfg.target_column]


def _compute_feature_stats(series: pl.Series) -> FeatureStats:
    n = series.len()
    null_count = series.null_count()
    clean = series.drop_nulls()

    result: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "missing_count": null_count,
        "missing_pct": round(null_count / n, 4) if n > 0 else 0.0,
        "n_unique": series.n_unique(),
    }

    if series.dtype.is_numeric() and len(clean) > 0:
        arr = clean.to_numpy()
        result.update(
            mean=round(float(np.mean(arr)), 6),
            std=round(float(np.std(arr, ddof=1)), 6),
            min=round(float(np.min(arr)), 6),
            p25=round(float(np.percentile(arr, 25)), 6),
            p50=round(float(np.percentile(arr, 50)), 6),
            p75=round(float(np.percentile(arr, 75)), 6),
            max=round(float(np.max(arr)), 6),
        )

    return FeatureStats(**result)


def _psi_numeric(ref: pl.Series, prod: pl.Series, n_bins: int) -> float:
    """PSI for continuous features using reference quantile bins.

    Uses adaptive binning: enforces at least 5 production records per bin so
    PSI doesn't explode when the production window is small. Full n_bins is
    restored once the window is large enough.
    """
    ref_arr = ref.drop_nulls().to_numpy()
    prod_arr = prod.drop_nulls().to_numpy()

    effective_bins = min(n_bins, max(2, len(prod_arr) // 5))

    breakpoints = np.quantile(ref_arr, np.linspace(0, 1, effective_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=breakpoints)
    prod_counts, _ = np.histogram(prod_arr, bins=breakpoints)

    ref_pct = (ref_counts + _EPSILON) / (len(ref_arr) + _EPSILON * effective_bins)
    prod_pct = (prod_counts + _EPSILON) / (len(prod_arr) + _EPSILON * effective_bins)

    return round(float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))), 6)


def _psi_categorical(ref: pl.Series, prod: pl.Series) -> float:
    """PSI for categorical features — computed directly from category proportions."""
    all_cats = set(ref.drop_nulls().unique().to_list()) | set(prod.drop_nulls().unique().to_list())
    n_cats = len(all_cats) or 1
    ref_n = ref.drop_nulls().len()
    prod_n = prod.drop_nulls().len()

    psi = 0.0
    for cat in all_cats:
        ref_pct = ((ref == cat).sum() + _EPSILON) / (ref_n + _EPSILON * n_cats)
        prod_pct = ((prod == cat).sum() + _EPSILON) / (prod_n + _EPSILON * n_cats)
        psi += (prod_pct - ref_pct) * math.log(prod_pct / ref_pct)

    return round(psi, 6)


def _ks_test(ref: pl.Series, prod: pl.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test for continuous features."""
    ks_stat, pvalue = stats.ks_2samp(
        ref.drop_nulls().to_numpy(),
        prod.drop_nulls().to_numpy(),
    )
    return round(float(ks_stat), 6), round(float(pvalue), 6)


def _chi2_test(ref: pl.Series, prod: pl.Series) -> tuple[float, float]:
    """Chi-squared test via 2×K contingency table for categorical features."""
    all_cats = sorted(
        set(ref.drop_nulls().unique().to_list()) | set(prod.drop_nulls().unique().to_list())
    )
    ref_counts = np.array([(ref == c).sum() for c in all_cats], dtype=float)
    prod_counts = np.array([(prod == c).sum() for c in all_cats], dtype=float)
    chi2, pvalue, *_ = stats.chi2_contingency(np.vstack([ref_counts, prod_counts]))
    return round(float(chi2), 6), round(float(pvalue), 6)


# ===========================================================================
# Service
# ===========================================================================


class ReporterService:
    """
    Central evaluation service for VigilantMLOps.

    Instantiate with a ReporterConfig (or load one from YAML via
    ReporterConfig.from_yaml(path)). Call evaluate_model() at least once
    before evaluate_production_performance() to establish a baseline.

    Remote model API contract
    -------------------------
    POST {model_api.predict_url}
    Body:    {"instances": [{"feat1": v, "feat2": v, ...}, ...]}
    Returns: {"predictions": [0, 1, ...], "probabilities": [[p0,p1], ...]}
             "probabilities" is optional — omit to skip ROC-AUC / AP metrics.
    """

    def __init__(
        self,
        config: ReporterConfig,
        db: Database | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.config = config
        self._loader = DataLoader(config.data)
        self._baseline: ModelEvaluationResult | None = None
        self._db = db
        self._alert_manager = alert_manager

    # ------------------------------------------------------------------
    # Pre-Production — 1: Data Evaluation
    # ------------------------------------------------------------------

    def evaluate_data(self, split: str = "test") -> DataEvaluationResult:
        """Statistical profile of a dataset split (train / test / val)."""
        df = self._loader.load_split(split)
        return self._evaluate_df(df, split)

    def _evaluate_df(self, df: pl.DataFrame, split_name: str) -> DataEvaluationResult:
        """Core statistical profiler — works with or without a target column."""
        target = self.config.target_column
        feature_cols = [c for c in df.columns if c != target]

        if target in df.columns:
            vc = df[target].value_counts().sort(target)
            dist: dict[str, int] = dict(zip(
                vc[target].cast(pl.String).to_list(),
                vc["count"].to_list(),
            ))
            counts = list(dist.values())
            imbalance = (
                round(max(counts) / min(counts), 4)
                if len(counts) > 1 and min(counts) > 0
                else 1.0
            )
        else:
            dist = {}
            imbalance = 0.0

        return DataEvaluationResult(
            split=split_name,
            n_rows=df.height,
            n_features=len(feature_cols),
            class_distribution=dist,
            imbalance_ratio=imbalance,
            duplicate_rows=int(df.is_duplicated().sum()),
            missing_cells=sum(df[col].null_count() for col in df.columns),
            features=[_compute_feature_stats(df[col]) for col in feature_cols],
        )

    def evaluate_all_data(self) -> dict[str, dict[str, Any]]:
        """
        Profile every data stage and split in one call.

        Returns a two-key dict:
          "balanced" → train / test / val (processed balanced splits)
          "raw"      → unsw_nb15 / ciciot2023 / combined (original CSVs)

        Raw stages can be slow — each source may be hundreds of CSV files.
        If a raw source is unavailable its entry contains {"error": "<message>"}.
        """
        balanced: dict[str, DataEvaluationResult] = {
            split: self.evaluate_data(split) for split in ("train", "test", "val")
        }
        if self._db is not None:
            for result in balanced.values():
                self._save_data_eval_to_db(result, stage="balanced")

        raw_sources: dict[str, Any] = {
            "unsw_nb15": self._loader.load_unsw_nb15,
            "ciciot2023": self._loader.load_ciciot2023,
            "combined": self._loader.load_raw,
        }
        raw: dict[str, Any] = {}
        for name, load_fn in raw_sources.items():
            try:
                result = self._evaluate_df(load_fn(), name)
                raw[name] = result
                if self._db is not None:
                    self._save_data_eval_to_db(result, stage="raw")
            except Exception as exc:
                raw[name] = {"error": str(exc)}

        return {"balanced": balanced, "raw": raw}

    # ------------------------------------------------------------------
    # Pre-Production — 2: Model Evaluation (via remote API)
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        df: pl.DataFrame | None = None,
        model_version: str | None = None,
    ) -> ModelEvaluationResult:
        """
        Send the test set to the remote model API and compute classification metrics.
        Stores the result as the production baseline for later decay tracking.
        Automatically persists the report to DuckDB if the service was given a db instance.

        df: optional labeled DataFrame (features + target column). When omitted the
            service tries to load the test parquet from disk; if unavailable it falls
            back to the latest PRE_PROD report stored in DuckDB.
        """
        if df is None:
            try:
                df = self._loader.load_test()
            except Exception:
                if self._db is not None:
                    result = self._load_latest_model_report()
                    self._baseline = result
                    return result
                raise
        feature_cols = _resolve_feature_cols(df, self.config)
        y_true: list[int] = df[self.config.target_column].to_list()

        api_response = self._call_predict(df.select(feature_cols).to_dicts())
        y_pred: list[int] = api_response["predictions"]
        y_prob: list[list[float]] | None = api_response.get("probabilities")

        result = self._build_classification_result(y_true, y_pred, y_prob)
        self._baseline = result

        if self._db is not None:
            model_version = model_version or self.config.model_api.model_version
            self.save_report_to_db(result, report_type="PRE_PROD", model_version=model_version)

        return result

    # ------------------------------------------------------------------
    # Production — 1: Ongoing Data Evaluation (drift)
    # ------------------------------------------------------------------

    def evaluate_data_drift(
        self,
        production_df: pl.DataFrame,
        model_version: str | None = None,
    ) -> DataDriftResult:
        """
        Compare production data against the reference distribution.

        When a Database is attached: persists the batch to production_log,
        then computes PSI against baselines stored in the feature_stats table
        (populated by scripts/init_baseline.py). No reference parquet needed.

        When no Database is attached (e.g. unit tests): falls back to loading
        the reference parquet via DataLoader and running PSI + KS / Chi².
        """
        model_version = model_version or self.config.model_api.model_version

        if self._db is not None:
            self._append_production_records(production_df)
            production_df = self._load_production_records()
            from services.drift_detector import DriftDetector as _DriftDetector
            detector = _DriftDetector(db=self._db, alert_manager=self._alert_manager)
            check = detector.check_drift(production_df)
            return self._map_detector_result(check, n_accumulated=production_df.height)

        reference_df = self._loader.load_reference()
        feature_cols = [
            col for col in _resolve_feature_cols(reference_df, self.config)
            if col in production_df.columns
        ]

        # Align production column dtypes to the reference so comparisons don't
        # blow up when JSON inference picks Float64 for a column that is Utf8 in
        # the parquet (e.g. proto sent as 6.0 but stored as "tcp").
        cast_exprs = [
            pl.col(col).cast(reference_df[col].dtype, strict=False)
            for col in feature_cols
            if production_df[col].dtype != reference_df[col].dtype
        ]
        if cast_exprs:
            production_df = production_df.with_columns(cast_exprs)

        t = self.config.thresholds
        # PSI requires a large enough window to be reliable. Below this threshold
        # PSI scores are still reported but capped at WARNING so small-batch noise
        # doesn't fire false CRITICAL alerts; KS / Chi² remain the authoritative
        # signal for small windows.
        _MIN_PSI_SAMPLES = 50
        psi_reliable = production_df.height >= _MIN_PSI_SAMPLES
        feature_results: list[FeatureDriftResult] = []

        for col in feature_cols:
            ref_s, prod_s = reference_df[col], production_df[col]
            is_cat = col in self.config.categorical_columns

            if is_cat:
                psi = _psi_categorical(ref_s, prod_s)
                _, pvalue = _chi2_test(ref_s, prod_s)
                psi_status = _status_from_psi(psi, t)
                if not psi_reliable and psi_status == DriftStatus.CRITICAL:
                    psi_status = DriftStatus.WARNING
                status = _max_status(
                    psi_status,
                    _status_from_pvalue(pvalue, t.chi2_pvalue_threshold),
                )
                feat_result = FeatureDriftResult(
                    feature=col, method="psi+chi2",
                    statistic=psi, pvalue=pvalue, status=status,
                )
            else:
                psi = _psi_numeric(ref_s, prod_s, self.config.psi_bins)
                _, pvalue = _ks_test(ref_s, prod_s)
                psi_status = _status_from_psi(psi, t)
                if not psi_reliable and psi_status == DriftStatus.CRITICAL:
                    psi_status = DriftStatus.WARNING
                status = _max_status(
                    psi_status,
                    _status_from_pvalue(pvalue, t.ks_pvalue_threshold),
                )
                feat_result = FeatureDriftResult(
                    feature=col, method="psi+ks",
                    statistic=psi, pvalue=pvalue, status=status,
                )

            _logger.info(
                "Drift check | feature={} | psi={:.6f} | pvalue={} | status={}",
                col,
                psi,
                f"{pvalue:.6f}" if pvalue is not None else "N/A",
                status.value,
            )

            if self._alert_manager is not None and status != DriftStatus.OK:
                self._alert_manager.trigger_alert(
                    severity="CRITICAL" if status == DriftStatus.CRITICAL else "WARNING",
                    event_type="DRIFT",
                    description=f"Feature '{col}' drift detected (PSI={psi:.4f})",
                    metadata={
                        "feature": col,
                        "psi": psi,
                        "pvalue": pvalue,
                        "model_version": model_version,
                    },
                )

            feature_results.append(feat_result)

        drifted = [r for r in feature_results if r.status != DriftStatus.OK]
        drift_rate = round(len(drifted) / len(feature_results), 4) if feature_results else 0.0
        overall = (
            _max_status(*(r.status for r in feature_results))
            if feature_results
            else DriftStatus.OK
        )

        return DataDriftResult(
            n_accumulated=production_df.height,
            n_features_checked=len(feature_results),
            n_drifted=len(drifted),
            drift_rate=drift_rate,
            overall_status=overall,
            features=feature_results,
        )

    # ------------------------------------------------------------------
    # Production — 2: Ongoing Model Evaluation
    # ------------------------------------------------------------------

    def evaluate_production_performance(
        self,
        predictions: list[int],
        ground_truth: list[int],
    ) -> ProductionModelEvaluationResult:
        """
        Compute aggregate and windowed metrics against the pre-production baseline.
        Positive decay values indicate degradation (baseline metric > current metric).
        Requires evaluate_model() to have been called first.
        """
        if self._baseline is None:
            raise RuntimeError(
                "No baseline found. Call evaluate_model() before evaluate_production_performance()."
            )

        y_true = np.array(ground_truth)
        y_pred = np.array(predictions)
        t = self.config.thresholds
        window_size = self.config.production_window_size
        n = len(y_true)

        current_acc = round(float(accuracy_score(y_true, y_pred)), 4)
        current_f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        decay_acc = round(self._baseline.accuracy - current_acc, 4)
        decay_f1 = round(self._baseline.f1 - current_f1, 4)

        windows: list[PerformanceWindow] = []
        for i, start in enumerate(range(0, n, window_size)):
            wt = y_true[start : start + window_size]
            wp = y_pred[start : start + window_size]
            if len(wt) < 2:
                continue
            w_acc = round(float(accuracy_score(wt, wp)), 4)
            w_f1 = round(float(f1_score(wt, wp, zero_division=0)), 4)
            w_decay_acc = round(self._baseline.accuracy - w_acc, 4)
            w_decay_f1 = round(self._baseline.f1 - w_f1, 4)
            windows.append(PerformanceWindow(
                window_index=i,
                n_samples=int(len(wt)),
                accuracy=w_acc,
                precision=round(float(precision_score(wt, wp, zero_division=0)), 4),
                recall=round(float(recall_score(wt, wp, zero_division=0)), 4),
                f1=w_f1,
                decay_accuracy=w_decay_acc,
                decay_f1=w_decay_f1,
                status=_status_from_decay(w_decay_acc, w_decay_f1, t),
            ))

        return ProductionModelEvaluationResult(
            n_samples=n,
            baseline_accuracy=self._baseline.accuracy,
            baseline_f1=self._baseline.f1,
            current_accuracy=current_acc,
            current_f1=current_f1,
            decay_accuracy=decay_acc,
            decay_f1=decay_f1,
            overall_status=_status_from_decay(decay_acc, decay_f1, t),
            windows=windows,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_data_eval_to_db(self, result: DataEvaluationResult, *, stage: str) -> str:
        """Persist a DataEvaluationResult to the reports table as report_type='DATA_EVAL'."""
        report_id = str(uuid.uuid4())
        metrics = json.dumps({
            "split": result.split,
            "stage": stage,
            "n_rows": result.n_rows,
            "n_features": result.n_features,
            "class_distribution": result.class_distribution,
            "imbalance_ratio": result.imbalance_ratio,
            "duplicate_rows": result.duplicate_rows,
            "missing_cells": result.missing_cells,
        })
        artifacts = json.dumps({
            "features": [f.model_dump() for f in result.features],
        })
        self._db.execute(
            "INSERT INTO reports (report_id, report_type, model_version, metrics, artifacts) "
            "VALUES (?, ?, ?, ?, ?)",
            [report_id, "DATA_EVAL", f"{stage}/{result.split}", metrics, artifacts],
        )
        return report_id

    def save_report_to_db(
        self,
        result: ModelEvaluationResult,
        *,
        report_type: str = "PRE_PROD",
        model_version: str | None = None,
    ) -> str:
        """
        Persist an evaluation result to the reports table.
        Returns the generated report_id (UUID string).
        Raises RuntimeError if no Database was provided at construction time.
        """
        if self._db is None:
            raise RuntimeError("No Database instance — pass db= when constructing ReporterService.")

        report_id = str(uuid.uuid4())
        metrics = json.dumps({
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
            "avg_precision": result.avg_precision,
        })
        artifacts = json.dumps({
            "confusion_matrix": result.confusion_matrix,
            "roc_curve_fpr": result.roc_curve_fpr,
            "roc_curve_tpr": result.roc_curve_tpr,
            "classification_report": result.report,
        })
        self._db.execute(
            "INSERT INTO reports (report_id, report_type, model_version, metrics, artifacts) "
            "VALUES (?, ?, ?, ?, ?)",
            [report_id, report_type, model_version, metrics, artifacts],
        )
        return report_id

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _map_detector_result(self, check, n_accumulated: int) -> DataDriftResult:
        """Convert a DriftCheckResult (DriftDetector) into DataDriftResult."""
        _s = {
            "OK": DriftStatus.OK,
            "WARNING": DriftStatus.WARNING,
            "CRITICAL": DriftStatus.CRITICAL,
        }
        n_checked = len(check.feature_results)
        n_drifted = len(check.drifted_features)
        return DataDriftResult(
            n_accumulated=n_accumulated,
            n_features_checked=n_checked,
            n_drifted=n_drifted,
            drift_rate=round(n_drifted / n_checked, 4) if n_checked > 0 else 0.0,
            overall_status=_s.get(check.status, DriftStatus.OK),
            features=[
                FeatureDriftResult(
                    feature=r.feature_name,
                    method="psi",
                    statistic=r.psi,
                    pvalue=None,
                    status=_s.get(r.status, DriftStatus.OK),
                )
                for r in check.feature_results
            ],
        )

    def _append_production_records(self, df: pl.DataFrame) -> None:
        """Persist each row of a production batch to the DuckDB production_log."""
        for row in df.to_dicts():
            self._db.execute(
                "INSERT INTO production_log (log_id, features) VALUES (?, ?)",
                [str(uuid.uuid4()), json.dumps(row)],
            )

    def _load_production_records(self) -> pl.DataFrame:
        """Load all accumulated production records from DuckDB as a Polars DataFrame."""
        rows = self._db.fetchall(
            "SELECT features FROM production_log ORDER BY received_at"
        )
        if not rows:
            return pl.DataFrame()
        return pl.from_dicts([json.loads(r["features"]) for r in rows])

    def reset_production_log(self) -> int:
        """Delete all accumulated production records. Returns the row count deleted."""
        count_row = self._db.fetchone("SELECT COUNT(*) AS n FROM production_log")
        n = int(count_row["n"]) if count_row else 0
        self._db.execute("DELETE FROM production_log")
        return n

    def _load_latest_model_report(self) -> ModelEvaluationResult:
        """Return the most-recent PRE_PROD report from DuckDB, or raise ValueError."""
        row = self._db.fetchone(
            "SELECT metrics, artifacts FROM reports "
            "WHERE report_type = 'PRE_PROD' ORDER BY timestamp DESC LIMIT 1"
        )
        if row is None:
            raise ValueError(
                "No test parquet on disk and no stored evaluation found in the database. "
                "POST labeled test records in the request body to run a fresh evaluation."
            )
        metrics = json.loads(row["metrics"]) if isinstance(row["metrics"], str) else row["metrics"]
        artifacts = json.loads(row["artifacts"]) if isinstance(row["artifacts"], str) else row["artifacts"]
        return ModelEvaluationResult(
            **metrics,
            confusion_matrix=artifacts["confusion_matrix"],
            roc_curve_fpr=artifacts.get("roc_curve_fpr", []),
            roc_curve_tpr=artifacts.get("roc_curve_tpr", []),
            report=artifacts["classification_report"],
        )

    def _call_predict(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.model_api.api_key:
            headers["Authorization"] = f"Bearer {self.config.model_api.api_key}"

        with httpx.Client(timeout=self.config.model_api.timeout_seconds) as client:
            response = client.post(
                self.config.model_api.predict_url,
                json={"instances": instances},
                headers=headers,
            )
        if not response.is_success:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )
        return response.json()

    def _build_classification_result(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_prob: list[list[float]] | None,
    ) -> ModelEvaluationResult:
        yt, yp = np.array(y_true), np.array(y_pred)

        roc_auc, avg_precision = 0.0, 0.0
        fpr_list: list[float] = []
        tpr_list: list[float] = []
        if y_prob is not None:
            probs = np.array(y_prob)
            pos_probs = probs[:, 1] if probs.ndim == 2 else probs
            roc_auc = round(float(roc_auc_score(yt, pos_probs)), 4)
            avg_precision = round(float(average_precision_score(yt, pos_probs)), 4)
            fpr, tpr, _ = roc_curve(yt, pos_probs)
            fpr_list = [round(float(v), 6) for v in fpr]
            tpr_list = [round(float(v), 6) for v in tpr]

        return ModelEvaluationResult(
            accuracy=round(float(accuracy_score(yt, yp)), 4),
            precision=round(float(precision_score(yt, yp, zero_division=0)), 4),
            recall=round(float(recall_score(yt, yp, zero_division=0)), 4),
            f1=round(float(f1_score(yt, yp, zero_division=0)), 4),
            roc_auc=roc_auc,
            avg_precision=avg_precision,
            confusion_matrix=confusion_matrix(yt, yp).tolist(),
            roc_curve_fpr=fpr_list,
            roc_curve_tpr=tpr_list,
            report=sk_classification_report(yt, yp, zero_division=0),
        )
