"""Drift detection service — PSI-based checks against PostgreSQL-stored baselines."""
from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from core.logger import get_logger
from repositories import DriftResultRepository, FeatureStatsRepository
from services.alerting_engine import AlertManager

_logger = get_logger("vigilant.drift")

_PSI_WARNING = 0.10
_PSI_CRITICAL = 0.25
_EPSILON = 1e-8


class FeatureDriftResult(BaseModel):
    feature_name: str
    psi: float
    status: str  # OK | WARNING | CRITICAL
    details: dict[str, Any] = Field(default_factory=dict)


class DriftCheckResult(BaseModel):
    status: str  # OK | WARNING | CRITICAL
    drifted_features: list[str] = Field(default_factory=list)
    feature_results: list[FeatureDriftResult] = Field(default_factory=list)
    alerts_fired: list[str] = Field(default_factory=list)


class DriftDetector:
    """
    Computes PSI between incoming production batches and stored reference baselines.

    Reference distributions live in the PostgreSQL `feature_stats` table.
    Raises ValueError when the table is empty — run scripts/init_baseline.py first.

    PSI thresholds:  < 0.10 → OK  |  0.10–0.25 → WARNING  |  ≥ 0.25 → CRITICAL
    """

    def __init__(
        self,
        db: "Database | None" = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self._db = db
        self._alert_manager = alert_manager
        self._feature_stats_repo = FeatureStatsRepository(db) if db is not None else None
        self._drift_result_repo = DriftResultRepository(db) if db is not None else None

    def check_drift(self, current_df: pl.DataFrame) -> DriftCheckResult:
        """Main entry point — fetches baseline from PostgreSQL then runs drift calculation.

        Raises ValueError if the feature_stats table is empty.
        """
        baseline_stats = self._fetch_baseline_stats()
        _logger.info(
            "Drift check | n_baseline_features={} | n_production_rows={}",
            len(baseline_stats),
            len(current_df),
        )
        return self.calculate_drift(current_df, baseline_stats)

    def calculate_drift(
        self, current_df: pl.DataFrame, baseline_stats: dict[str, dict]
    ) -> DriftCheckResult:
        """Compare current_df against pre-loaded baseline_stats.

        baseline_stats: mapping of feature_name → stats dict as stored in PostgreSQL.
        """
        feature_results: list[FeatureDriftResult] = []
        alerts_fired: list[str] = []
        worst_status = "OK"

        for feature_name, stats in baseline_stats.items():
            if feature_name not in current_df.columns:
                _logger.debug("Feature '{}' absent from batch — skipping", feature_name)
                continue

            feature_type = stats.get("type", "numeric")

            if feature_type == "numeric":
                arr = (
                    current_df[feature_name]
                    .cast(pl.Float64, strict=False)
                    .fill_null(0.0)
                    .to_numpy()
                )
                hist = stats["histogram"]
                psi = self._psi_numeric(hist["bins"], hist["counts"], arr)
                details: dict[str, Any] = {
                    "batch_mean": round(float(np.mean(arr)), 6),
                    "batch_std": round(float(np.std(arr)), 6),
                }
            else:
                col = current_df[feature_name].cast(pl.Utf8, strict=False)
                values = [str(v) for v in col.to_list() if v is not None]
                psi = self._psi_categorical(stats["distribution"], values)
                details = {}

            status = _status_from_psi(psi)
            _logger.info(
                "Drift | feature={} | psi={:.6f} | status={}", feature_name, psi, status
            )

            if self._drift_result_repo is not None:
                self._drift_result_repo.insert(
                    str(uuid.uuid4()), feature_name, self._db.default_model_id,
                    "psi", psi, status, len(current_df),
                )

            if status != "OK":
                if _severity_rank(status) > _severity_rank(worst_status):
                    worst_status = status
                incident_id = self._fire_alert(
                    severity=status, feature_name=feature_name, psi=psi
                )
                if incident_id:
                    alerts_fired.append(incident_id)

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feature_name,
                    psi=psi,
                    status=status,
                    details=details,
                )
            )

        drifted = [r.feature_name for r in feature_results if r.status != "OK"]
        return DriftCheckResult(
            status=worst_status,
            drifted_features=drifted,
            feature_results=feature_results,
            alerts_fired=alerts_fired,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_baseline_stats(self) -> dict[str, dict]:
        """Fetch per-feature stats from the feature_stats table.

        Raises ValueError if the table is empty.
        Raises RuntimeError if no Database was provided at construction time.
        """
        if self._feature_stats_repo is None:
            raise RuntimeError("DriftDetector requires a Database instance.")
        result = self._feature_stats_repo.fetch_all()
        if not result:
            raise ValueError(
                "Baseline statistics not found in database. Please run init_baseline.py first."
            )
        return result

    @staticmethod
    def _psi_numeric(
        bin_edges: list[float], ref_counts: list[int], values: np.ndarray
    ) -> float:
        """PSI for numeric features using stored bin_edges from the baseline histogram."""
        edges = np.array(bin_edges)
        ref_pct = np.array(ref_counts, dtype=float)
        ref_pct /= ref_pct.sum() + _EPSILON

        clipped = np.clip(values, edges[0], edges[-1])
        cur_counts, _ = np.histogram(clipped, bins=edges)
        cur_pct = cur_counts.astype(float) / (cur_counts.sum() + _EPSILON)

        ref_pct = np.clip(ref_pct, _EPSILON, None)
        cur_pct = np.clip(cur_pct, _EPSILON, None)

        return round(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 6)

    @staticmethod
    def _psi_categorical(ref_dist: dict[str, float], values: list[str]) -> float:
        """PSI for categorical features — compares observed frequencies against stored percentages."""
        total = len(values)
        if total == 0:
            return 0.0
        cur_counts: dict[str, int] = {}
        for v in values:
            cur_counts[v] = cur_counts.get(v, 0) + 1

        psi = 0.0
        for cat in set(ref_dist) | set(cur_counts):
            ref_pct = max(ref_dist.get(cat, 0.0), _EPSILON)
            cur_pct = max(cur_counts.get(cat, 0) / total, _EPSILON)
            psi += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
        return round(psi, 6)

    def _fire_alert(self, severity: str, feature_name: str, psi: float) -> str | None:
        description = (
            f"Data drift detected in '{feature_name}': PSI={psi:.4f} "
            f"({'CRITICAL' if psi >= _PSI_CRITICAL else 'WARNING'} threshold exceeded)"
        )
        _logger.warning("DRIFT:{} | {}", severity, description)
        if self._alert_manager is None:
            return None
        return self._alert_manager.trigger_alert(
            severity=severity,
            event_type="DRIFT",
            description=description,
            metadata={"feature": feature_name, "psi": psi},
        )


def _status_from_psi(psi: float) -> str:
    if psi >= _PSI_CRITICAL:
        return "CRITICAL"
    if psi >= _PSI_WARNING:
        return "WARNING"
    return "OK"


def _severity_rank(s: str) -> int:
    return {"OK": 0, "WARNING": 1, "CRITICAL": 2}.get(s, 0)
