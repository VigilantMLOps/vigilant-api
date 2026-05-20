"""Performance monitoring service — real-time accuracy and F1 checks with alerting."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from core.logger import get_logger
from .alerting_engine import AlertManager

_logger = get_logger("vigilant.performance")


# ===========================================================================
# Configuration
# ===========================================================================


class PerformanceThresholds(BaseModel):
    accuracy_critical: float = 0.85
    f1_warning_drop: float = 0.03  # absolute F1 drop vs previous batch triggers WARNING


# ===========================================================================
# Result schema
# ===========================================================================


class PerformanceCheckResult(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    f1_previous: float | None = None
    f1_delta: float | None = None       # negative means degradation
    status: str                          # "OK" | "WARNING" | "CRITICAL"
    alerts_fired: list[str] = Field(default_factory=list)


# ===========================================================================
# Service
# ===========================================================================


class PerformanceService:
    """
    Checks live prediction batches for accuracy and F1 regressions.

    Maintains an internal F1 history so downward trends are detected across
    successive calls. Wire it up with an AlertManager to push incidents to
    DuckDB and the notification log.

    Usage
    -----
    svc = PerformanceService(alert_manager=alert_manager)
    result = svc.check_model_performance(y_true, y_pred)
    """

    def __init__(
        self,
        thresholds: PerformanceThresholds | None = None,
        db: "Database | None" = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.thresholds = thresholds or PerformanceThresholds()
        self._db = db
        self._alert_manager = alert_manager
        self._f1_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_model_performance(
        self,
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> PerformanceCheckResult:
        """
        Evaluate a prediction batch and fire alerts for degradation.

        Accuracy guard  — CRITICAL when accuracy falls below the configured
                          threshold (default 0.85).
        F1 trend guard  — WARNING when F1 drops more than `f1_warning_drop`
                          (default 0.03) compared to the previous batch.

        Returns a PerformanceCheckResult with current metrics, delta, status,
        and any incident IDs that were created.
        """
        yt = np.array(y_true)
        yp = np.array(y_pred)
        metadata = metadata or {}

        acc = round(float(accuracy_score(yt, yp)), 4)
        prec = round(float(precision_score(yt, yp, zero_division=0)), 4)
        rec = round(float(recall_score(yt, yp, zero_division=0)), 4)
        f1 = round(float(f1_score(yt, yp, zero_division=0)), 4)

        _logger.info(
            "Performance check | acc={} | prec={} | rec={} | f1={}",
            acc, prec, rec, f1,
        )

        alerts_fired: list[str] = []
        worst_status = "OK"

        # --- Accuracy guard ------------------------------------------------
        if acc < self.thresholds.accuracy_critical:
            worst_status = "CRITICAL"
            incident_id = self._fire_alert(
                severity="CRITICAL",
                description=(
                    f"Model accuracy dropped to {acc:.4f} — "
                    f"below critical threshold {self.thresholds.accuracy_critical}"
                ),
                extra={"accuracy": acc, "threshold": self.thresholds.accuracy_critical, **metadata},
            )
            if incident_id:
                alerts_fired.append(incident_id)

        # --- F1 trend guard ------------------------------------------------
        f1_previous: float | None = self._f1_history[-1] if self._f1_history else None
        f1_delta: float | None = None

        if f1_previous is not None:
            f1_delta = round(f1 - f1_previous, 4)
            drop = f1_previous - f1                 # positive = degradation
            if drop >= self.thresholds.f1_warning_drop:
                if worst_status != "CRITICAL":
                    worst_status = "WARNING"
                incident_id = self._fire_alert(
                    severity="WARNING",
                    description=(
                        f"F1-score fell from {f1_previous:.4f} to {f1:.4f} "
                        f"(drop={drop:.4f}) — significant downward trend detected"
                    ),
                    extra={
                        "f1_current": f1,
                        "f1_previous": f1_previous,
                        "f1_drop": round(drop, 4),
                        **metadata,
                    },
                )
                if incident_id:
                    alerts_fired.append(incident_id)

        self._f1_history.append(f1)

        return PerformanceCheckResult(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            f1_previous=f1_previous,
            f1_delta=f1_delta,
            status=worst_status,
            alerts_fired=alerts_fired,
        )

    def reset_history(self) -> None:
        """Clear the accumulated F1 history (e.g. when starting a new evaluation period)."""
        self._f1_history.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fire_alert(
        self,
        severity: str,
        description: str,
        extra: dict[str, Any],
    ) -> str | None:
        _logger.warning("ALERT:{} | {}", severity, description)
        if self._alert_manager is None:
            return None
        return self._alert_manager.trigger_alert(
            severity=severity,
            event_type="PERFORMANCE",
            description=description,
            metadata=extra,
        )
