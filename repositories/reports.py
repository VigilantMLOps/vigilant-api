from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database


class ReportRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def insert_model_eval(
        self,
        report_id: str,
        report_type: str,
        model_version: str | None,
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        roc_auc: float,
        avg_precision: float,
    ) -> None:
        self._db.execute(
            "INSERT INTO reports "
            "(report_id, report_type, model_version, metrics, artifacts, "
            "accuracy, precision_score, recall, f1_score, roc_auc, avg_precision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                report_id, report_type, model_version,
                json.dumps(metrics), json.dumps(artifacts),
                accuracy, precision, recall, f1, roc_auc, avg_precision,
            ],
        )
        self._db.execute(
            "INSERT INTO report_metrics "
            "(report_id, report_type, model_id, model_version, "
            "accuracy, precision_score, recall, f1_score, roc_auc, avg_precision) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                report_id, report_type, "default", model_version or "",
                accuracy or 0.0, precision or 0.0, recall or 0.0,
                f1 or 0.0, roc_auc or 0.0, avg_precision or 0.0,
            ],
        )

    def insert_data_eval(
        self,
        report_id: str,
        model_version: str,
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
        split: str,
        stage: str,
        n_rows: int,
        n_features: int,
        imbalance_ratio: float,
        duplicate_rows: int,
        missing_cells: int,
        class_distribution: dict[str, int],
    ) -> None:
        self._db.execute(
            "INSERT INTO reports "
            "(report_id, report_type, model_version, metrics, artifacts, "
            "split, stage, n_rows, n_features, imbalance_ratio, duplicate_rows, missing_cells, class_distribution) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                report_id, "DATA_EVAL", model_version,
                json.dumps(metrics), json.dumps(artifacts),
                split, stage, n_rows, n_features,
                imbalance_ratio, duplicate_rows, missing_cells,
                json.dumps(class_distribution),
            ],
        )
        self._db.execute(
            "INSERT INTO report_metrics "
            "(report_id, report_type, model_id, model_version, "
            "n_rows, n_features, imbalance_ratio, duplicate_rows, missing_cells) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                report_id, "DATA_EVAL", "default", model_version,
                n_rows or 0, n_features or 0, imbalance_ratio or 0.0,
                duplicate_rows or 0, missing_cells or 0,
            ],
        )

    def fetch_latest_pre_prod(self) -> dict | None:
        return self._db.fetchone(
            "SELECT metrics, artifacts FROM reports "
            "WHERE report_type = 'PRE_PROD' ORDER BY timestamp DESC LIMIT 1"
        )