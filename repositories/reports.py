from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database


class ReportRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def insert(
        self,
        report_id: str,
        report_type: str,
        content: dict[str, Any],
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Single entry point for all report types. Writes the row to
        PostgreSQL `reports` and mirrors it to ClickHouse `report_metrics`
        (analytics layer). The payload shape is the caller's contract per
        report_type.

        `model_id=None` defers to the database's resolved default model.
        """
        model_id = model_id or self._db.default_model_id
        content_json = json.dumps(content)
        self._db.execute(
            "INSERT INTO reports (report_id, report_type, model_id, model_version, content)"
            " VALUES (?, ?, ?, ?, ?)",
            [report_id, report_type, model_id, model_version, content_json],
        )
        self._db.execute(
            "INSERT INTO report_metrics "
            "(report_id, report_type, model_id, model_version, content)"
            " VALUES (?, ?, ?, ?, ?)",
            [report_id, report_type, model_id, model_version or "", content_json],
        )

    def insert_model_eval(
        self,
        report_id: str,
        report_type: str,
        content: dict[str, Any],
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Insert a PRE_PROD (or other model-eval) report.

        `content` shape (PRE_PROD): {accuracy, precision, recall, f1, roc_auc,
        avg_precision, confusion_matrix, roc_curve_fpr, roc_curve_tpr,
        classification_report, …}
        """
        self.insert(report_id, report_type, content, model_id, model_version)

    def insert_data_eval(
        self,
        report_id: str,
        content: dict[str, Any],
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Insert a DATA_EVAL report.

        `content` shape: {split, stage, n_rows, n_features, imbalance_ratio,
        duplicate_rows, missing_cells, class_distribution, features, …}
        """
        self.insert(report_id, "DATA_EVAL", content, model_id, model_version)

    def fetch_latest_pre_prod(self) -> dict | None:
        return self._db.fetchone(
            "SELECT content FROM reports "
            "WHERE report_type = 'PRE_PROD' ORDER BY timestamp DESC LIMIT 1"
        )