from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database


class DriftResultRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def insert(
        self,
        drift_id: str,
        feature_name: str,
        model_id: str,
        method: str,
        psi_score: float,
        status: str,
        n_production_rows: int,
    ) -> None:
        self._db.execute(
            "INSERT INTO drift_results "
            "(drift_id, feature_name, model_id, method, psi_score, status, n_production_rows) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [drift_id, feature_name, model_id, method, psi_score, status, n_production_rows],
        )