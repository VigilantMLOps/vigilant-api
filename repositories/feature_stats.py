from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database


class FeatureStatsRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def fetch_all(self) -> dict[str, dict]:
        """Return all baseline stats keyed by feature_name."""
        rows = self._db.fetchall("SELECT feature_name, stats_json FROM feature_stats")
        result: dict[str, dict] = {}
        for row in rows:
            stats = row["stats_json"]
            if isinstance(stats, str):
                stats = json.loads(stats)
            result[row["feature_name"]] = stats
        return result