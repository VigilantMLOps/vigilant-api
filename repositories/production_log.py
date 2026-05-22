from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database


class ProductionLogRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def append(self, log_id: str, features_json: str) -> None:
        self._db.execute(
            "INSERT INTO production_log_buffer (log_id, features) VALUES (?, ?)",
            [log_id, features_json],
        )

    def fetch_all(self) -> list[dict]:
        return self._db.fetchall("SELECT features FROM production_log ORDER BY received_at")

    def count(self) -> int:
        row = self._db.fetchone("SELECT COUNT(*) AS n FROM production_log")
        return int(row["n"]) if row else 0

    def reset(self) -> None:
        self._db.execute("DELETE FROM production_log")