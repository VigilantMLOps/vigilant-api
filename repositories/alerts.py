from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database


class AlertRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def create(
        self,
        alert_id: str,
        level: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any],
    ) -> None:
        self._db.execute(
            "INSERT INTO alerts (alert_id, level, event_type, message, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            [alert_id, level, event_type, message, json.dumps(metadata)],
        )