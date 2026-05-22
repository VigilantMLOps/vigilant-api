from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database


class IncidentRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def create(self, incident_id: str, severity: str, incident_type: str, description: str) -> None:
        self._db.execute(
            "INSERT INTO incidents (incident_id, severity, incident_type, description) "
            "VALUES (?, ?, ?, ?)",
            [incident_id, severity, incident_type, description],
        )

    def update_status(self, incident_id: str, status: str) -> None:
        self._db.execute(
            "UPDATE incidents SET status = ? WHERE incident_id = ?",
            [status, incident_id],
        )