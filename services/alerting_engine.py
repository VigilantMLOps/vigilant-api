"""Alerting engine — incident lifecycle management backed by PostgreSQL."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database

from core.logger import get_logger
from repositories import IncidentRepository
from services.notification_service import AlertLevel, send_alert

_logger = get_logger("vigilant.alerting")


class AlertManager:
    """Translates drift/performance signals into incidents and notifications.

    Writes a record to the `incidents` table (queryable via /api/v1/incidents)
    and dispatches a visible log entry + `alerts` table row via send_alert().
    """

    def __init__(self, db: Database | None = None) -> None:
        self._incident_repo = IncidentRepository(db) if db is not None else None

    def trigger_alert(
        self,
        severity: str,
        event_type: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create an incident and fire a notification. Returns the incident_id."""
        incident_id = str(uuid.uuid4())
        metadata = metadata or {}
        severity_upper = severity.upper()

        if self._incident_repo is not None:
            self._incident_repo.create(incident_id, severity_upper, event_type, description)
            _logger.debug("Incident persisted | id={} | type={}", incident_id, event_type)

        send_alert(
            message=description,
            level=_to_alert_level(severity_upper),
            metadata={"incident_id": incident_id, "event_type": event_type, **metadata},
        )

        return incident_id


def _to_alert_level(severity: str) -> AlertLevel:
    return {
        "CRITICAL": AlertLevel.CRITICAL,
        "WARNING": AlertLevel.WARNING,
    }.get(severity, AlertLevel.INFO)
