"""Dispatcher Service — procedure-driven incident remediation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from core.database import db
from core.logger import get_logger
from repositories import IncidentRepository

_logger = get_logger("vigilant.dispatcher")
_PROCEDURES_PATH = Path(__file__).parent.parent / "core" / "procedures.yaml"


@dataclass(frozen=True)
class Procedure:
    action: str
    risk: str
    auto_trigger: bool


class DispatcherService:
    """Routes incidents to automated remediation or human escalation based on procedures.yaml."""

    def __init__(self, procedures_path: Path = _PROCEDURES_PATH) -> None:
        self._procedures: dict[str, Procedure] = self._load_procedures(procedures_path)
        _logger.info(
            "DispatcherService initialized | {} procedures loaded",
            len(self._procedures),
        )

    def _load_procedures(self, path: Path) -> dict[str, Procedure]:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return {
            key: Procedure(
                action=values["action"],
                risk=values["risk"],
                auto_trigger=values["auto_trigger"],
            )
            for key, values in raw["procedures"].items()
        }

    def process_incident(self, incident_id: str, event_type: str) -> None:
        """Dispatch an incident based on its event_type procedure.

        Updates the incident status in PostgreSQL and logs the outcome.
        """
        procedure = self._procedures.get(event_type)
        if procedure is None:
            _logger.warning(
                "No procedure found for event_type='{}'; skipping dispatch | incident_id={}",
                event_type,
                incident_id,
            )
            return

        if procedure.auto_trigger:
            self._execute_action(procedure.action, incident_id, event_type)
            self._update_status(incident_id, "AUTO_RESOLVED")
            _logger.info(
                "Auto-resolved low-risk incident | incident_id={} event_type={} action={}",
                incident_id,
                event_type,
                procedure.action,
            )
        else:
            self._update_status(incident_id, "ESCALATED")
            _logger.warning(
                "Escalated high-risk incident to human review | incident_id={} event_type={} risk={}",
                incident_id,
                event_type,
                procedure.risk,
            )

    def _execute_action(self, action: str, incident_id: str, event_type: str) -> None:
        handlers = {
            "REFETCH_DB": self._refetch_db,
            "REFETCH_SCHEMA": self._refetch_schema,
        }
        handler = handlers.get(action)
        if handler is None:
            _logger.warning(
                "Unknown action '{}' for incident_id={} event_type={}",
                action,
                incident_id,
                event_type,
            )
            return
        handler()

    def _refetch_db(self) -> None:
        """Re-establish both database connections to recover from a stale state."""
        db.shutdown()
        db.startup()
        _logger.info("REFETCH_DB: database connections refreshed")

    def _refetch_schema(self) -> None:
        """Reload the ML feature schema from disk to clear any cached skew."""
        from core.ml_engine.engine import SchemaValidator
        SchemaValidator()
        _logger.info("REFETCH_SCHEMA: feature schema reloaded from disk")

    def _update_status(self, incident_id: str, status: str) -> None:
        IncidentRepository(db).update_status(incident_id, status)
