"""Incidents routes — alerting events raised by drift / performance monitors."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.database import db

router = APIRouter()


class IncidentRecord(BaseModel):
    incident_id: str
    timestamp: datetime
    severity: str
    incident_type: str
    description: str | None = None
    status: str
    model_id: str | None = None


@router.get("/incidents", response_model=list[IncidentRecord])
def list_incidents(model_version: str | None = Query(None)):
    """Return the 50 most recent incidents, optionally filtered by model_version.

    `model_version` is the report-level version string (e.g. "1.0.0" or a
    vigilant-detect model ID).  The endpoint resolves it to the internal
    model_id via the reports table so callers never need to know UUIDs.
    """
    if model_version:
        row = db.fetchone(
            "SELECT model_id FROM reports"
            " WHERE model_version = ? AND model_id IS NOT NULL"
            " ORDER BY timestamp DESC LIMIT 1",
            [model_version],
        )
        if row and row.get("model_id"):
            return db.fetchall(
                "SELECT * FROM incidents WHERE model_id = ?"
                " ORDER BY timestamp DESC LIMIT 50",
                [row["model_id"]],
            )
        return []
    return db.fetchall(
        "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 50"
    )


@router.get("/incidents/{incident_id}", response_model=IncidentRecord)
def get_incident(incident_id: str):
    """Return a single incident by ID."""
    row = db.fetchone(
        "SELECT * FROM incidents WHERE incident_id = ?",
        [incident_id],
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return row
