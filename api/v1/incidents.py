"""Incidents routes — alerting events raised by drift / performance monitors."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
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


@router.get("/incidents", response_model=list[IncidentRecord])
def list_incidents():
    """Return the 50 most recent incidents."""
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
