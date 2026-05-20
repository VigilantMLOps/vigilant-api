"""Telemetry routes — placeholder for real-time metrics ingestion."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class TelemetryStatusResponse(BaseModel):
    status: str
    message: str


@router.get("/telemetry/status", response_model=TelemetryStatusResponse)
def telemetry_status():
    return {"status": "ok", "message": "Telemetry endpoint not yet implemented."}
