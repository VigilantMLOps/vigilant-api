"""Monitoring routes — evaluation report history for the React dashboard."""
from __future__ import annotations

import json as _json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.database import db

router = APIRouter()


class ReportRecord(BaseModel):
    report_id: str
    timestamp: datetime
    report_type: str
    model_version: str | None = None
    metrics: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None


def _parse_json_fields(row: dict) -> dict:
    """Deserialize JSON string columns (metrics, artifacts) into Python dicts."""
    for field in ("metrics", "artifacts"):
        value = row.get(field)
        if isinstance(value, str):
            try:
                row[field] = _json.loads(value)
            except _json.JSONDecodeError:
                pass
    return row


@router.get("/reports/latest", response_model=ReportRecord)
def get_latest_report():
    """Return the most recent evaluation report."""
    row = db.fetchone(
        "SELECT * FROM reports ORDER BY timestamp DESC LIMIT 1"
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No reports found.")
    return _parse_json_fields(row)


@router.get("/reports/history", response_model=list[ReportRecord])
def get_report_history():
    """Return the last 10 evaluation reports for trend analysis."""
    rows = db.fetchall(
        "SELECT * FROM reports ORDER BY timestamp DESC LIMIT 10"
    )
    return [_parse_json_fields(row) for row in rows]
