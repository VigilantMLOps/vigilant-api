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
    model_id: str | None = None
    model_version: str | None = None
    content: dict[str, Any] | None = None


def _parse_content(row: dict) -> dict:
    value = row.get("content")
    if isinstance(value, str):
        try:
            row["content"] = _json.loads(value)
        except _json.JSONDecodeError:
            pass
    return row


@router.get("/reports/latest", response_model=ReportRecord)
def get_latest_report():
    """Return the most recent evaluation report."""
    row = db.fetchone(
        "SELECT report_id, timestamp, report_type, model_id, model_version, content"
        " FROM reports ORDER BY timestamp DESC LIMIT 1"
    )
    if row is None:
        raise HTTPException(status_code=404, detail="No reports found.")
    return _parse_content(row)


@router.get("/reports/history", response_model=list[ReportRecord])
def get_report_history():
    """Return the last 10 evaluation reports for trend analysis."""
    rows = db.fetchall(
        "SELECT report_id, timestamp, report_type, model_id, model_version, content"
        " FROM reports ORDER BY timestamp DESC LIMIT 10"
    )
    return [_parse_content(row) for row in rows]
