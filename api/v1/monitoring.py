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


class ModelVersionEntry(BaseModel):
    version: str
    label: str


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
    """Return the last 100 evaluation reports (PRE_PROD + DATA_EVAL) for the dashboard."""
    rows = db.fetchall(
        "SELECT report_id, timestamp, report_type, model_id, model_version, content"
        " FROM reports WHERE report_type IN ('PRE_PROD', 'DATA_EVAL')"
        " ORDER BY timestamp DESC LIMIT 100"
    )
    return [_parse_content(row) for row in rows]


@router.get("/reports/model-versions", response_model=list[ModelVersionEntry])
def get_model_versions():
    """Return distinct PRE_PROD model versions with human-readable labels.

    Uses the models.display_name when the report links to a known model row;
    falls back to the raw model_version string so unregistered pushes still appear.
    """
    rows = db.fetchall(
        "SELECT DISTINCT ON (r.model_version)"
        "   r.model_version,"
        "   COALESCE(m.display_name, r.model_version) AS label,"
        "   r.timestamp"
        " FROM reports r"
        " LEFT JOIN models m ON m.model_id = r.model_id"
        " WHERE r.report_type = 'PRE_PROD' AND r.model_version IS NOT NULL"
        " ORDER BY r.model_version, r.timestamp DESC"
    )
    return [{"version": row["model_version"], "label": row["label"]} for row in rows]
