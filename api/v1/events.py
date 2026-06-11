"""Ingestion layer — POST /events/login.

Receives a login event from the caller, forwards it synchronously to
vigilant-detect for scoring, logs the raw event + decision to ClickHouse,
and returns the decision to the caller.

Hot path (synchronous, caller blocks):
    1. Validate schema (Pydantic)
    2. POST to vigilant-detect /predict  (httpx, timeout 5s)
    3. Return {event_id, decision, risk_score, context_flags}

Background (after response is sent):
    4. Insert raw event + decision into ClickHouse login_events
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from core.database import db
from core.logger import get_logger

_logger = get_logger("vigilant.events")
router = APIRouter()

_DETECT_URL = os.getenv("MODEL_API_URL", "http://vigilant-detect:8001")
_TIMEOUT = 5.0  # seconds — hard cap on vigilant-detect call


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class LoginEvent(BaseModel):
    event_id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id:            str
    session_id:         str = ""
    ip_address:         str = ""
    geo_country:        str = ""
    geo_lat:            float | None = None
    geo_lon:            float | None = None
    user_agent:         str = ""
    device_fingerprint: str = ""
    login_success:      bool = True
    mfa_used:           bool = False
    mfa_method:         Literal["none", "totp", "sms", "email"] = "none"
    login_duration_ms:  float | None = None
    # Pre-computed offline features (passed through to vigilant-detect)
    failed_attempts_7d:      float | None = None
    distinct_ips_7d:         float | None = None
    login_success_rate_30d:  float | None = None
    avg_login_hour_7d:       float | None = None
    account_age_days:        float | None = None
    # Online feature overrides (passed through to vigilant-detect)
    last_login_gap_h:        float | None = None
    geo_distance_delta:      float | None = None


class LoginDecision(BaseModel):
    event_id:            str
    decision:            Literal["ALLOW", "CHALLENGE", "BLOCK"]
    risk_score:          float
    context_flags:       list[str]
    model_version:       str
    degraded:            bool


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/events/login", response_model=LoginDecision, tags=["Events"])
def ingest_login(event: LoginEvent, background_tasks: BackgroundTasks) -> LoginDecision:
    """
    Score a login event and return an ALLOW / CHALLENGE / BLOCK decision.

    Synchronous: caller blocks until vigilant-detect responds (P95 < 50ms).
    Logging to ClickHouse happens in the background after the response is sent.
    """
    start = time.perf_counter()

    payload = event.model_dump(mode="json")

    try:
        resp = httpx.post(
            f"{_DETECT_URL}/predict",
            json=payload,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
    except httpx.TimeoutException:
        _logger.warning("vigilant-detect /predict timed out for event {}", event.event_id)
        raise HTTPException(status_code=504, detail="Model service timed out.")
    except httpx.HTTPStatusError as exc:
        _logger.error("vigilant-detect returned {}: {}", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=502, detail="Model service error.")
    except httpx.RequestError as exc:
        _logger.error("vigilant-detect unreachable: {}", exc)
        raise HTTPException(status_code=503, detail="Model service unreachable.")

    latency_ms = (time.perf_counter() - start) * 1_000

    decision = LoginDecision(
        event_id=result.get("event_id", event.event_id),
        decision=result["decision"],
        risk_score=result["risk_score"],
        context_flags=result.get("context_flags", []),
        model_version=result.get("model_version", ""),
        degraded=result.get("degraded", False),
    )

    background_tasks.add_task(_log_to_clickhouse, event, decision, latency_ms)

    return decision


# ---------------------------------------------------------------------------
# Background logging
# ---------------------------------------------------------------------------

def _log_to_clickhouse(event: LoginEvent, decision: LoginDecision, latency_ms: float) -> None:
    try:
        db.execute(
            "INSERT INTO login_events "
            "(event_id, received_at, user_id, session_id, ip_address, geo_country, "
            "geo_lat, geo_lon, device_fingerprint, login_success, mfa_used, mfa_method, "
            "login_duration_ms, decision, risk_score, context_flags, model_version, latency_ms) "
            "VALUES (?, now64(3), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                event.event_id,
                event.user_id,
                event.session_id,
                event.ip_address,
                event.geo_country,
                event.geo_lat,
                event.geo_lon,
                event.device_fingerprint,
                int(event.login_success),
                int(event.mfa_used),
                event.mfa_method,
                event.login_duration_ms,
                decision.decision,
                round(decision.risk_score, 4),
                json.dumps(decision.context_flags),
                decision.model_version,
                round(latency_ms, 2),
            ],
        )
    except Exception as exc:
        _logger.warning("ClickHouse login_events insert failed (non-fatal): {}", exc)
