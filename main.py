"""VigilantMLOps — FastAPI application entry point.

Run with:
    uvicorn main:app --reload
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.database import db
from core.logger import get_logger
from services.alerting_engine import AlertManager
from api.v1 import incidents, monitoring, reporter, telemetry, events

logger = get_logger("vigilant.app")
_logger = get_logger("vigilant.middleware")
_alert_manager = AlertManager(db=db)

_LATENCY_WARNING_MS = 500.0


class HealthResponse(BaseModel):
    status: str
    service: str


class SystemHealthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        _logger.debug(
            "Request {} {} completed in {:.1f}ms — status {}",
            request.method,
            request.url.path,
            elapsed_ms,
            response.status_code,
        )

        if elapsed_ms > _LATENCY_WARNING_MS:
            _alert_manager.trigger_alert(
                severity="WARNING",
                event_type="SYSTEM",
                description=(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {elapsed_ms:.1f}ms (threshold {_LATENCY_WARNING_MS:.0f}ms)"
                ),
                metadata={"latency_ms": round(elapsed_ms, 2), "path": request.url.path},
            )

        if response.status_code >= 500:
            _alert_manager.trigger_alert(
                severity="CRITICAL",
                event_type="SYSTEM",
                description=(
                    f"Server error: {request.method} {request.url.path} "
                    f"returned HTTP {response.status_code}"
                ),
                metadata={"status_code": response.status_code, "path": request.url.path},
            )

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VigilantMLOps Backend is starting up...")
    db.startup()
    yield
    db.shutdown()
    logger.info("VigilantMLOps API shut down.")


app = FastAPI(
    title="VigilantMLOps API",
    description=(
        "Production-grade MLOps monitoring and observability platform for binary classification models. "
        "Provides endpoints for pre-production model evaluation, real-time data drift detection (PSI/KS/Chi²), "
        "incident management, and system health monitoring."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",")] if _cors_origins_raw != "*" else ["*"]

app.add_middleware(SystemHealthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"], response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "service": "vigilant-api"}


app.include_router(events.router, prefix="/api/v1", tags=["Events"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])
app.include_router(incidents.router, prefix="/api/v1", tags=["Monitoring"])
app.include_router(reporter.router, prefix="/api/v1", tags=["Data"])
app.include_router(telemetry.router, prefix="/api/v1", tags=["System"])
