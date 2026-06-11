"""Telemetry routes — RAG trace ingestion from vigilant-rag."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.database import db
from repositories import RagTraceRepository

router = APIRouter()

_rag_repo = RagTraceRepository(db=db)


# ── Request / Response models ─────────────────────────────────────────────────

class RagTracePayload(BaseModel):
    trace_id: str
    query_text: str
    query_mode: str
    n_retrieved: int
    top_retrieval_score: float
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model_name: str
    sources: list[str]
    prompt_version: str = "v1"


class RagTraceResponse(BaseModel):
    status: str
    trace_id: str


class TelemetryStatusResponse(BaseModel):
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/telemetry/rag-trace", response_model=RagTraceResponse)
def ingest_rag_trace(payload: RagTracePayload) -> RagTraceResponse:
    """Ingest one RAG query trace from vigilant-rag into ClickHouse llm_traces."""
    try:
        _rag_repo.insert(
            trace_id=payload.trace_id,
            query_text=payload.query_text,
            query_mode=payload.query_mode,
            n_retrieved=payload.n_retrieved,
            top_retrieval_score=payload.top_retrieval_score,
            retrieval_latency_ms=payload.retrieval_latency_ms,
            generation_latency_ms=payload.generation_latency_ms,
            total_latency_ms=payload.total_latency_ms,
            prompt_tokens=payload.prompt_tokens,
            completion_tokens=payload.completion_tokens,
            model_name=payload.model_name,
            sources=payload.sources,
            prompt_version=payload.prompt_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store trace: {exc}")
    return RagTraceResponse(status="ok", trace_id=payload.trace_id)


@router.get("/telemetry/rag-traces", response_model=list[dict])
def list_rag_traces(limit: int = 50, since: str | None = None) -> list[dict]:
    """Return recent RAG traces, optionally filtered by time window.

    `since` is an ISO-8601 timestamp; only traces at or after this time are returned.
    """
    try:
        return _rag_repo.fetch_recent(limit=min(limit, 200), since=since)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch traces: {exc}")


@router.get("/telemetry/status", response_model=TelemetryStatusResponse)
def telemetry_status() -> TelemetryStatusResponse:
    return TelemetryStatusResponse(status="ok", message="Telemetry active.")
