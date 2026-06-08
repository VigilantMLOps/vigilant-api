"""Repository for RAG query traces — writes to ClickHouse llm_traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.database import Database


class RagTraceRepository:
    def __init__(self, db: "Database") -> None:
        self._db = db

    def insert(
        self,
        trace_id: str,
        query_text: str,
        query_mode: str,
        n_retrieved: int,
        top_retrieval_score: float,
        retrieval_latency_ms: int,
        generation_latency_ms: int,
        total_latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
        sources: list[str],
        prompt_version: str = "v1",
    ) -> None:
        self._db.execute(
            "INSERT INTO llm_traces ("
            "  trace_id, model_id, model_version, provider,"
            "  prompt_tokens, completion_tokens, total_tokens,"
            "  latency_ms, is_success, error_type,"
            "  prompt_preview, completion_preview,"
            "  user_id, session_id, tags,"
            "  query_text, query_mode, n_retrieved, top_retrieval_score,"
            "  retrieval_latency_ms, generation_latency_ms, sources, prompt_version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                trace_id,
                model_name,                         # model_id
                "",                                 # model_version
                "local",                            # provider
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,  # total_tokens
                total_latency_ms,                   # latency_ms
                1,                                  # is_success
                "",                                 # error_type
                query_text[:200],                   # prompt_preview
                "",                                 # completion_preview
                "",                                 # user_id
                "",                                 # session_id
                {},                                 # tags (Map[String, String])
                query_text,
                query_mode,
                n_retrieved,
                top_retrieval_score,
                retrieval_latency_ms,
                generation_latency_ms,
                sources,                            # Array(String)
                prompt_version,
            ],
        )

    def fetch_recent(self, limit: int = 50) -> list[dict]:
        return self._db.fetchall(
            "SELECT trace_id, timestamp, query_text, query_mode,"
            "  n_retrieved, top_retrieval_score, total_tokens,"
            "  latency_ms, retrieval_latency_ms, generation_latency_ms,"
            "  model_id, sources, prompt_version"
            " FROM llm_traces"
            " WHERE query_mode != ''"
            f" ORDER BY timestamp DESC LIMIT {limit}"
        )
