-- Phase 3: RAG observability columns on llm_traces.
-- vigilant-rag emits one row per query via POST /api/v1/telemetry/rag-trace.
-- All columns use IF NOT EXISTS so this migration is safe to re-run.

ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS query_text             String                  DEFAULT '';
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS query_mode             LowCardinality(String)  DEFAULT '';
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS n_retrieved            UInt8                   DEFAULT 0;
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS top_retrieval_score    Float32                 DEFAULT 0;
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS retrieval_latency_ms   UInt32                  DEFAULT 0;
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS generation_latency_ms  UInt32                  DEFAULT 0;
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS sources                Array(String)           DEFAULT [];
ALTER TABLE llm_traces ADD COLUMN IF NOT EXISTS prompt_version         LowCardinality(String)  DEFAULT ''
