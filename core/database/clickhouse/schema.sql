-- =============================================================================
-- ClickHouse Schema — VigilantMLOps (OLAP layer)
--
-- Responsibility boundary:
--   ClickHouse owns all append-only, high-volume analytics data:
--     · production_log   — feature batches from production inference requests
--     · alerts           — append-only notification event log
--     · drift_results    — per-feature PSI scores over time (new)
--     · report_metrics   — analytics mirror of PostgreSQL reports metrics (new)
--     · llm_traces       — LLM inference observability (new, for scaling)
--
-- Backward compatibility:
--   `production_log.features` uses String (JSON text) so the current
--   reporter.py INSERT continues to work. The next migration step converts
--   it to Map(String, Float64) + Map(String, String).
--   `alerts.metadata` uses String (JSON text) for the same reason.
--
-- Key design rules:
--   · LowCardinality(String) for columns with <10k distinct values.
--   · PARTITION BY month for time-range pruning.
--   · ORDER BY starts with the most selective column (model_id).
--   · TTL for automatic storage management on high-volume tables.
--
-- Env vars:
--   CLICKHOUSE_HOST  CLICKHOUSE_PORT  CLICKHOUSE_DB
--   CLICKHOUSE_USER  CLICKHOUSE_PASSWORD
-- =============================================================================

-- ---------------------------------------------------------------------------
-- production_log
-- Feature data accumulated from live production inference requests.
-- Written by: ReporterService._append_production_records() (row-by-row)
-- Read by:    ReporterService._load_production_records() (full table scan)
-- Reset by:   ReporterService.reset_production_log() (DELETE → TRUNCATE)
--
-- `features String` stores the JSON-serialised feature dict from the current
-- service code. Replace with Map(String, Float64) / Map(String, String) when
-- reporter.py is updated to use the typed ingestion path.
--
-- TTL: 90 days. Adjust via ALTER TABLE production_log MODIFY TTL.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS production_log (
    log_id          UUID,
    received_at     DateTime64(3, 'UTC')        DEFAULT now64(3),
    model_id        LowCardinality(String)      DEFAULT '',
    model_version   LowCardinality(String)      DEFAULT '',
    batch_id        UUID                        DEFAULT generateUUIDv4(),

    -- JSON text — backward compat with current service INSERT.
    -- Target shape: {"feature_a": 1.2, "feature_b": "tcp", ...}
    features        String                      DEFAULT '',

    source_tag      LowCardinality(String)      DEFAULT ''  -- api | kafka | batch | script
)
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(received_at), model_id)
ORDER BY (model_id, received_at, log_id)
TTL toDateTime(received_at) + INTERVAL 90 DAY DELETE
SETTINGS index_granularity = 8192;


-- Buffer table for production_log.
-- Future write path: INSERT into production_log_buffer instead of
-- production_log directly. Buffer batches row-by-row inserts into fewer,
-- larger MergeTree writes.
-- Flush triggers (first to fire wins):
--   10 s idle  |  60 s max wait
--   10k rows   |  500k rows max
--   10 MB      |  500 MB max
CREATE TABLE IF NOT EXISTS production_log_buffer AS production_log
ENGINE = Buffer(
    currentDatabase(), 'production_log',
    16,
    10, 60,
    10000, 500000,
    10485760, 524288000
);


-- ---------------------------------------------------------------------------
-- alerts
-- Append-only notification log. Never updated after insert.
-- Written by: notification_service.send_alert()
--
-- `metadata String` stores JSON text — backward compat with current
-- notification_service INSERT. Rename to `metadata_json` and tighten
-- the schema when notification_service.py is updated.
--
-- `incident_id` is a soft reference to PostgreSQL incidents.incident_id.
-- The ingestion layer writes PostgreSQL first, then ClickHouse, to maintain
-- referential consistency without FK enforcement.
--
-- TTL: 365 days.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        UUID,
    timestamp       DateTime64(3, 'UTC')        DEFAULT now64(3),
    level           LowCardinality(String),     -- INFO | WARNING | CRITICAL
    event_type      LowCardinality(String)      DEFAULT '',  -- DRIFT | SYSTEM | PERFORMANCE
    message         String,
    model_id        LowCardinality(String)      DEFAULT '',
    incident_id     Nullable(UUID),             -- soft ref → PostgreSQL incidents
    metadata        String                      DEFAULT '{}'  -- JSON text
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, level, alert_id)
TTL toDateTime(timestamp) + INTERVAL 365 DAY DELETE
SETTINGS index_granularity = 8192;


-- ---------------------------------------------------------------------------
-- drift_results
-- Per-feature PSI scores recorded at every drift check. New table.
-- Enables time-series drift monitoring:
--   "Which features drifted most in the last 7 days?"
--   "Show PSI trend for feature 'proto' over the last month."
-- Written by: DriftDetector.check_drift() after service is updated.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS drift_results (
    drift_id            UUID                        DEFAULT generateUUIDv4(),
    checked_at          DateTime64(3, 'UTC')        DEFAULT now64(3),
    model_id            LowCardinality(String),
    model_version       LowCardinality(String)      DEFAULT '',
    feature_name        LowCardinality(String),
    method              LowCardinality(String),     -- psi+ks | psi+chi2 | psi
    psi_score           Float32,
    pvalue              Nullable(Float32),
    status              LowCardinality(String),     -- OK | WARNING | CRITICAL
    n_production_rows   UInt32                      DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(checked_at)
ORDER BY (model_id, feature_name, checked_at)
SETTINGS index_granularity = 8192;


-- ---------------------------------------------------------------------------
-- report_metrics
-- Analytics mirror of scalar metrics from PostgreSQL reports.
-- Written by ingestion layer alongside the PostgreSQL INSERT.
-- Enables fast time-series aggregation without hitting PostgreSQL:
--   "Show F1 trend for model X over the last 90 days."
-- ReplacingMergeTree collapses duplicate report_ids on background merge,
-- making idempotent re-syncs from PostgreSQL safe.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS report_metrics (
    report_id       UUID,
    timestamp       DateTime64(3, 'UTC')        DEFAULT now64(3),
    report_type     LowCardinality(String),
    model_id        LowCardinality(String),
    model_version   LowCardinality(String)      DEFAULT '',
    -- JSON text — shape varies by report_type so each report owns its payload.
    -- PRE_PROD  → {"accuracy":…, "f1":…, "roc_auc":…, …}
    -- DATA_EVAL → {"n_rows":…, "imbalance_ratio":…, "missing_cells":…, …}
    -- DRIFT     → {"n_drifted":…, "drift_rate":…, "features":[…], …}
    content         String                      DEFAULT '{}'
)
ENGINE = ReplacingMergeTree(timestamp)
PARTITION BY toYYYYMM(timestamp)
ORDER BY (model_id, report_type, timestamp, report_id)
SETTINGS index_granularity = 8192;


-- ---------------------------------------------------------------------------
-- llm_traces
-- Inference observability for LLM deployments. New table.
-- Captures the minimum signal for:
--   · Cost tracking (prompt_tokens, completion_tokens per model/day)
--   · Latency p50/p95/p99 over time
--   · Error rate by model and error_type
--   · Content sampling (previews capped at 500 chars)
-- `tags Map(String, String)` allows arbitrary labeling without schema changes:
--   e.g. {"env": "prod", "tier": "enterprise", "use_case": "summarization"}
-- TTL: 180 days.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS llm_traces (
    trace_id            UUID,
    timestamp           DateTime64(3, 'UTC')        DEFAULT now64(3),
    model_id            LowCardinality(String),
    model_version       LowCardinality(String)      DEFAULT '',
    provider            LowCardinality(String)      DEFAULT '',  -- openai | anthropic | local

    prompt_tokens       UInt32                      DEFAULT 0,
    completion_tokens   UInt32                      DEFAULT 0,
    total_tokens        UInt32                      DEFAULT 0,

    latency_ms          UInt32                      DEFAULT 0,
    is_success          UInt8                       DEFAULT 1,
    error_type          LowCardinality(String)      DEFAULT '',

    prompt_preview      String                      DEFAULT '',
    completion_preview  String                      DEFAULT '',

    user_id             String                      DEFAULT '',
    session_id          String                      DEFAULT '',
    tags                Map(String, String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (model_id, timestamp, trace_id)
TTL toDateTime(timestamp) + INTERVAL 180 DAY DELETE
SETTINGS index_granularity = 8192;