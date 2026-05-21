-- =============================================================================
-- PostgreSQL Schema — VigilantMLOps (OLTP layer)
--
-- Responsibility boundary:
--   PostgreSQL owns all mutable, relational state:
--     · models        — model registry (FK anchor for multi-model scaling)
--     · feature_stats — drift baselines per (model_id, feature_name)
--     · reports       — evaluation results (pre-prod + drift)
--     · incidents     — alert lifecycle with UPDATE-able status
--
-- ClickHouse owns all append-only, high-volume analytics:
--     · production_log, alerts, drift_results, report_metrics, llm_traces
--
-- Column naming: `timestamp` is kept (not renamed to `created_at`) so that
-- existing service SQL continues to work without changes in this migration step.
-- Typed metric columns are added alongside JSON blobs; services will migrate
-- to them in the next step.
--
-- Env vars:
--   POSTGRES_HOST  POSTGRES_PORT  POSTGRES_DB  POSTGRES_USER  POSTGRES_PASSWORD
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "btree_gin";  -- GIN indexes on JSONB

-- ---------------------------------------------------------------------------
-- Enum types
-- ---------------------------------------------------------------------------

DO $$ BEGIN
    CREATE TYPE report_type_enum AS ENUM ('DATA_EVAL', 'PRE_PROD', 'DRIFT');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE severity_enum AS ENUM ('INFO', 'WARNING', 'CRITICAL');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE incident_type_enum AS ENUM ('DRIFT', 'SYSTEM', 'PERFORMANCE');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE incident_status_enum AS ENUM ('TRIGGERED', 'ESCALATED', 'AUTO_RESOLVED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE feature_type_enum AS ENUM ('numeric', 'categorical');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ---------------------------------------------------------------------------
-- Shared trigger: keep updated_at current on any UPDATE
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- models
-- Registry of all ML and LLM models managed by the platform.
-- FK anchor for feature_stats, reports, and incidents.
-- A 'default' seed row is inserted so existing services that don't yet
-- pass a model_id can still write to feature_stats without FK violations.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS models (
    model_id        VARCHAR(255)    PRIMARY KEY,
    display_name    VARCHAR(255)    NOT NULL,
    -- classification | regression | llm | embedding | other
    model_type      VARCHAR(50)     NOT NULL DEFAULT 'classification',
    framework       VARCHAR(100),
    api_url         VARCHAR(500),
    description     TEXT,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    config          JSONB,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_models_active
    ON models (is_active) WHERE is_active = TRUE;

DROP TRIGGER IF EXISTS trg_models_updated_at ON models;
CREATE TRIGGER trg_models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

-- Seed row so feature_stats FK is satisfied before any model is registered.
INSERT INTO models (model_id, display_name, model_type)
VALUES ('default', 'Default Model', 'classification')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- feature_stats
-- Baseline distributions computed from training data for drift detection.
--
-- Primary key: (model_id, feature_name) for multi-model support.
-- `stats_json` keeps the original column name so DriftDetector SQL works
-- without changes. Typed numeric summary columns are added alongside for
-- the next migration step.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS feature_stats (
    model_id        VARCHAR(255)        NOT NULL DEFAULT 'default',
    feature_name    VARCHAR(255)        NOT NULL,
    stat_type       feature_type_enum,

    -- Typed summary stats (NULL until services are updated to write them)
    mean            DOUBLE PRECISION,
    std_dev         DOUBLE PRECISION,
    min_val         DOUBLE PRECISION,
    max_val         DOUBLE PRECISION,
    p25             DOUBLE PRECISION,
    p50             DOUBLE PRECISION,
    p75             DOUBLE PRECISION,

    -- Full distribution for PSI — original column name preserved for compat.
    -- numeric:     {"bins": [...], "counts": [...]}
    -- categorical: {"tcp": 0.45, "udp": 0.42, ...}
    stats_json      JSONB               NOT NULL DEFAULT '{}',

    updated_at      TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    PRIMARY KEY (model_id, feature_name),
    FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feature_stats_model
    ON feature_stats (model_id);

-- ---------------------------------------------------------------------------
-- reports
-- Evaluation reports written at pre-production and drift check time.
--
-- `timestamp` column name is kept (not renamed to `created_at`) so that
-- monitoring.py queries like `ORDER BY timestamp DESC` continue to work.
-- `metrics JSONB` is kept alongside typed metric columns for the same reason.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS reports (
    report_id       UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    report_type     report_type_enum    NOT NULL,
    model_id        VARCHAR(255)        REFERENCES models (model_id) ON DELETE SET NULL,
    model_version   VARCHAR(255),

    -- Original JSON blob — kept for backward compat with service SELECT queries.
    -- PRE_PROD: {"accuracy":…, "f1":…, "roc_auc":…, …}
    -- DATA_EVAL: {"n_rows":…, "class_distribution":…, …}
    metrics         JSONB,

    -- Heavy artifacts: confusion matrix, ROC arrays, per-feature profile list.
    artifacts       JSONB,

    -- Typed metric columns (NULL until service inserts are updated).
    -- Enables ORDER BY / WHERE on metrics without JSONB extraction.
    accuracy        DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall          DOUBLE PRECISION,
    f1_score        DOUBLE PRECISION,
    roc_auc         DOUBLE PRECISION,
    avg_precision   DOUBLE PRECISION,

    -- DATA_EVAL quality metrics
    split           VARCHAR(50),
    stage           VARCHAR(50),
    n_rows          INTEGER,
    n_features      SMALLINT,
    imbalance_ratio DOUBLE PRECISION,
    duplicate_rows  INTEGER,
    missing_cells   INTEGER,
    class_distribution JSONB
);

CREATE INDEX IF NOT EXISTS idx_reports_timestamp
    ON reports (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_reports_type_ts
    ON reports (report_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_reports_model
    ON reports (model_id, model_version, timestamp DESC);

-- ---------------------------------------------------------------------------
-- incidents
-- Alert events with a mutable status lifecycle.
-- The ONLY table with UPDATE operations — must stay in PostgreSQL.
--
-- `timestamp` is kept (not renamed) for backward compat with incident queries.
-- `updated_at` is new — set automatically by trigger on every UPDATE.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     UUID                    PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    severity        severity_enum           NOT NULL,
    incident_type   incident_type_enum      NOT NULL,
    status          incident_status_enum    NOT NULL DEFAULT 'TRIGGERED',
    description     TEXT,
    model_id        VARCHAR(255)            REFERENCES models (model_id) ON DELETE SET NULL,
    -- Contextual data: {"feature": "proto", "psi": 0.34, "event_type": "DRIFT"}
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_incidents_timestamp
    ON incidents (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_status
    ON incidents (status);

CREATE INDEX IF NOT EXISTS idx_incidents_severity
    ON incidents (severity, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_model
    ON incidents (model_id, timestamp DESC);

DROP TRIGGER IF EXISTS trg_incidents_updated_at ON incidents;
CREATE TRIGGER trg_incidents_updated_at
    BEFORE UPDATE ON incidents
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

-- ---------------------------------------------------------------------------
-- schema_migrations
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version         INTEGER         PRIMARY KEY,
    description     VARCHAR(500)    NOT NULL,
    applied_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version, description)
VALUES (1, 'Initial schema: models, feature_stats, reports, incidents')
ON CONFLICT DO NOTHING;