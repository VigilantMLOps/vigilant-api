-- =============================================================================
-- PostgreSQL Schema — VigilantMLOps (OLTP layer)
--
-- Responsibility boundary:
--   PostgreSQL owns all mutable, relational state:
--     · models        — model registry; each row carries its baseline,
--                       latest static evals, and schema snapshot as JSONB
--     · reports       — evaluation event log (PRE_PROD + DATA_EVAL + DRIFT)
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
-- FK anchor for reports and incidents.
--
-- Each row carries its own static state as JSONB:
--   baseline       — {feature_name: stats_json}, used by the drift detector
--                    (replaces the old per-row feature_stats table)
--   data_eval      — {"<stage>.<split>": eval}, latest pre-production data
--                    profile per stage/split
--   pre_prod_eval  — latest pre-production test-set evaluation
--   schema_yaml    — snapshot of core/ml_engine/schema.yaml at registration
--
-- A 'default' seed row is inserted so legacy code paths that don't pass
-- a model_id can still attribute writes.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS models (
    model_id        VARCHAR(255)    PRIMARY KEY,
    model_name      VARCHAR(255)    NOT NULL,
    model_version   VARCHAR(255)    NOT NULL,
    display_name    VARCHAR(255)    NOT NULL,
    -- classification | regression | llm | embedding | other
    model_type      VARCHAR(50)     NOT NULL DEFAULT 'classification',
    framework       VARCHAR(100),
    api_url         VARCHAR(500),
    description     TEXT,
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    config          JSONB,
    baseline        JSONB,
    data_eval       JSONB,
    pre_prod_eval   JSONB,
    schema_yaml     JSONB,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Two models can share a name but not a (name, version) pair.
    CONSTRAINT models_name_version_unique UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_models_active
    ON models (is_active) WHERE is_active = TRUE;

DROP TRIGGER IF EXISTS trg_models_updated_at ON models;
CREATE TRIGGER trg_models_updated_at
    BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

-- Seed the canonical "Malicious detector" v1 model. App code resolves the
-- generated UUID at startup by querying (model_name, model_version) so
-- different installs end up with different UUIDs without code changes.
INSERT INTO models (model_id, model_name, model_version, display_name, model_type)
VALUES (gen_random_uuid()::TEXT, 'Malicious detector', 'v1', 'Malicious detector', 'classification')
ON CONFLICT (model_name, model_version) DO NOTHING;

-- ---------------------------------------------------------------------------
-- reports
-- Event log of evaluation reports. Three types share one table; the
-- type-specific payload lives in `content` (JSONB) so each shape can evolve
-- independently:
--   PRE_PROD  → {"accuracy":…, "f1":…, "confusion_matrix":[…], …}
--   DATA_EVAL → {"split":…, "stage":…, "n_rows":…, "features":[…], …}
--   DRIFT     → {"n_drifted":…, "drift_rate":…, "features":[…], …}
--
-- Only cross-type metadata is kept as plain columns (filter/order targets).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS reports (
    report_id       UUID                PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp       TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    report_type     report_type_enum    NOT NULL,
    model_id        VARCHAR(255)        REFERENCES models (model_id) ON DELETE SET NULL,
    model_version   VARCHAR(255),
    content         JSONB
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