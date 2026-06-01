-- =============================================================================
-- Migration v3 — Collapse `reports` type-specific columns into a single
-- `content` JSONB.
--
-- The previous shape mixed metrics-as-typed-columns (accuracy / f1 / ...) with
-- metrics-as-JSON (metrics / artifacts) and forced two unrelated report
-- shapes (PRE_PROD model evals vs. DATA_EVAL data profiles) into the same
-- table — every row had ~half the typed columns NULL.
--
-- After v3 the table holds only type-agnostic metadata as columns and the
-- type-specific payload as JSONB:
--
--   report_id      UUID
--   timestamp      TIMESTAMPTZ
--   report_type    enum (PRE_PROD | DATA_EVAL | DRIFT)
--   model_id       FK → models
--   model_version  VARCHAR
--   content        JSONB         — shape varies by report_type
--
-- Each report_type defines its own `content` shape; new types (DRIFT) just
-- pick a shape and write it without schema changes.
-- =============================================================================

-- 1) Add the content column --------------------------------------------------
ALTER TABLE reports
    ADD COLUMN IF NOT EXISTS content JSONB;

-- 2) Backfill content from existing rows -------------------------------------
--    Strip nulls so the resulting JSON is clean per type. The COALESCE on
--    metrics / artifacts protects against rows that had NULL JSONB.
UPDATE reports
SET content =
    jsonb_strip_nulls(
        COALESCE(metrics,   '{}'::jsonb)
        || COALESCE(artifacts, '{}'::jsonb)
        || jsonb_build_object(
            'accuracy',           accuracy,
            'precision',          precision_score,
            'recall',             recall,
            'f1',                 f1_score,
            'roc_auc',            roc_auc,
            'avg_precision',      avg_precision,
            'split',              split,
            'stage',              stage,
            'n_rows',             n_rows,
            'n_features',         n_features,
            'imbalance_ratio',    imbalance_ratio,
            'duplicate_rows',     duplicate_rows,
            'missing_cells',      missing_cells,
            'class_distribution', class_distribution
        )
    )
WHERE content IS NULL;

-- 3) Drop the now-redundant typed columns ------------------------------------
ALTER TABLE reports
    DROP COLUMN IF EXISTS metrics,
    DROP COLUMN IF EXISTS artifacts,
    DROP COLUMN IF EXISTS accuracy,
    DROP COLUMN IF EXISTS precision_score,
    DROP COLUMN IF EXISTS recall,
    DROP COLUMN IF EXISTS f1_score,
    DROP COLUMN IF EXISTS roc_auc,
    DROP COLUMN IF EXISTS avg_precision,
    DROP COLUMN IF EXISTS split,
    DROP COLUMN IF EXISTS stage,
    DROP COLUMN IF EXISTS n_rows,
    DROP COLUMN IF EXISTS n_features,
    DROP COLUMN IF EXISTS imbalance_ratio,
    DROP COLUMN IF EXISTS duplicate_rows,
    DROP COLUMN IF EXISTS missing_cells,
    DROP COLUMN IF EXISTS class_distribution;
