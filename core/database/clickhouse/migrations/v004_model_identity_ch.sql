-- =============================================================================
-- Migration v4 — CH side.
--
-- 1) Collapse report_metrics typed columns into a single `content` String.
--    Each report type has its own metric shape; sharing typed columns forced
--    half of them NULL/0 on every row. Same shape decision as PG `reports`.
--
-- 2) Truncate analytics tables where model_id sits in the ORDER BY sort key:
--    ClickHouse forbids ALTER UPDATE on sort-key columns. These tables are
--    derived from PostgreSQL events (the source of truth), so truncating is
--    safe — next /reporter/evaluate-* and drift runs repopulate them with
--    the new model UUID. The price: the rolling production_log drift window
--    resets and must be re-seeded.
--
-- 3) `alerts` is left alone (model_id is NOT in its ORDER BY) — the Python
--    runner backfills it via ALTER UPDATE.
-- =============================================================================

TRUNCATE TABLE IF EXISTS report_metrics;
TRUNCATE TABLE IF EXISTS production_log;
TRUNCATE TABLE IF EXISTS drift_results;
TRUNCATE TABLE IF EXISTS llm_traces;

ALTER TABLE report_metrics
    DROP COLUMN IF EXISTS accuracy,
    DROP COLUMN IF EXISTS precision_score,
    DROP COLUMN IF EXISTS recall,
    DROP COLUMN IF EXISTS f1_score,
    DROP COLUMN IF EXISTS roc_auc,
    DROP COLUMN IF EXISTS avg_precision,
    DROP COLUMN IF EXISTS n_rows,
    DROP COLUMN IF EXISTS n_features,
    DROP COLUMN IF EXISTS imbalance_ratio,
    DROP COLUMN IF EXISTS duplicate_rows,
    DROP COLUMN IF EXISTS missing_cells;

ALTER TABLE report_metrics
    ADD COLUMN IF NOT EXISTS content String DEFAULT '{}';
