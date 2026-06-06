-- =============================================================================
-- Migration v2 — Consolidate baselines and static evals onto `models`
--
-- Each model now owns:
--   models.baseline       — JSON dict of {feature_name: stats_json}
--                           (replaces the per-row feature_stats table)
--   models.data_eval      — JSON dict of {"<stage>.<split>": eval}
--                           (latest DATA_EVAL snapshot per stage/split)
--   models.pre_prod_eval  — JSON of the latest PRE_PROD test-set evaluation
--   models.schema_yaml    — JSON snapshot of core/ml_engine/schema.yaml
--                           captured at baseline-registration time
--
-- The `reports` table keeps the full audit trail (one row per evaluation
-- event). The model columns are the fast-path "latest snapshot" used by the
-- drift detector and the model registry UI.
-- =============================================================================

-- 1) Add the new JSONB columns ------------------------------------------------
ALTER TABLE models
    ADD COLUMN IF NOT EXISTS baseline       JSONB,
    ADD COLUMN IF NOT EXISTS data_eval      JSONB,
    ADD COLUMN IF NOT EXISTS pre_prod_eval  JSONB,
    ADD COLUMN IF NOT EXISTS schema_yaml    JSONB;

-- 2) Backfill baseline from feature_stats (when the old table still exists) --
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'feature_stats'
    ) THEN
        UPDATE models m
        SET baseline = sub.baseline
        FROM (
            SELECT model_id,
                   jsonb_object_agg(feature_name, stats_json) AS baseline
            FROM feature_stats
            GROUP BY model_id
        ) sub
        WHERE m.model_id = sub.model_id;
    END IF;
END $$;

-- 3) Backfill pre_prod_eval from the latest PRE_PROD report per model --------
--    Reports may have a NULL model_id (legacy inserts); attribute those to
--    the seed 'default' model.
UPDATE models m
SET pre_prod_eval = sub.eval
FROM (
    SELECT DISTINCT ON (COALESCE(model_id, 'default'))
        COALESCE(model_id, 'default') AS model_id,
        jsonb_build_object(
            'report_id',  report_id,
            'metrics',    metrics,
            'artifacts',  artifacts,
            'timestamp',  timestamp
        ) AS eval
    FROM reports
    WHERE report_type = 'PRE_PROD'
    ORDER BY COALESCE(model_id, 'default'), timestamp DESC
) sub
WHERE m.model_id = sub.model_id;

-- 4) Backfill data_eval: latest DATA_EVAL per (model, stage, split) ----------
UPDATE models m
SET data_eval = sub.eval
FROM (
    SELECT model_id, jsonb_object_agg(stage_split, eval) AS eval
    FROM (
        SELECT DISTINCT ON (COALESCE(model_id, 'default'), stage, split)
            COALESCE(model_id, 'default') AS model_id,
            COALESCE(stage, 'unknown') || '.' || COALESCE(split, 'unknown') AS stage_split,
            jsonb_build_object(
                'report_id',  report_id,
                'metrics',    metrics,
                'artifacts',  artifacts,
                'timestamp',  timestamp
            ) AS eval
        FROM reports
        WHERE report_type = 'DATA_EVAL'
        ORDER BY COALESCE(model_id, 'default'), stage, split, timestamp DESC
    ) latest
    GROUP BY model_id
) sub
WHERE m.model_id = sub.model_id;

-- 5) Drop the now-redundant feature_stats table ------------------------------
DROP TABLE IF EXISTS feature_stats;
