-- =============================================================================
-- Migration v4 — PG side: model identity (name + version + unique constraint).
--
-- Adds the columns needed to identify a model by (model_name, model_version)
-- and converts the existing 'default' seed row to a real UUID. All FK
-- references in PostgreSQL are repointed in this file. ClickHouse-side
-- updates happen in v004_model_identity_ch.sql + the Python runner.
-- =============================================================================

-- 1) Add the new identity columns to models ----------------------------------
ALTER TABLE models
    ADD COLUMN IF NOT EXISTS model_name    VARCHAR(255),
    ADD COLUMN IF NOT EXISTS model_version VARCHAR(255);

-- 2) Convert the 'default' seed row to a UUID + name + version --------------
--    Idempotent: if there's no 'default' row (already migrated), this is a no-op.
--    The new UUID is captured in models for the runner to read.
DO $$
DECLARE
    new_id TEXT;
BEGIN
    IF EXISTS (SELECT 1 FROM models WHERE model_id = 'default') THEN
        new_id := gen_random_uuid()::TEXT;

        -- Repoint FKs first (ON UPDATE CASCADE isn't configured)
        ALTER TABLE reports   DROP CONSTRAINT IF EXISTS reports_model_id_fkey;
        ALTER TABLE incidents DROP CONSTRAINT IF EXISTS incidents_model_id_fkey;

        UPDATE models SET
            model_id      = new_id,
            model_name    = 'Malicious detector',
            model_version = 'v1',
            display_name  = 'Malicious detector'
        WHERE model_id = 'default';

        UPDATE reports
        SET model_id = new_id,
            model_version = COALESCE(NULLIF(model_version, ''), 'v1')
        WHERE model_id = 'default' OR model_id IS NULL;

        UPDATE incidents
        SET model_id = new_id
        WHERE model_id = 'default' OR model_id IS NULL;

        ALTER TABLE reports   ADD CONSTRAINT reports_model_id_fkey
            FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE SET NULL;
        ALTER TABLE incidents ADD CONSTRAINT incidents_model_id_fkey
            FOREIGN KEY (model_id) REFERENCES models (model_id) ON DELETE SET NULL;
    END IF;
END $$;

-- 3) Backfill model_version='v1' on any remaining empty rows -----------------
UPDATE reports SET model_version = 'v1' WHERE model_version IS NULL OR model_version = '';

-- 4) Lock down the new identity columns --------------------------------------
ALTER TABLE models
    ALTER COLUMN model_name    SET NOT NULL,
    ALTER COLUMN model_version SET NOT NULL;

-- (model_name, model_version) must be unique — two models can share a name
-- but not a name+version pair. Idempotent: skip if either the constraint
-- name (duplicate_object) or the implicit index (duplicate_table) exists.
DO $$ BEGIN
    ALTER TABLE models
        ADD CONSTRAINT models_name_version_unique UNIQUE (model_name, model_version);
EXCEPTION
    WHEN duplicate_object THEN NULL;
    WHEN duplicate_table  THEN NULL;
END $$;
