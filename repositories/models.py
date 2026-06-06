from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.database import Database


class ModelRepository:
    """
    Read/write the per-model snapshot columns on the `models` table:
    baseline, data_eval, pre_prod_eval, schema_yaml.

    These columns are the fast-path "latest state" used by the drift
    detector and the model registry UI. The reports table still records
    the full evaluation event history.
    """

    def __init__(self, db: "Database") -> None:
        self._db = db

    # ------------------------------------------------------------------
    # baseline
    # ------------------------------------------------------------------

    def get_baseline(self, model_id: str | None = None) -> dict[str, dict] | None:
        """Return {feature_name: stats_json}, or None if no baseline is set.
        Falls back to the database's resolved default model when model_id is None."""
        model_id = model_id or self._db.default_model_id
        row = self._db.fetchone(
            "SELECT baseline FROM models WHERE model_id = ?",
            [model_id],
        )
        if row is None:
            return None
        return _coerce_jsonb(row["baseline"])

    def set_baseline(self, model_id: str, baseline: dict[str, dict]) -> None:
        self._db.execute(
            "UPDATE models SET baseline = ?::jsonb WHERE model_id = ?",
            [json.dumps(baseline), model_id],
        )

    # ------------------------------------------------------------------
    # data_eval / pre_prod_eval
    # ------------------------------------------------------------------

    def upsert_data_eval(
        self,
        model_id: str,
        stage: str,
        split: str,
        eval_payload: dict[str, Any],
    ) -> None:
        """Merge a single (stage.split) entry into models.data_eval."""
        key = f"{stage}.{split}"
        self._db.execute(
            "UPDATE models"
            " SET data_eval = COALESCE(data_eval, '{}'::jsonb) || jsonb_build_object(?, ?::jsonb)"
            " WHERE model_id = ?",
            [key, json.dumps(eval_payload), model_id],
        )

    def set_pre_prod_eval(self, model_id: str, eval_payload: dict[str, Any]) -> None:
        self._db.execute(
            "UPDATE models SET pre_prod_eval = ?::jsonb WHERE model_id = ?",
            [json.dumps(eval_payload), model_id],
        )

    # ------------------------------------------------------------------
    # schema snapshot
    # ------------------------------------------------------------------

    def set_schema_yaml(self, model_id: str, schema: dict[str, Any]) -> None:
        self._db.execute(
            "UPDATE models SET schema_yaml = ?::jsonb WHERE model_id = ?",
            [json.dumps(schema), model_id],
        )

    # ------------------------------------------------------------------
    # registry
    # ------------------------------------------------------------------

    def ensure_exists(self, model_id: str, display_name: str | None = None) -> None:
        """Insert a registry row if missing — required before setting JSONB cols."""
        self._db.execute(
            "INSERT INTO models (model_id, display_name) VALUES (?, ?)"
            " ON CONFLICT (model_id) DO NOTHING",
            [model_id, display_name or model_id],
        )


def _coerce_jsonb(value: Any) -> dict | None:
    """psycopg2 returns JSONB as dict; older paths may pass through str."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value
