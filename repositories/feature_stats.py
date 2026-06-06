from __future__ import annotations

from typing import TYPE_CHECKING

from .models import ModelRepository

if TYPE_CHECKING:
    from core.database import Database


class FeatureStatsRepository:
    """
    Thin read-side facade over models.baseline. Kept as a separate class so
    the drift detector's import stays stable; new code should depend on
    ModelRepository directly.
    """

    def __init__(self, db: "Database", model_id: str | None = None) -> None:
        self._models = ModelRepository(db)
        self._model_id = model_id  # None → resolve via db.default_model_id at read

    def fetch_all(self) -> dict[str, dict]:
        """Return all baseline stats keyed by feature_name (empty when unset)."""
        return self._models.get_baseline(self._model_id) or {}
