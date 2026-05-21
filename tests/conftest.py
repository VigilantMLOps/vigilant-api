"""Shared pytest fixtures for the VigilantMLOps backend test suite."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from main import app
from tests.fake_database import FakeDatabase

# ---------------------------------------------------------------------------
# Seed data — two reports so the "latest" endpoint has a deterministic answer.
# Mirrors the row shape produced by ReporterService.save_report_to_db().
# Timestamps are explicit so ordering is stable regardless of insert speed.
# ---------------------------------------------------------------------------
_SEED_REPORTS = [
    {
        "report_id": "seed-001",
        "timestamp": "2026-01-01T10:00:00+00:00",
        "report_type": "PRE_PROD",
        "model_version": "v1.0-test",
        "metrics": json.dumps({
            "accuracy": 0.9200,
            "precision": 0.9100,
            "recall": 0.9300,
            "f1": 0.9200,
            "roc_auc": 0.9600,
            "avg_precision": 0.9400,
        }),
        "artifacts": json.dumps({
            "confusion_matrix": [[460, 40], [30, 470]],
            "roc_curve_fpr": [0.0, 0.04, 1.0],
            "roc_curve_tpr": [0.0, 0.93, 1.0],
            "classification_report": "              precision    recall  f1-score\n   0       0.94      0.92      0.93\n   1       0.90      0.93      0.91",
        }),
    },
    {
        "report_id": "seed-002",
        "timestamp": "2026-01-01T11:00:00+00:00",  # newer — becomes the "latest"
        "report_type": "DRIFT",
        "model_version": "v1.0-test",
        "metrics": json.dumps({
            "accuracy": 0.9500,
            "precision": 0.9400,
            "recall": 0.9600,
            "f1": 0.9500,
            "roc_auc": 0.9800,
            "avg_precision": 0.9700,
        }),
        "artifacts": json.dumps({
            "confusion_matrix": [[480, 20], [10, 490]],
            "roc_curve_fpr": [0.0, 0.02, 1.0],
            "roc_curve_tpr": [0.0, 0.96, 1.0],
            "classification_report": "              precision    recall  f1-score\n   0       0.98      0.96      0.97\n   1       0.94      0.96      0.95",
        }),
    },
]

_INSERT_SQL = (
    "INSERT INTO reports "
    "(report_id, timestamp, report_type, model_version, metrics, artifacts) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


@pytest.fixture
def mock_db() -> FakeDatabase:
    """SQLite in-memory database with schema applied and two seed report rows."""
    db = FakeDatabase()
    db.startup()
    for row in _SEED_REPORTS:
        db.execute(
            _INSERT_SQL,
            [
                row["report_id"],
                row["timestamp"],
                row["report_type"],
                row["model_version"],
                row["metrics"],
                row["artifacts"],
            ],
        )
    yield db
    db.shutdown()


@pytest_asyncio.fixture
async def async_client(mock_db, monkeypatch):
    """AsyncClient wired to the FastAPI app with a seeded in-memory database.

    Two patches are applied:
      1. monitoring.db  → mock_db so route handlers read from seed data.
      2. core.database.db lifecycle methods → no-ops in case the ASGI
         lifespan fires and tries to connect to the real databases.
    """
    import core.database
    import api.v1.monitoring as monitoring_module

    monkeypatch.setattr(monitoring_module, "db", mock_db)
    monkeypatch.setattr(core.database.db, "startup", lambda: None)
    monkeypatch.setattr(core.database.db, "shutdown", lambda: None)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client