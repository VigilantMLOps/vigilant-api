"""API endpoint tests for the VigilantMLOps monitoring routes.

Uses httpx.AsyncClient (via the async_client fixture from conftest.py) so
requests travel through the full FastAPI/Starlette stack with a seeded
in-memory SQLite database — no external services required.
"""
from __future__ import annotations

import pytest

from tests.fake_database import FakeDatabase


async def test_get_latest_report_returns_newest_seed(async_client):
    """GET /reports/latest returns the seed row with the most recent timestamp."""
    response = await async_client.get("/api/v1/reports/latest")

    assert response.status_code == 200
    data = response.json()

    assert data["report_id"] == "seed-002"
    assert data["report_type"] == "DRIFT"
    assert data["model_version"] == "v1.0-test"
    assert data["timestamp"] is not None

    content = data["content"]
    assert isinstance(content, dict)
    assert content["accuracy"] == pytest.approx(0.95)
    assert content["f1"] == pytest.approx(0.95)
    assert content["roc_auc"] == pytest.approx(0.98)
    assert content["confusion_matrix"] == [[480, 20], [10, 490]]


async def test_get_latest_report_empty_db(monkeypatch):
    """GET /reports/latest returns 404 when no reports exist."""
    import core.database
    import api.v1.monitoring as monitoring_module
    from httpx import ASGITransport, AsyncClient
    from main import app

    empty_db = FakeDatabase()
    empty_db.startup()

    monkeypatch.setattr(monitoring_module, "db", empty_db)
    monkeypatch.setattr(core.database.db, "startup", lambda: None)
    monkeypatch.setattr(core.database.db, "shutdown", lambda: None)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/v1/reports/latest")

    empty_db.shutdown()

    assert response.status_code == 404
    assert "No reports found" in response.json()["detail"]


async def test_get_report_history_returns_all_seeds(async_client):
    """GET /reports/history returns both seed rows in descending timestamp order."""
    response = await async_client.get("/api/v1/reports/history")

    assert response.status_code == 200
    rows = response.json()

    assert len(rows) == 2
    assert rows[0]["report_id"] == "seed-002"
    assert rows[1]["report_id"] == "seed-001"
    for row in rows:
        assert isinstance(row["content"], dict)