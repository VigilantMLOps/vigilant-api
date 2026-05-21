"""Smoke tests that verify basic infrastructure is wired up correctly."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from services.alerting_engine import AlertManager
from tests.fake_database import FakeDatabase


async def test_health_endpoint_returns_200(async_client):
    response = await async_client.get("/health")

    assert response.status_code == 200


async def test_health_endpoint_body_is_healthy(async_client):
    response = await async_client.get("/health")
    body = response.json()

    assert body["status"] == "healthy"


def test_core_database_directory_exists():
    db_dir = Path(__file__).parent.parent / "core" / "database"

    assert db_dir.is_dir(), f"Expected core/database/ directory at {db_dir}"


def test_postgres_schema_file_exists():
    schema = Path(__file__).parent.parent / "core" / "database" / "postgres" / "schema.sql"

    assert schema.is_file(), f"Expected postgres/schema.sql at {schema}"


def test_clickhouse_schema_file_exists():
    schema = Path(__file__).parent.parent / "core" / "database" / "clickhouse" / "schema.sql"

    assert schema.is_file(), f"Expected clickhouse/schema.sql at {schema}"


# ---------------------------------------------------------------------------
# Logger verification
# ---------------------------------------------------------------------------

def test_logger_directory_is_created():
    from core.logger import get_logger

    get_logger("test.infra")

    log_dir = Path(__file__).parent.parent / "core" / "logs"
    assert log_dir.is_dir(), f"Expected core/logs/ directory at {log_dir}"


def test_logger_file_is_created():
    from core.logger import get_logger

    get_logger("test.infra")

    log_file = Path(__file__).parent.parent / "core" / "logs" / "backend.log"
    assert log_file.is_file(), f"Expected backend.log at {log_file}"


def test_logger_write_does_not_raise():
    from core.logger import get_logger

    log = get_logger("test.infra")
    log.info("Test Log")


# ---------------------------------------------------------------------------
# SystemHealthMiddleware tests
# ---------------------------------------------------------------------------

@pytest.fixture
def infra_db() -> FakeDatabase:
    """Fresh in-memory database with schema applied, isolated to system-health tests."""
    db = FakeDatabase()
    db.startup()
    yield db
    db.shutdown()


async def test_slow_request_triggers_latency_warning(async_client, monkeypatch):
    """Requests exceeding the 500 ms threshold must emit a WARNING system alert."""
    import main as main_module

    mock_trigger = MagicMock()
    monkeypatch.setattr(main_module._alert_manager, "trigger_alert", mock_trigger)

    class _FakePerf:
        def __sub__(self, other):
            return 0.6

    monkeypatch.setattr(main_module.time, "perf_counter", _FakePerf)

    await async_client.get("/health")

    mock_trigger.assert_called_once()
    assert mock_trigger.call_args.kwargs["severity"] == "WARNING"
    assert mock_trigger.call_args.kwargs["event_type"] == "SYSTEM"


async def test_server_error_logs_critical_alert_to_db(async_client, infra_db, monkeypatch):
    """A route returning HTTP 500 must write a CRITICAL incident row to the database."""
    import main as main_module
    from main import app
    from fastapi import HTTPException

    monkeypatch.setattr(main_module, "_alert_manager", AlertManager(db=infra_db))
    monkeypatch.setattr("services.notification_service.db", infra_db)

    @app.get("/_test_500_infra")
    def _always_fails():
        raise HTTPException(status_code=500, detail="intentional test error")

    try:
        response = await async_client.get("/_test_500_infra")
        assert response.status_code == 500
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes
            if getattr(r, "path", None) != "/_test_500_infra"
        ]

    rows = infra_db.fetchall(
        "SELECT * FROM incidents WHERE severity = 'CRITICAL' AND incident_type = 'SYSTEM'"
    )
    assert len(rows) == 1
    assert "_test_500_infra" in rows[0]["description"]