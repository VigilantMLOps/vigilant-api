"""Tests for PerformanceService — performance drop detection and alert metadata."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from services.alerting_engine import AlertManager
from services.performance_service import PerformanceService
from tests.fake_database import FakeDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db() -> FakeDatabase:
    """Fresh in-memory database with schema applied, isolated per test."""
    db = FakeDatabase()
    db.startup()
    yield db
    db.shutdown()


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_Y_TRUE_POOR    = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_Y_PRED_POOR    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 50% accuracy

_Y_TRUE_PERFECT = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_Y_PRED_PERFECT = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_performance_drop_triggers_critical_alert():
    """50% accuracy (below 0.85 threshold) must fire trigger_alert(severity='CRITICAL')."""
    mock_alert_manager = MagicMock(spec=AlertManager)
    mock_alert_manager.trigger_alert.return_value = "mock-incident-id"
    svc = PerformanceService(alert_manager=mock_alert_manager)

    result = svc.check_model_performance(_Y_TRUE_POOR, _Y_PRED_POOR)

    assert result.status == "CRITICAL"
    assert result.accuracy == pytest.approx(0.5)
    mock_alert_manager.trigger_alert.assert_called_once()
    assert mock_alert_manager.trigger_alert.call_args.kwargs["severity"] == "CRITICAL"


def test_healthy_performance_no_critical_alert():
    """Perfect accuracy must not fire any alert — status is OK."""
    mock_alert_manager = MagicMock(spec=AlertManager)
    svc = PerformanceService(alert_manager=mock_alert_manager)

    result = svc.check_model_performance(_Y_TRUE_PERFECT, _Y_PRED_PERFECT)

    assert result.status == "OK"
    mock_alert_manager.trigger_alert.assert_not_called()


def test_alert_metadata_contains_accuracy_score(tmp_db, monkeypatch):
    """The persisted alert row must carry the exact accuracy value (0.5) in metadata."""
    monkeypatch.setattr("services.notification_service.db", tmp_db)

    alert_manager = AlertManager(db=tmp_db)
    svc = PerformanceService(alert_manager=alert_manager)

    svc.check_model_performance(_Y_TRUE_POOR, _Y_PRED_POOR)

    row = tmp_db.fetchone("SELECT metadata FROM alerts")
    assert row is not None, "Expected an alert row in the alerts table"

    meta = json.loads(row["metadata"])
    assert meta["accuracy"] == pytest.approx(0.5)