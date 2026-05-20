"""Tests for PerformanceService — performance drop detection and alert metadata."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.database import Database
from services.alerting_engine import AlertManager
from services.performance_service import PerformanceService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db() -> Database:
    """Fresh in-memory DuckDB with schema applied, isolated per test."""
    db = Database(":memory:")
    db.startup()
    yield db
    db.shutdown()


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# 50% accuracy: first 5 labels [0,0,0,0,0] predicted correctly;
# last 5 labels [1,1,1,1,1] predicted as 0 → 5/10 correct.
_Y_TRUE_POOR = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_Y_PRED_POOR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

_Y_TRUE_PERFECT = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
_Y_PRED_PERFECT = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Test 1: Performance drop triggers CRITICAL alert
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


# ---------------------------------------------------------------------------
# Test 2: Healthy performance does not trigger any alert
# ---------------------------------------------------------------------------

def test_healthy_performance_no_critical_alert():
    """Perfect accuracy must not fire any alert — status is OK (INFO log only)."""
    mock_alert_manager = MagicMock(spec=AlertManager)
    svc = PerformanceService(alert_manager=mock_alert_manager)

    result = svc.check_model_performance(_Y_TRUE_PERFECT, _Y_PRED_PERFECT)

    assert result.status == "OK"
    mock_alert_manager.trigger_alert.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Alert metadata contains the exact accuracy score
# ---------------------------------------------------------------------------

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
