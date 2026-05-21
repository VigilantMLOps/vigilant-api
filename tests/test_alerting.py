"""Tests for the alerting engine — threshold severity, DB persistence, and metadata."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from services.alerting_engine import AlertManager
from services.data_loader import DataPaths, RawDataPaths
from services.reporter import ModelAPIConfig, ReporterConfig, ReporterService
from tests.fake_database import FakeDatabase

_DUMMY = Path("/tmp")

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
_REFERENCE_RECORDS = [
    {"flow_duration": 0.02,  "bytes_total": 50,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "state": "ACC"},
    {"flow_duration": 0.05,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "state": "na"},
    {"flow_duration": 0.10,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "state": "ACC"},
    {"flow_duration": 0.30,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "state": "ACC"},
    {"flow_duration": 0.53,  "bytes_total": 872,   "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "state": "na"},
    {"flow_duration": 0.40,  "bytes_total": 870,   "pkts_total": 9.5,   "rate": 10.0,    "srate": 10.0,    "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "state": "na"},
    {"flow_duration": 64.0,  "bytes_total": 18700, "pkts_total": 82.0,  "rate": 18.0,    "srate": 18.0,    "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "state": "na"},
    {"flow_duration": 64.0,  "bytes_total": 19000, "pkts_total": 82.0,  "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "state": "na"},
    {"flow_duration": 64.5,  "bytes_total": 40000, "pkts_total": 120.0, "rate": 500.0,   "srate": 500.0,   "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "state": "na"},
    {"flow_duration": 65.0,  "bytes_total": 40500, "pkts_total": 120.0, "rate": 5000.0,  "srate": 5000.0,  "drate": 0.0, "min": 54.0, "max": 200.0, "avg": 59.0, "std": 5.0, "state": "na"},
]

_PRODUCTION_RECORDS = _REFERENCE_RECORDS * 6  # 60 rows ≥ _MIN_PSI_SAMPLES=50


def _make_config() -> ReporterConfig:
    return ReporterConfig(
        data=DataPaths(
            raw=RawDataPaths(unsw_nb15_dir=_DUMMY, ciciot2023_dir=_DUMMY),
            balanced_dir=_DUMMY,
            blacklist=_DUMMY / "blacklist.parquet",
        ),
        model_api=ModelAPIConfig(base_url="http://localhost:9999"),
        categorical_columns=["state"],
        target_column="target",
    )


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
# Threshold tests
# ---------------------------------------------------------------------------

def test_critical_threshold_triggers_critical_alert(monkeypatch):
    """PSI=0.9 on feature 'state' must fire trigger_alert(severity='CRITICAL')."""
    mock_alert_manager = MagicMock(spec=AlertManager)
    config = _make_config()
    service = ReporterService(config=config, alert_manager=mock_alert_manager)

    ref_df = pl.DataFrame(_REFERENCE_RECORDS)
    prod_df = pl.DataFrame(_PRODUCTION_RECORDS)
    monkeypatch.setattr(service._loader, "load_reference", lambda: ref_df)

    with (
        patch("services.reporter._psi_categorical", return_value=0.9),
        patch("services.reporter._chi2_test", return_value=(1.0, 0.5)),
    ):
        service.evaluate_data_drift(prod_df)

    mock_alert_manager.trigger_alert.assert_called_once()
    assert mock_alert_manager.trigger_alert.call_args.kwargs["severity"] == "CRITICAL"


def test_warning_threshold_triggers_warning_alert(monkeypatch):
    """PSI=0.15 on feature 'state' must fire trigger_alert(severity='WARNING')."""
    mock_alert_manager = MagicMock(spec=AlertManager)
    config = _make_config()
    service = ReporterService(config=config, alert_manager=mock_alert_manager)

    ref_df = pl.DataFrame(_REFERENCE_RECORDS)
    prod_df = pl.DataFrame(_PRODUCTION_RECORDS)
    monkeypatch.setattr(service._loader, "load_reference", lambda: ref_df)

    with (
        patch("services.reporter._psi_categorical", return_value=0.15),
        patch("services.reporter._chi2_test", return_value=(1.0, 0.5)),
    ):
        service.evaluate_data_drift(prod_df)

    mock_alert_manager.trigger_alert.assert_called_once()
    assert mock_alert_manager.trigger_alert.call_args.kwargs["severity"] == "WARNING"


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_alert_persists_to_alerts_table(tmp_db, monkeypatch):
    """trigger_alert writes one row to alerts with the correct message and UUID."""
    monkeypatch.setattr("services.notification_service.db", tmp_db)

    manager = AlertManager(db=tmp_db)
    description = "Feature 'state' drift detected (PSI=0.9000)"
    incident_id = manager.trigger_alert(
        severity="CRITICAL",
        event_type="DRIFT",
        description=description,
        metadata={"feature": "state", "psi": 0.9},
    )

    rows = tmp_db.fetchall("SELECT * FROM alerts")
    assert len(rows) == 1, "Expected exactly one alert row in the alerts table"

    alert = rows[0]
    assert alert["message"] == description

    meta = json.loads(alert["metadata"])
    assert meta["incident_id"] == incident_id


def test_alert_metadata_contains_feature_and_psi(tmp_db, monkeypatch):
    """The persisted alert metadata JSON carries the feature name and PSI score."""
    monkeypatch.setattr("services.notification_service.db", tmp_db)

    manager = AlertManager(db=tmp_db)
    manager.trigger_alert(
        severity="CRITICAL",
        event_type="DRIFT",
        description="Feature 'state' drift detected (PSI=0.9000)",
        metadata={"feature": "state", "psi": 0.9},
    )

    row = tmp_db.fetchone("SELECT metadata FROM alerts")
    assert row is not None

    meta = json.loads(row["metadata"])
    assert meta["feature"] == "state"
    assert meta["psi"] == pytest.approx(0.9)