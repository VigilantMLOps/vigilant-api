"""End-to-end integration test: drifted data → production_log → DRIFT alert → logger.

Flow under test
---------------
POST /api/v1/reporter/evaluate-drift  (60 drifted records)
    → ReporterService._append_production_records()  →  rows written to production_log
    → evaluate_data_drift()                         →  per-feature PSI/KS/Chi² checks
    → AlertManager.trigger_alert()                  →  row written to incidents table
    → notification_service.send_alert()             →  row written to alerts table
                                                    →  loguru ALERT line emitted
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import polars as pl
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from loguru import logger as loguru_logger

from core.database import Database
from main import app
from services.alerting_engine import AlertManager
from services.data_loader import DataPaths, RawDataPaths
from services.reporter import ModelAPIConfig, ReporterConfig, ReporterService

_DUMMY = Path("/tmp")

# ---------------------------------------------------------------------------
# Reference distribution — mirrors the test records used across the test suite
# ---------------------------------------------------------------------------

_REFERENCE_RECORDS = [
    {"flow_duration": 0.02,  "bytes_total": 50,    "pkts_total": 9.5,   "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "na",   "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.05,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 0.10,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "6.0",  "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.30,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "17.0", "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.53,  "bytes_total": 872,   "pkts_total": 9.5,   "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 0.40,  "bytes_total": 870,   "pkts_total": 9.5,   "rate": 10.0,   "srate": 10.0,   "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 64.0,  "bytes_total": 18700, "pkts_total": 82.0,  "rate": 18.0,   "srate": 18.0,   "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "ciciot"},
    {"flow_duration": 64.0,  "bytes_total": 19000, "pkts_total": 82.0,  "rate": 0.0,    "srate": 0.0,    "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "proto": "6.0",  "state": "na",  "source": "unsw"},
    {"flow_duration": 64.5,  "bytes_total": 40000, "pkts_total": 120.0, "rate": 500.0,  "srate": 500.0,  "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "proto": "1.0",  "state": "na",  "source": "ciciot"},
    {"flow_duration": 65.0,  "bytes_total": 40500, "pkts_total": 120.0, "rate": 5000.0, "srate": 5000.0, "drate": 0.0, "min": 54.0, "max": 200.0, "avg": 59.0, "std": 5.0, "proto": "na",   "state": "na",  "source": "unsw"},
]

# ---------------------------------------------------------------------------
# Drifted payload — 60 records (≥ _MIN_PSI_SAMPLES=50 in reporter.py) with
# numeric values ~1000× larger and unseen categorical labels so every feature
# type (numeric PSI+KS and categorical PSI+Chi²) reliably triggers drift.
# ---------------------------------------------------------------------------

_DRIFTED_UNIT = [
    {"flow_duration": 200.0, "bytes_total": 500_000, "pkts_total": 1000.0, "rate": 100_000.0, "srate": 100_000.0, "drate": 50_000.0, "min": 500.0, "max": 5_000.0,  "avg": 1000.0, "std": 200.0, "proto": "udp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 210.0, "bytes_total": 510_000, "pkts_total": 1010.0, "rate": 110_000.0, "srate": 110_000.0, "drate": 55_000.0, "min": 510.0, "max": 5_100.0,  "avg": 1010.0, "std": 210.0, "proto": "tcp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 220.0, "bytes_total": 520_000, "pkts_total": 1020.0, "rate": 120_000.0, "srate": 120_000.0, "drate": 60_000.0, "min": 520.0, "max": 5_200.0,  "avg": 1020.0, "std": 220.0, "proto": "udp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 230.0, "bytes_total": 530_000, "pkts_total": 1030.0, "rate": 130_000.0, "srate": 130_000.0, "drate": 65_000.0, "min": 530.0, "max": 5_300.0,  "avg": 1030.0, "std": 230.0, "proto": "tcp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 240.0, "bytes_total": 540_000, "pkts_total": 1040.0, "rate": 140_000.0, "srate": 140_000.0, "drate": 70_000.0, "min": 540.0, "max": 5_400.0,  "avg": 1040.0, "std": 240.0, "proto": "udp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 250.0, "bytes_total": 550_000, "pkts_total": 1050.0, "rate": 150_000.0, "srate": 150_000.0, "drate": 75_000.0, "min": 550.0, "max": 5_500.0,  "avg": 1050.0, "std": 250.0, "proto": "tcp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 260.0, "bytes_total": 560_000, "pkts_total": 1060.0, "rate": 160_000.0, "srate": 160_000.0, "drate": 80_000.0, "min": 560.0, "max": 5_600.0,  "avg": 1060.0, "std": 260.0, "proto": "udp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 270.0, "bytes_total": 570_000, "pkts_total": 1070.0, "rate": 170_000.0, "srate": 170_000.0, "drate": 85_000.0, "min": 570.0, "max": 5_700.0,  "avg": 1070.0, "std": 270.0, "proto": "tcp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 280.0, "bytes_total": 580_000, "pkts_total": 1080.0, "rate": 180_000.0, "srate": 180_000.0, "drate": 90_000.0, "min": 580.0, "max": 5_800.0,  "avg": 1080.0, "std": 280.0, "proto": "udp", "state": "FIN", "source": "ciciot"},
    {"flow_duration": 290.0, "bytes_total": 590_000, "pkts_total": 1090.0, "rate": 190_000.0, "srate": 190_000.0, "drate": 95_000.0, "min": 590.0, "max": 5_900.0,  "avg": 1090.0, "std": 290.0, "proto": "tcp", "state": "FIN", "source": "ciciot"},
]

_DRIFTED_RECORDS = _DRIFTED_UNIT * 6  # 60 rows — above _MIN_PSI_SAMPLES=50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_db() -> Database:
    """Fresh in-memory DuckDB with full schema, isolated per test."""
    db = Database(":memory:")
    db.startup()
    yield db
    db.shutdown()


@pytest.fixture
def e2e_reporter(e2e_db: Database) -> ReporterService:
    """ReporterService wired to the in-memory DB with an attached AlertManager."""
    config = ReporterConfig(
        data=DataPaths(
            raw=RawDataPaths(unsw_nb15_dir=_DUMMY, ciciot2023_dir=_DUMMY),
            balanced_dir=_DUMMY,
            blacklist=_DUMMY / "blacklist.parquet",
        ),
        model_api=ModelAPIConfig(base_url="http://localhost:9999"),
        categorical_columns=["proto", "state", "source"],
        target_column="target",
    )
    return ReporterService(
        config=config,
        db=e2e_db,
        alert_manager=AlertManager(db=e2e_db),
    )


@pytest_asyncio.fixture
async def e2e_client(e2e_db: Database, e2e_reporter: ReporterService, monkeypatch):
    """AsyncClient wired to the FastAPI app with all services pointing at e2e_db.

    Patches applied:
      - reporter._reporter       → e2e_reporter (skips disk-based config + file DB)
      - reporter._loader.load_reference → in-memory reference DataFrame
      - notification_service.db  → e2e_db (alert rows land in the test DB)
      - main._alert_manager      → AlertManager(e2e_db) (middleware incidents too)
      - core.database.db lifecycle → no-ops (prevents file-backed DB from opening)
    """
    import api.v1.reporter as reporter_module
    import core.database
    import main as main_module
    import services.notification_service as notif_module

    ref_df = pl.DataFrame(_REFERENCE_RECORDS)
    monkeypatch.setattr(e2e_reporter._loader, "load_reference", lambda: ref_df)
    monkeypatch.setattr(reporter_module, "_reporter", e2e_reporter)
    monkeypatch.setattr(notif_module, "db", e2e_db)
    monkeypatch.setattr(main_module, "_alert_manager", AlertManager(db=e2e_db))
    monkeypatch.setattr(core.database.db, "startup", lambda: None)
    monkeypatch.setattr(core.database.db, "shutdown", lambda: None)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_drifted_payload_triggers_full_pipeline(e2e_client, e2e_db):
    """Posting 60 drifted records must propagate through the entire pipeline.

    Verified chain:
      1. HTTP 200 — endpoint accepts the payload and reports drift detected.
      2. production_log — all 60 feature records are persisted.
      3. alerts table   — at least one DRIFT alert with correct severity and metadata.
      4. Logger         — 'Drift check' and '[ALERT:' lines emitted during processing.
    """
    log_buffer = io.StringIO()
    sink_id = loguru_logger.add(log_buffer, format="{message}", level="DEBUG")

    try:
        response = await e2e_client.post(
            "/api/v1/reporter/evaluate-drift",
            json={"records": _DRIFTED_RECORDS},
        )
        # Drift detection is synchronous within the HTTP request; yield control
        # once so any deferred coroutines (e.g. middleware) can complete.
        await asyncio.sleep(0)
    finally:
        loguru_logger.remove(sink_id)

    # ── 1. HTTP response ──────────────────────────────────────────────────────
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["overall_status"] in ("warning", "critical"), (
        f"Expected drift detected; got overall_status={body['overall_status']!r}"
    )
    assert body["n_drifted"] > 0, "Expected at least one drifted feature in response"

    # ── 2. production_log — feature records persisted ─────────────────────────
    log_rows = e2e_db.fetchall("SELECT log_id, features FROM production_log")
    assert len(log_rows) == len(_DRIFTED_RECORDS), (
        f"Expected {len(_DRIFTED_RECORDS)} rows in production_log, got {len(log_rows)}"
    )
    # Spot-check the first three rows: JSON must round-trip with the expected keys.
    expected_keys = {"flow_duration", "bytes_total", "pkts_total", "rate", "state"}
    for row in log_rows[:3]:
        features = json.loads(row["features"])
        assert expected_keys.issubset(features.keys()), (
            f"Feature record missing expected keys. Got: {set(features.keys())}"
        )

    # ── 3. alerts table — at least one DRIFT alert ────────────────────────────
    alert_rows = e2e_db.fetchall(
        "SELECT level, message, metadata FROM alerts WHERE message LIKE '%drift detected%'"
    )
    assert len(alert_rows) >= 1, (
        "Expected at least one DRIFT alert in the alerts table after posting drifted data"
    )

    alert_levels = {row["level"] for row in alert_rows}
    assert alert_levels & {"WARNING", "CRITICAL"}, (
        f"Expected WARNING or CRITICAL alert level for drift; got {alert_levels}"
    )

    for row in alert_rows:
        meta = json.loads(row["metadata"])
        assert "feature" in meta, "Alert metadata must contain the drifted feature name"
        assert "psi" in meta, "Alert metadata must contain the PSI score"
        assert isinstance(meta["psi"], float), "PSI in metadata must be a float"

    # ── 4. Logger captured the full journey ───────────────────────────────────
    captured = log_buffer.getvalue()
    assert "Drift check" in captured, (
        "Logger must emit 'Drift check' lines from ReporterService.evaluate_data_drift()"
    )
    assert "[ALERT:" in captured, (
        "Logger must emit [ALERT:...] lines from notification_service.send_alert()"
    )
