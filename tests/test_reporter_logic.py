"""Logic tests for ReporterService.evaluate_data_drift (a.k.a. check_drift).

Two scenarios are exercised:
  - stable:  production matches reference → no drift detected
  - drifted: production is radically different → drift detected on multiple features

No disk access: DataLoader.load_reference is monkeypatched to return an
in-memory DataFrame built from the seed records below.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from services.data_loader import DataPaths, RawDataPaths
from services.reporter import DriftStatus, ModelAPIConfig, ReporterConfig, ReporterService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY = Path("/tmp")


def _make_test_config() -> ReporterConfig:
    return ReporterConfig(
        data=DataPaths(
            raw=RawDataPaths(unsw_nb15_dir=_DUMMY, ciciot2023_dir=_DUMMY),
            balanced_dir=_DUMMY,
            blacklist=_DUMMY / "blacklist.parquet",
        ),
        model_api=ModelAPIConfig(base_url="http://localhost:9999"),
        categorical_columns=["proto", "state", "source"],
        target_column="target",
    )


# ---------------------------------------------------------------------------
# Seed records (mirroring notebooks/example1.json)
# ---------------------------------------------------------------------------

_REFERENCE_RECORDS = [
    {"flow_duration": 0.02,  "bytes_total": 50,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "na",   "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.05,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 0.10,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "6.0",  "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.30,  "bytes_total": 54,    "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 0.0,  "max": 0.0,   "avg": 0.0,  "std": 0.0, "proto": "17.0", "state": "ACC", "source": "ciciot"},
    {"flow_duration": 0.53,  "bytes_total": 872,   "pkts_total": 9.5,   "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 0.40,  "bytes_total": 870,   "pkts_total": 9.5,   "rate": 10.0,    "srate": 10.0,    "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "unsw"},
    {"flow_duration": 64.0,  "bytes_total": 18700, "pkts_total": 82.0,  "rate": 18.0,    "srate": 18.0,    "drate": 0.0, "min": 54.0, "max": 54.0,  "avg": 54.0, "std": 0.0, "proto": "na",   "state": "na",  "source": "ciciot"},
    {"flow_duration": 64.0,  "bytes_total": 19000, "pkts_total": 82.0,  "rate": 0.0,     "srate": 0.0,     "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "proto": "6.0",  "state": "na",  "source": "unsw"},
    {"flow_duration": 64.5,  "bytes_total": 40000, "pkts_total": 120.0, "rate": 500.0,   "srate": 500.0,   "drate": 0.0, "min": 54.0, "max": 97.0,  "avg": 59.0, "std": 3.0, "proto": "1.0",  "state": "na",  "source": "ciciot"},
    {"flow_duration": 65.0,  "bytes_total": 40500, "pkts_total": 120.0, "rate": 5000.0,  "srate": 5000.0,  "drate": 0.0, "min": 54.0, "max": 200.0, "avg": 59.0, "std": 5.0, "proto": "na",   "state": "na",  "source": "unsw"},
]

# Extreme shift: numeric values 1000× larger, completely different categorical values
_DRIFTED_RECORDS = [
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_check_drift_stable_distribution(monkeypatch):
    """Identical production and reference distributions produce zero drift."""
    ref_df = pl.DataFrame(_REFERENCE_RECORDS)
    current_df = pl.DataFrame(_REFERENCE_RECORDS)

    config = _make_test_config()
    service = ReporterService(config=config, db=None)
    monkeypatch.setattr(service._loader, "load_reference", lambda: ref_df)

    result = service.evaluate_data_drift(current_df)

    assert result.n_features_checked > 0
    assert result.n_drifted == 0
    assert result.drift_rate == 0.0
    assert result.overall_status == DriftStatus.OK
    for feat in result.features:
        # PSI is always non-negative by the information inequality
        assert feat.statistic >= 0.0
        # p-values are probabilities
        if feat.pvalue is not None:
            assert 0.0 <= feat.pvalue <= 1.0


def test_check_drift_drifted_distribution(monkeypatch):
    """Radically different production data triggers drift on multiple features."""
    ref_df = pl.DataFrame(_REFERENCE_RECORDS)
    current_df = pl.DataFrame(_DRIFTED_RECORDS)

    config = _make_test_config()
    service = ReporterService(config=config, db=None)
    monkeypatch.setattr(service._loader, "load_reference", lambda: ref_df)

    result = service.evaluate_data_drift(current_df)

    assert result.n_features_checked > 0
    assert result.n_drifted > 0
    assert result.drift_rate > 0.0
    assert result.overall_status != DriftStatus.OK

    # Every feature result must carry valid score and optional p-value
    for feat in result.features:
        assert feat.statistic >= 0.0
        if feat.pvalue is not None:
            assert 0.0 <= feat.pvalue <= 1.0

    # Numeric features checked via PSI + KS
    numeric_results = [f for f in result.features if f.method == "psi+ks"]
    assert len(numeric_results) > 0

    # Categorical features checked via PSI + Chi²
    categorical_results = [f for f in result.features if f.method == "psi+chi2"]
    assert len(categorical_results) == 3  # proto, state, source
