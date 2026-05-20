"""
Baseline initialization script — computes per-feature statistics from a training
file and stores them in the DuckDB `feature_stats` table.

Run from apps/backend/:
    python scripts/init_baseline.py --input /path/to/training.csv
    python scripts/init_baseline.py --input /path/to/training.parquet
    python scripts/init_baseline.py --input /path/to/training.csv --db /custom/vigilant.db

The script is idempotent: it clears the table before inserting, so re-running
with updated training data always produces a clean, consistent baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

# Allow running as `python scripts/init_baseline.py` from apps/backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Database  # noqa: E402 — must come after sys.path fix
from core.logger import get_logger  # noqa: E402

_logger = get_logger("vigilant.init_baseline")

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "core" / "ml_engine" / "schema.yaml"
_NUM_BINS = 50
_BACKEND_PORT = int(os.getenv("PORT", "8000"))


def _assert_backend_down() -> None:
    """Abort early if the FastAPI backend appears to be running (DuckDB allows only one writer)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", _BACKEND_PORT)) == 0:
            print(
                f"\nERROR: Backend is running on port {_BACKEND_PORT}. "
                "DuckDB only allows one writer — stop the backend first "
                "(Ctrl+C / kill the uvicorn process) then re-run.\n",
                file=sys.stderr,
            )
            sys.exit(1)


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as fh:
        return yaml.safe_load(fh)


def _numeric_stats(series: pl.Series) -> dict:
    # strict=False converts non-numeric sentinels (e.g. "na" in proto) to null;
    # fill_null(0.0) matches the inference-time null_strategy: fill_zero in schema.yaml
    arr = series.cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
    counts, bin_edges = np.histogram(arr, bins=_NUM_BINS)
    return {
        "type": "numeric",
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "count": int(len(arr)),
        "histogram": {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
        },
    }


def _categorical_stats(series: pl.Series) -> dict:
    total = series.len()
    vc = series.cast(pl.Utf8).value_counts(sort=True)
    # Polars value_counts returns columns [<name>, "count"]
    val_col, cnt_col = vc.columns[0], vc.columns[1]
    distribution = {
        str(row[val_col]): round(row[cnt_col] / total, 8)
        for row in vc.iter_rows(named=True)
    }
    return {
        "type": "categorical",
        "distribution": distribution,
        "count": total,
    }


def run(input_path: Path, db_path: str | None = None) -> None:
    _assert_backend_down()
    _logger.info("Loading training data from {}", input_path)

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif suffix in (".csv", ".tsv"):
        df = pl.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported format '{suffix}'. Use .csv or .parquet")

    _logger.info("Dataset: {} rows × {} columns", df.height, df.width)

    schema = _load_schema()
    numeric_features: dict = schema["features"].get("numeric", {})
    categorical_features: dict = schema["features"].get("categorical", {})

    db = Database(db_path)
    db.startup()

    try:
        # Wipe previous baselines so re-runs are fully idempotent
        db.execute("DELETE FROM feature_stats")
        _logger.info("Cleared existing feature_stats rows")

        inserted = 0

        for feature_name in numeric_features:
            if feature_name not in df.columns:
                _logger.warning(
                    "Numeric feature '{}' not found in dataset — skipping", feature_name
                )
                continue
            stats = _numeric_stats(df[feature_name])
            db.execute(
                "INSERT INTO feature_stats (feature_name, stats_json) VALUES (?, ?)",
                [feature_name, json.dumps(stats)],
            )
            _logger.info(
                "  [numeric]      {} — mean={}, std={}", feature_name, stats["mean"], stats["std"]
            )
            inserted += 1

        for feature_name in categorical_features:
            if feature_name not in df.columns:
                _logger.warning(
                    "Categorical feature '{}' not found in dataset — skipping", feature_name
                )
                continue
            stats = _categorical_stats(df[feature_name])
            db.execute(
                "INSERT INTO feature_stats (feature_name, stats_json) VALUES (?, ?)",
                [feature_name, json.dumps(stats)],
            )
            _logger.info(
                "  [categorical]  {} — {} categories",
                feature_name,
                len(stats["distribution"]),
            )
            inserted += 1

        _logger.info("Done. {} features stored in feature_stats.", inserted)

    finally:
        db.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize DuckDB feature baselines from training data"
    )
    parser.add_argument(
        "--input", required=True, help="Path to .csv or .parquet training file"
    )
    parser.add_argument(
        "--db",
        default=None,
        help=(
            "DuckDB file path "
            "(default: core/database/vigilant.db relative to apps/backend/)"
        ),
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    run(path, args.db)
