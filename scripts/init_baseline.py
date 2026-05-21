"""
Baseline initialization script — computes per-feature statistics from a training
file and stores them in the PostgreSQL `feature_stats` table.

Run from the project root:
    python scripts/init_baseline.py --input /path/to/training.csv
    python scripts/init_baseline.py --input /path/to/training.parquet

The script is idempotent: it clears all rows for the target model before
inserting, so re-running with updated training data produces a clean baseline.

The backend can be running while this script executes — PostgreSQL supports
concurrent writers, unlike the old DuckDB setup.

Both PostgreSQL and ClickHouse must be reachable (configured via env vars).
Connection defaults:
    POSTGRES_HOST=localhost  POSTGRES_PORT=5432  POSTGRES_DB=vigilant
    POSTGRES_USER=vigilant   POSTGRES_PASSWORD=vigilant
    CLICKHOUSE_HOST=localhost CLICKHOUSE_PORT=9000 CLICKHOUSE_DB=vigilant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Database  # noqa: E402
from core.logger import get_logger  # noqa: E402

_logger = get_logger("vigilant.init_baseline")

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "core" / "ml_engine" / "schema.yaml"
_NUM_BINS = 50


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as fh:
        return yaml.safe_load(fh)


def _numeric_stats(series: pl.Series) -> dict:
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


def run(input_path: Path, model_id: str = "default") -> None:
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

    db = Database()
    db.startup()

    try:
        # Clear existing baselines for this model so re-runs are idempotent.
        # feature_stats is routed to PostgreSQL; DELETE without WHERE removes all rows.
        db.execute(
            "DELETE FROM feature_stats WHERE model_id = ?",
            [model_id],
        )
        _logger.info("Cleared existing feature_stats for model_id='{}'", model_id)

        inserted = 0

        for feature_name in numeric_features:
            if feature_name not in df.columns:
                _logger.warning("Numeric feature '{}' not in dataset — skipping", feature_name)
                continue
            stats = _numeric_stats(df[feature_name])
            db.execute(
                "INSERT INTO feature_stats (model_id, feature_name, stats_json)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT (model_id, feature_name) DO UPDATE SET"
                "   stats_json = EXCLUDED.stats_json,"
                "   updated_at = NOW()",
                [model_id, feature_name, json.dumps(stats)],
            )
            _logger.info(
                "  [numeric]      {} — mean={}, std={}",
                feature_name, stats["mean"], stats["std"],
            )
            inserted += 1

        for feature_name in categorical_features:
            if feature_name not in df.columns:
                _logger.warning("Categorical feature '{}' not in dataset — skipping", feature_name)
                continue
            stats = _categorical_stats(df[feature_name])
            db.execute(
                "INSERT INTO feature_stats (model_id, feature_name, stats_json)"
                " VALUES (?, ?, ?)"
                " ON CONFLICT (model_id, feature_name) DO UPDATE SET"
                "   stats_json = EXCLUDED.stats_json,"
                "   updated_at = NOW()",
                [model_id, feature_name, json.dumps(stats)],
            )
            _logger.info(
                "  [categorical]  {} — {} categories",
                feature_name, len(stats["distribution"]),
            )
            inserted += 1

        _logger.info("Done. {} features stored in feature_stats (model_id='{}').", inserted, model_id)

    finally:
        db.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL feature baselines from training data"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to .csv or .parquet training file",
    )
    parser.add_argument(
        "--model-id",
        default="default",
        help="Model ID to store baselines under (default: 'default')",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    run(path, model_id=args.model_id)