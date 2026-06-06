"""
Baseline initialization script — computes per-feature statistics from a training
file and stores them on the target model's row as `models.baseline`.

Run from the project root:
    python scripts/init_baseline.py --input /path/to/training.csv
    python scripts/init_baseline.py --input /path/to/training.parquet

The script is idempotent: each run overwrites models.baseline and
models.schema_yaml for the target model_id.

The backend can be running while this script executes — PostgreSQL supports
concurrent writers, unlike the old DuckDB setup.

Both PostgreSQL and ClickHouse must be reachable (configured via env vars).
Connection defaults:
    POSTGRES_HOST=localhost  POSTGRES_PORT=5432  POSTGRES_DB=vigilant
    POSTGRES_USER=vigilant   POSTGRES_PASSWORD=vigilant
    CLICKHOUSE_HOST=localhost CLICKHOUSE_PORT=8123 CLICKHOUSE_DB=vigilant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import Database  # noqa: E402
from core.logger import get_logger  # noqa: E402
from repositories import ModelRepository  # noqa: E402

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


def run(input_path: Path, model_id: str | None = None) -> None:
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
        models = ModelRepository(db)
        # Default to the canonical seed model (Malicious detector / v1) when
        # no model_id is given. The seed row is created by the v4 migration
        # and resolved at db.startup().
        if model_id is None:
            model_id = db.default_model_id

        baseline: dict[str, dict] = {}

        for feature_name in numeric_features:
            if feature_name not in df.columns:
                _logger.warning("Numeric feature '{}' not in dataset — skipping", feature_name)
                continue
            stats = _numeric_stats(df[feature_name])
            baseline[feature_name] = stats
            _logger.info(
                "  [numeric]      {} — mean={}, std={}",
                feature_name, stats["mean"], stats["std"],
            )

        for feature_name in categorical_features:
            if feature_name not in df.columns:
                _logger.warning("Categorical feature '{}' not in dataset — skipping", feature_name)
                continue
            stats = _categorical_stats(df[feature_name])
            baseline[feature_name] = stats
            _logger.info(
                "  [categorical]  {} — {} categories",
                feature_name, len(stats["distribution"]),
            )

        models.set_baseline(model_id, baseline)
        models.set_schema_yaml(model_id, schema)
        _logger.info(
            "Done. {} features stored in models.baseline (model_id='{}').",
            len(baseline), model_id,
        )

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
        default=None,
        help="Model UUID to store baselines under. Defaults to the seed model"
             " resolved by (model_name='Malicious detector', model_version='v1').",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    run(path, model_id=args.model_id)