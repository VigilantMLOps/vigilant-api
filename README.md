# Vigilant API

FastAPI backend for the Vigilant MLOps platform. Provides endpoints for ML model monitoring, pre-production evaluation, real-time feature drift detection (PSI / KS / Chi²), incident management, and system health.

## Tech Stack

- **Python 3.12** + **FastAPI** + **Uvicorn**
- **DuckDB** — embedded persistence (reports, incidents, alerts, feature stats)
- **Polars** — data processing
- **scikit-learn / scipy / numpy** — ML evaluation and drift statistics
- **Poetry** — dependency management

## API Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/api/v1/reports/latest` | Most recent evaluation report |
| `GET` | `/api/v1/reports/history` | Last 10 evaluation reports |
| `POST` | `/api/v1/reporter/evaluate-data` | Run data evaluation on all splits |
| `POST` | `/api/v1/reporter/evaluate-model` | Run model evaluation against model API |
| `POST` | `/api/v1/reporter/evaluate-drift` | Run feature drift detection |
| `DELETE` | `/api/v1/reporter/production-log` | Reset production log |
| `GET` | `/api/v1/reporter/feature-stats` | Export baseline feature statistics |
| `GET` | `/api/v1/reporter/model-health` | Proxy model API health check |

Interactive docs available at `/docs` (Swagger) and `/redoc`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port uvicorn listens on |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins. Use `*` to allow all |
| `MODEL_API_URL` | `http://model-api:8001` | Base URL of the model inference API |
| `VIGILANT_DB_PATH` | `core/database/vigilant.db` | Path to the DuckDB database file |
| `DATA_RAW_UNSW_DIR` | `/data/raw/UNSW-NB15` | UNSW-NB15 raw CSV directory |
| `DATA_RAW_CICIOT_DIR` | `/data/raw/CICIoT2023` | CICIoT2023 raw CSV directory |
| `DATA_BALANCED_DIR` | `/data/processed/balanced` | Balanced/processed parquet directory |
| `DATA_BLACKLIST_PATH` | `/data/ip_blacklists/blacklist.parquet` | IP blacklist parquet file |

## Docker

```bash
docker build -t vigilant-api .

docker run -p 8000:8000 \
  -e MODEL_API_URL=http://your-model-api:8001 \
  -e CORS_ORIGINS=https://your-ui.example.com \
  -v /host/path/to/data:/data \
  -v vigilant-db:/app/core/database \
  vigilant-api
```

Data is expected to be mounted at `/data/` with the following layout:

```
/data/
├── raw/
│   ├── UNSW-NB15/          # override with DATA_RAW_UNSW_DIR
│   └── CICIoT2023/         # override with DATA_RAW_CICIOT_DIR
├── processed/
│   └── balanced/           # override with DATA_BALANCED_DIR
└── ip_blacklists/
    └── blacklist.parquet   # override with DATA_BLACKLIST_PATH
```

## Local Development

```bash
poetry install
poetry run uvicorn main:app --reload
```

The server starts at `http://localhost:8000`. Set `MODEL_API_URL` if you need model evaluation endpoints.

## Running Tests

```bash
poetry run pytest
```

## Initializing the Baseline

Run once (with the backend stopped) to compute per-feature statistics from a training file and store them in DuckDB:

```bash
poetry run python scripts/init_baseline.py --input /path/to/training.parquet
```

## The 3 Pillars

**Drift Detection** — monitors statistical shifts between training reference distribution and incoming production data using PSI, KS-test, and Chi² test.

**Performance Monitoring** — tracks accuracy, F1, precision, recall, and confusion matrix deltas over time. Alerts on decay relative to the pre-production baseline.

**System Health** — monitors API latency and schema consistency via middleware. Triggers alerts on slow requests (>500ms) and 5xx errors.

## Alerting Procedures

Defined in `core/procedures.yaml`. Low-risk incidents auto-resolve; high-risk ones create tickets for human review.

| Incident | Risk | Behavior |
|---|---|---|
| `system_latency` | Low | Auto-resolves (re-fetches DB) |
| `schema_skew` | Low | Auto-resolves (refreshes schema) |
| `data_drift` | High | Creates incident ticket |
| `performance_drop` | High | Creates incident ticket |

## Database Schema

| Table | Purpose |
|---|---|
| `reports` | Pre-production and live evaluation metrics |
| `incidents` | Triggered alerts awaiting human review |
| `production_log` | Incoming feature data from live traffic |
| `alerts` | Alert messages with severity metadata |

## Project Structure

```
api/v1/          # Route handlers (incidents, monitoring, reporter, telemetry)
config/          # reporter.json — default reporter configuration
core/
├── database/    # DuckDB connection, schema, seed data
├── logger.py    # Loguru-based logger
└── ml_engine/   # Schema validation engine
services/        # Business logic (alerting, drift detection, reporting, etc.)
scripts/         # One-off operational scripts (init_baseline.py)
tests/           # pytest test suite
```