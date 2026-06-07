# Vigilant API

FastAPI backend for the Vigilant MLOps platform. Provides endpoints for ML model monitoring, pre-production evaluation, real-time feature drift detection (PSI / KS / Chi²), incident management, and system health.

## Architecture

- **OLTP (PostgreSQL)** — model registry, evaluation reports, incident lifecycle. Tables that need transactions, updates, or relational integrity.
- **OLAP (ClickHouse)** — production traffic log, alert history, drift results, report metrics, LLM traces. Append-only, high-volume analytics.
- **Repository layer** — every SQL statement lives in `repositories/`. Services and routes depend on these interfaces, not on the database wrapper directly.
- **In-memory test backend** — `tests/fake_database.py` implements the same `Database` interface against SQLite, so the full suite runs in ~0.1s with no real databases needed.

## Tech Stack

- **Python 3.12** + **FastAPI** + **Uvicorn**
- **PostgreSQL 16** — OLTP persistence
- **ClickHouse 24** — OLAP analytics
- **Polars** — data processing
- **scikit-learn / scipy / numpy** — ML evaluation and drift statistics
- **Poetry** — dependency management
- **Docker Compose** — local stack orchestration
- **Caddy** — reverse proxy + automatic HTTPS in production

## Quick Start (Local)

```bash
cp .env.example .env   # fill in any values you want to override
make up                # builds and starts postgres, clickhouse, and the api
make logs SERVICE=api  # tail the api logs
```

The api container's entrypoint applies any pending migrations (`python -m core.db_manager init`) before `uvicorn` starts, so a fresh stack comes up at the latest schema version without manual steps.

API is reachable at `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and `/redoc`.

Stop with `make down`.

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
| `GET` | `/api/v1/incidents` | List incidents |
| `GET` | `/api/v1/incidents/{incident_id}` | Get incident details |
| `GET` | `/api/v1/telemetry/status` | Telemetry / alert manager status |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port uvicorn listens on |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins. Use `*` to allow all |
| `MODEL_API_URL` | `http://model-api:8001` | Base URL of the model inference API |
| `POSTGRES_HOST` | `localhost` | PostgreSQL hostname (set to `postgres` inside compose) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `vigilant` | PostgreSQL database name |
| `POSTGRES_USER` | `vigilant` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `vigilant` | PostgreSQL password |
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse hostname (set to `clickhouse` inside compose) |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port |
| `CLICKHOUSE_DB` | `vigilant` | ClickHouse database name |
| `CLICKHOUSE_USER` | `default` | ClickHouse user |
| `CLICKHOUSE_PASSWORD` | (empty) | ClickHouse password |
| `DATA_DIR` | `../vigilant-mlops/artifacts/data` | Host path mounted at `/data` in the api container |
| `CADDY_DOMAIN` | (unset) | Domain Caddy terminates TLS for the api (e.g. `vigilant-api.duckdns.org`); only used when the `prod` compose profile is active |
| `CADDY_UI_DOMAIN` | (unset) | Domain Caddy terminates TLS for the UI, reverse-proxied to host port 8080 (e.g. `vigilant-ui.duckdns.org`); only used when the `prod` compose profile is active |

A minimal local `.env`:

```dotenv
POSTGRES_PASSWORD=vigilant
CLICKHOUSE_PASSWORD=
MODEL_API_URL=http://host.docker.internal:8001
DATA_DIR=/tmp
```

## Makefile Targets

| Target | Description |
|---|---|
| `make up` | Start postgres, clickhouse, and the api |
| `make down` | Stop and remove containers |
| `make logs [SERVICE=api]` | Tail logs (one service or all) |
| `make dev` | Run the api with hot reload (requires running databases) |
| `make test` | Run the pytest suite |
| `make db-init` | Apply pending migrations (idempotent) |
| `make db-reset` | Drop all tables and re-apply migrations from scratch |
| `make db-status` | Show applied migration history |
| `make migration NAME=...` | Scaffold a new migration |
| `make init-baseline INPUT=...` | Compute per-feature baselines from a training file |
| `make seed [ARGS="--skip <stage>"]` | Seed the DB via the API (evaluate-data → evaluate-model → evaluate-drift) |
| `make seed-snapshot` | `make seed` then `make db-dump` |
| `make db-dump [DIR=./snapshots]` | Snapshot the postgres+clickhouse volumes for transfer |
| `make db-restore [DIR=./snapshots] [FORCE=1]` | Restore both volumes from a snapshot |

## Database Migrations

PostgreSQL, ClickHouse, and the Python migration runner are version-tracked together in `schema_migrations`. Pending migrations are applied automatically when the api container starts (via `entrypoint.sh`); the Makefile targets are for the host-side dev workflow.

```bash
make migration NAME=add_some_table   # scaffold v00X_add_some_table_pg.sql, _ch.sql, .py
# edit the files...
make db-init                          # apply all pending versions (idempotent)
make db-status                        # confirm
```

Migration files live in:
- `core/database/postgres/migrations/v00X_*.sql`
- `core/database/clickhouse/migrations/v00X_*.sql`
- `core/database/migrations/v00X_*.py` (optional Python runner for cross-DB backfills)

## Testing

```bash
make test
```

All 23 tests run against an in-memory SQLite `FakeDatabase` — no PostgreSQL or ClickHouse needed. The suite covers route handlers (`httpx.AsyncClient` + ASGI transport), service logic, repository contracts, and a full end-to-end drift pipeline. Total runtime: ~0.1s.

## Seeding and Snapshots

`make seed` drives the api through `evaluate-data` → `evaluate-model` → `evaluate-drift`, so a few prerequisites must be in place locally:

- the stack must be running (`make up`)
- the **ml-serve** model service must be reachable at `MODEL_API_URL` (the seed script can start a local checkout at the path configured in `config/seed.json`)
- the raw datasets referenced by `config/reporter.json` must exist at `DATA_DIR`

Two-step flow for getting realistic data onto a fresh instance (local or remote):

```bash
# 1. On your laptop — populate the local DB via the API:
make seed-snapshot
# Produces ./snapshots/pg.tgz and ./snapshots/ch.tgz

# 2. Transfer + restore on the target host:
scp snapshots/*.tgz target-host:~/vigilant-api/snapshots/
ssh target-host 'cd ~/vigilant-api && make db-restore'
```

The two databases form a single logical dataset (model UUIDs in PG are referenced by CH rows), so snapshots always travel as a pair.

## Deployment

The production VM runs the same `docker-compose.yml` with the `prod` profile, which adds a Caddy sidecar:

```bash
docker compose --profile prod up -d
```

Caddy reverse-proxies the api on ports 80/443 and obtains a Let's Encrypt certificate for `CADDY_DOMAIN` automatically.

A GitHub Actions workflow (`.github/workflows/deploy.yml`) ships new code to the VM on every tag matching `deploy.api.*`:

```bash
git tag deploy.api.v4
git push origin deploy.api.v4
# → SSHs in, pulls, rebuilds api + caddy, restarts
```

Required GitHub repo secrets: `VM_HOST`, `VM_USER`, `VM_SSH_KEY`.

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

| Table | Owner | Purpose |
|---|---|---|
| `models` | PostgreSQL | Model registry with baseline / data_eval / pre_prod_eval / schema snapshots as JSONB |
| `reports` | PostgreSQL | Evaluation event log (PRE_PROD, DATA_EVAL, DRIFT) with per-type payload in `content` JSONB |
| `incidents` | PostgreSQL | Alert lifecycle with mutable status |
| `production_log` | ClickHouse | Append-only inference traffic buffer |
| `alerts` | ClickHouse | Notification history with severity metadata |
| `drift_results` | ClickHouse | Per-feature PSI checks over time |
| `report_metrics` | ClickHouse | Analytics mirror of `reports` |
| `llm_traces` | ClickHouse | LLM call traces |
| `schema_migrations` | PostgreSQL | Migration version tracking |

## Project Structure

```
api/v1/              # Route handlers (incidents, monitoring, reporter, telemetry)
config/              # reporter.json, seed.json — runtime configuration
core/
├── database/        # Dual-backend wrapper, schemas, migrations
│   ├── postgres/    # PG schema + per-version migrations
│   ├── clickhouse/  # CH schema + per-version migrations
│   └── migrations/  # Python runners for cross-DB backfills
├── db_manager.py    # Migration apply / status CLI
├── logger.py        # Loguru-based logger
└── ml_engine/       # Schema validation engine
repositories/        # Per-domain data-access layer (ReportRepository, ModelRepository, …)
services/            # Business logic (alerting, drift detection, reporting, …)
scripts/
├── seed.py          # Driver for the evaluate-data → evaluate-model → evaluate-drift seed flow
├── db_snapshot.sh   # docker volume dump / restore for both DBs
├── migration.py     # Scaffold a new migration
└── init_baseline.py # Compute per-feature baselines from a training file
notebooks/           # Drift payload fixtures used by scripts/seed.py
tests/               # pytest suite; uses FakeDatabase (SQLite) — no real DB required
.github/workflows/   # CI/CD (tag-triggered deploy to the Oracle VM)
Caddyfile            # HTTPS reverse proxy (prod compose profile)
docker-compose.yml   # postgres + clickhouse + api (+ caddy under prod profile)
Dockerfile           # Python 3.12-slim image build for the api service
entrypoint.sh        # Container entrypoint — runs migrations then launches uvicorn
.env.example         # Template for the .env consumed by docker compose
pyproject.toml       # Poetry project metadata + dependencies
poetry.lock          # Pinned dependency tree
main.py              # FastAPI app factory + router registration
```
