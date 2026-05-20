# Backend — VigilantMLOps

FastAPI service for ML monitoring, drift detection, and alerting.

---

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> The project uses `pyproject.toml` / `poetry.lock` as the source of truth. If you use Poetry: `poetry install`.

**3. Run the development server**

```bash
make dev-backend
```

This starts the FastAPI app with hot reload at `http://localhost:8000`.

---

## The 3 Pillars

### 1. Drift Detection
Monitors statistical shifts between the training reference distribution and incoming production data. Uses PSI (Population Stability Index) and the KS-test to detect when feature distributions have changed enough to invalidate model assumptions.

### 2. Performance Monitoring
Tracks model quality over time — accuracy, F1, precision, recall, and confusion matrix deltas. Flags performance decay when live metrics fall below thresholds relative to the pre-production baseline.

### 3. System Health
Monitors infrastructure-level signals: API latency, throughput, and schema consistency. Detects issues like slow DB queries or unexpected schema skew before they affect model outputs.

---

## Procedures

Procedures are defined in [core/procedures.yaml](core/procedures.yaml) and govern how the alerting engine responds to each incident type.

```yaml
procedures:
  system_latency:
    action: "REFETCH_DB"
    risk: "low"
    auto_trigger: true
  schema_skew:
    action: "REFETCH_SCHEMA"
    risk: "low"
    auto_trigger: true
  data_drift:
    action: "TICKET_ONLY"
    risk: "high"
    auto_trigger: false
  performance_drop:
    action: "TICKET_ONLY"
    risk: "high"
    auto_trigger: false
```

| Risk Level | Behavior |
|---|---|
| **Low** | `auto_trigger: true` — the system resolves automatically (e.g. re-fetches data or refreshes schema). No human intervention required. |
| **High** | `auto_trigger: false` — a ticket is created in the `incidents` table and escalated for human review. The system does not self-heal. |

---

## Testing

**Run all tests (unit + E2E)**

```bash
make test
```

Test files live under `tests/`:

| File | Coverage |
|---|---|
| `test_endpoints.py` | API route responses |
| `test_reporter_logic.py` | Report generation logic |
| `test_performance.py` | Performance service metrics |
| `test_alerting.py` | Alerting engine threshold logic |
| `test_infra.py` | System health checks |
| `test_e2e_flow.py` | Full end-to-end pipeline |

---

## Database Management

The DB is a local DuckDB file managed by `core/db_manager.py`. Use the following make targets:

| Command | Description |
|---|---|
| `make db-init` | Apply any pending migrations (idempotent — safe to run repeatedly) |
| `make db-reset` | Drop all tables and re-apply all migrations from scratch |
| `make db-status` | Show applied migration history and any pending versions |

The DB path defaults to `core/database/vigilant.db` and can be overridden with the `VIGILANT_DB_PATH` environment variable.

**Schema overview** (v1 migration):

| Table | Purpose |
|---|---|
| `reports` | Pre-production and live evaluation metrics |
| `incidents` | Triggered alerts awaiting human review |
| `production_log` | Incoming feature data from live traffic |
| `alerts` | Alert messages with severity metadata |

---

## Seeding

To populate the DB with synthetic evaluation data for local development:

```bash
make seed
```

This runs the full pipeline: `evaluate-data → evaluate-model → evaluate-drift` across 3 batches. Individual stages can be skipped with `ARGS="--skip <stage>"`.
