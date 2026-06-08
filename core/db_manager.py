"""
Database migration manager for VigilantMLOps (PostgreSQL + ClickHouse).

Usage (from project root):
    python -m core.db_manager           # apply all pending migrations
    python -m core.db_manager init      # same as above
    python -m core.db_manager reset     # drop all tables and re-apply schemas
    python -m core.db_manager status    # show migration history

Connection is configured via environment variables:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_DB, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import clickhouse_connect
import psycopg2
import psycopg2.extras
from clickhouse_connect.driver.client import Client as ClickHouseClient

_BASE     = Path(__file__).resolve().parent
_PG_SCHEMA = _BASE / "database" / "postgres"   / "schema.sql"
_CH_SCHEMA = _BASE / "database" / "clickhouse" / "schema.sql"
_PG_MIGRATIONS = _BASE / "database" / "postgres" / "migrations"


# ── Migration registry ────────────────────────────────────────────────────────
# Each Migration carries per-version SQL paths for whichever backends it
# changes. `None` means "no changes on that side for this version" — the
# runner skips it. The migration is only recorded in schema_migrations once
# all listed sides succeed.

@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    pg_sql: Path | None = None
    ch_sql: Path | None = None
    # Optional Python runner that runs after pg_sql + ch_sql. Use this for
    # cross-DB operations (e.g. generating a UUID in PostgreSQL and
    # propagating it into ClickHouse tables in the same migration step).
    # Spec: "module.path:function_name" where the function accepts
    # (pg_connection, ch_client) and returns None.
    runner: str | None = None


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description=(
            "Initial dual-DB schema: PostgreSQL (models, feature_stats, reports, incidents)"
            " + ClickHouse (production_log, alerts, drift_results, report_metrics, llm_traces)"
        ),
        pg_sql=_PG_SCHEMA,
        ch_sql=_CH_SCHEMA,
    ),
    Migration(
        version=2,
        description=(
            "Consolidate per-feature baselines and static evals onto models"
            " (models.baseline / data_eval / pre_prod_eval / schema_yaml; drop feature_stats)"
        ),
        pg_sql=_PG_MIGRATIONS / "v002_consolidate_models.sql",
    ),
    Migration(
        version=3,
        description=(
            "Collapse reports type-specific columns into a single `content` JSONB"
            " (PRE_PROD / DATA_EVAL / DRIFT share one shape; content varies by type)"
        ),
        pg_sql=_PG_MIGRATIONS / "v003_reports_content_jsonb.sql",
    ),
    Migration(
        version=4,
        description=(
            "Model identity: add name + version + UNIQUE on models, replace 'default'"
            " with a generated UUID across PG and CH, collapse report_metrics into content"
        ),
        pg_sql=_PG_MIGRATIONS / "v004_model_identity_pg.sql",
        ch_sql=_BASE / "database" / "clickhouse" / "migrations" / "v004_model_identity_ch.sql",
        runner="core.database.migrations.v004_model_identity:run",
    ),
    Migration(
        version=5,
        description=(
            "RAG observability: add query_text, query_mode, n_retrieved, top_retrieval_score,"
            " retrieval_latency_ms, generation_latency_ms, sources, prompt_version to llm_traces"
        ),
        ch_sql=_BASE / "database" / "clickhouse" / "migrations" / "v005_rag_traces_ch.sql",
    ),
]


# ── Runner loader ─────────────────────────────────────────────────────────────

def _resolve_runner(spec: str) -> Callable:
    """Resolve a 'module.path:function_name' spec to the callable."""
    module_path, _, func_name = spec.partition(":")
    if not module_path or not func_name:
        raise ValueError(f"Invalid runner spec '{spec}' — expected 'module:function'")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# ── Connection helpers ────────────────────────────────────────────────────────

def _pg_connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "vigilant"),
        user=os.getenv("POSTGRES_USER", "vigilant"),
        password=os.getenv("POSTGRES_PASSWORD", "vigilant"),
    )
    conn.autocommit = True
    return conn


def _ch_connect() -> ClickHouseClient:
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "8123")),
        database=os.getenv("CLICKHOUSE_DB", "vigilant"),
        username=os.getenv("CLICKHOUSE_USER", "default"),
        password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    )


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into individual statements, stripping comment lines."""
    sql = re.sub(r'--[^\n]*', '', sql)
    return [s.strip() for s in sql.split(';') if s.strip()]


# ── Version tracking (stored in PostgreSQL schema_migrations) ─────────────────

_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER         PRIMARY KEY,
    description VARCHAR(500)    NOT NULL,
    applied_at  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
)
"""


def _current_version(pg: psycopg2.extensions.connection) -> int:
    with pg.cursor() as cur:
        cur.execute(_VERSION_DDL)
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        return cur.fetchone()[0]


def _record(pg: psycopg2.extensions.connection, m: Migration) -> None:
    with pg.cursor() as cur:
        cur.execute(
            "INSERT INTO schema_migrations (version, description) VALUES (%s, %s)"
            " ON CONFLICT DO NOTHING",
            [m.version, m.description],
        )


# ── Commands ──────────────────────────────────────────────────────────────────

def init() -> None:
    """Apply all pending migrations to both PostgreSQL and ClickHouse."""
    pg = _pg_connect()
    ch = _ch_connect()
    try:
        current = _current_version(pg)
        pending = [m for m in MIGRATIONS if m.version > current]

        if not pending:
            print(f"Both databases are up to date (schema version {current}).")
            return

        for m in pending:
            print(f"Applying v{m.version}: {m.description}")

            if m.pg_sql is not None:
                print(f"  → PostgreSQL ({m.pg_sql.name}) ...")
                with pg.cursor() as cur:
                    cur.execute(m.pg_sql.read_text())
                print("  ✓ PostgreSQL done.")
            else:
                print("  · PostgreSQL — no changes.")

            if m.ch_sql is not None:
                print(f"  → ClickHouse ({m.ch_sql.name}) ...")
                for stmt in _split_statements(m.ch_sql.read_text()):
                    ch.command(stmt)
                print("  ✓ ClickHouse done.")
            else:
                print("  · ClickHouse — no changes.")

            if m.runner is not None:
                print(f"  → runner ({m.runner}) ...")
                _resolve_runner(m.runner)(pg, ch)
                print("  ✓ runner done.")

            _record(pg, m)
            print(f"  v{m.version} recorded in schema_migrations.")

        print(f"\nBoth databases ready at schema version {_current_version(pg)}.")
    finally:
        pg.close()
        ch.close()


def reset() -> None:
    """Drop all tables in both databases, then re-apply all schemas from scratch."""
    pg = _pg_connect()
    ch = _ch_connect()
    try:
        # PostgreSQL: drop all user tables in the public schema
        with pg.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            tables = [r[0] for r in cur.fetchall()]
        for table in tables:
            with pg.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            print(f"  Dropped PG table: {table}")

        # PostgreSQL: drop enum types created by this schema
        with pg.cursor() as cur:
            cur.execute("""
                SELECT typname FROM pg_type
                WHERE typtype = 'e'
                  AND typnamespace = (
                      SELECT oid FROM pg_namespace WHERE nspname = 'public'
                  )
            """)
            enums = [r[0] for r in cur.fetchall()]
        for enum in enums:
            with pg.cursor() as cur:
                cur.execute(f'DROP TYPE IF EXISTS "{enum}" CASCADE')
            print(f"  Dropped PG enum: {enum}")

        # ClickHouse: drop all tables in the current database
        for (table,) in ch.query("SHOW TABLES").result_rows:
            ch.command(f"DROP TABLE IF EXISTS {table}")
            print(f"  Dropped CH table: {table}")

    finally:
        pg.close()
        ch.close()

    init()


def status() -> None:
    """Print the migration history recorded in PostgreSQL."""
    pg = _pg_connect()
    try:
        current = _current_version(pg)
        with pg.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT version, description, applied_at"
                " FROM schema_migrations ORDER BY version"
            )
            rows = cur.fetchall()
    finally:
        pg.close()

    if not rows:
        print("No migrations applied yet.")
    else:
        header = f"{'Ver':>4}  {'Applied At':<32}  Description"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['version']:>4}  {str(row['applied_at']):<32}  {row['description']}")

    pending = [m for m in MIGRATIONS if m.version > current]
    if pending:
        print(f"\nPending: {[m.version for m in pending]}")
    else:
        print(f"\nAll {len(MIGRATIONS)} migration(s) applied.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m core.db_manager",
        description="VigilantMLOps database migration manager (PostgreSQL + ClickHouse)",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="init",
        choices=["init", "reset", "status"],
        help="init (default) | reset | status",
    )
    args = parser.parse_args(argv)

    if args.command == "init":
        init()
    elif args.command == "reset":
        reset()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
