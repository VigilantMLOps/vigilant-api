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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg2
import psycopg2.extras
from clickhouse_driver import Client as ClickHouseClient

_BASE     = Path(__file__).resolve().parent
_PG_SCHEMA = _BASE / "database" / "postgres"   / "schema.sql"
_CH_SCHEMA = _BASE / "database" / "clickhouse" / "schema.sql"


# ── Migration registry ────────────────────────────────────────────────────────
# To add a new migration, append a Migration with the next version number and
# list the SQL files or inline ALTER statements to run on each backend.
# The version is recorded in PostgreSQL's schema_migrations only after all
# statements on both databases succeed.

@dataclass(frozen=True)
class Migration:
    version: int
    description: str


MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description=(
            "Initial dual-DB schema: PostgreSQL (models, feature_stats, reports, incidents)"
            " + ClickHouse (production_log, alerts, drift_results, report_metrics, llm_traces)"
        ),
    ),
]


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
    return ClickHouseClient(
        host=os.getenv("CLICKHOUSE_HOST", "localhost"),
        port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
        database=os.getenv("CLICKHOUSE_DB", "vigilant"),
        user=os.getenv("CLICKHOUSE_USER", "default"),
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

            print("  → PostgreSQL ...")
            with pg.cursor() as cur:
                cur.execute(_PG_SCHEMA.read_text())
            print("  ✓ PostgreSQL done.")

            print("  → ClickHouse ...")
            for stmt in _split_statements(_CH_SCHEMA.read_text()):
                ch.execute(stmt)
            print("  ✓ ClickHouse done.")

            _record(pg, m)
            print(f"  v{m.version} recorded in schema_migrations.")

        print(f"\nBoth databases ready at schema version {_current_version(pg)}.")
    finally:
        pg.close()
        ch.disconnect()


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
        for (table,) in ch.execute("SHOW TABLES"):
            ch.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  Dropped CH table: {table}")

    finally:
        pg.close()
        ch.disconnect()

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
