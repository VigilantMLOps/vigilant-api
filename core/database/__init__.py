"""Dual-backend database client — PostgreSQL (OLTP) + ClickHouse (OLAP)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import psycopg2
import psycopg2.extras
from clickhouse_driver import Client as ClickHouseClient

from core.logger import get_logger

_logger = get_logger("vigilant.database")

_PG_SCHEMA = Path(__file__).parent / "postgres" / "schema.sql"
_CH_SCHEMA = Path(__file__).parent / "clickhouse" / "schema.sql"

# All tables owned by ClickHouse — everything else goes to PostgreSQL.
_CLICKHOUSE_TABLES = frozenset({
    "production_log",
    "production_log_buffer",
    "alerts",
    "drift_results",
    "report_metrics",
    "llm_traces",
})

# Matches the first table name in any DML statement.
_TABLE_RE = re.compile(
    r'\b(?:INSERT\s+INTO|TRUNCATE\s+TABLE|UPDATE|DELETE\s+FROM|FROM)\s+(\w+)',
    re.IGNORECASE,
)

# DELETE FROM <table> with no WHERE clause — used by reset_production_log().
_BARE_DELETE_RE = re.compile(r'DELETE\s+FROM\s+(\w+)\s*$', re.IGNORECASE)

# INSERT INTO <table> (...) VALUES (...) — strip the VALUES part for CH driver.
_INSERT_VALUES_RE = re.compile(r'\s+VALUES\s*\([\s\S]*\)\s*$', re.IGNORECASE)


def _table_of(sql: str) -> str | None:
    m = _TABLE_RE.search(sql)
    return m.group(1).lower() if m else None


def _is_clickhouse(sql: str) -> bool:
    table = _table_of(sql)
    return bool(table and table in _CLICKHOUSE_TABLES)


def _to_pg_params(sql: str) -> str:
    """Convert DuckDB/SQLite ? positional markers to psycopg2 %s style."""
    return sql.replace("?", "%s")


class Database:
    """
    Thin dual-backend persistence wrapper.

    Routes each SQL statement to the correct database based on the target
    table name — no changes required in services or routes.

      PostgreSQL (OLTP):  reports, incidents, feature_stats, models
      ClickHouse (OLAP):  production_log, alerts, drift_results,
                          report_metrics, llm_traces

    Public interface is identical to the old DuckDB wrapper:
      startup() / shutdown() / execute() / fetchall() / fetchone()

    Connection env vars:
      POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
      CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_DB, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD
    """

    def __init__(self) -> None:
        self._pg: psycopg2.extensions.connection | None = None
        self._ch: ClickHouseClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Open both database connections and apply schemas."""
        self._pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "vigilant"),
            user=os.getenv("POSTGRES_USER", "vigilant"),
            password=os.getenv("POSTGRES_PASSWORD", "vigilant"),
        )
        self._pg.autocommit = True
        _logger.info("PostgreSQL connection established.")

        self._ch = ClickHouseClient(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
            database=os.getenv("CLICKHOUSE_DB", "vigilant"),
            user=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
        )
        _logger.info("ClickHouse connection established.")

        self._apply_pg_schema()
        self._apply_ch_schema()

    def shutdown(self) -> None:
        if self._pg is not None:
            self._pg.close()
            self._pg = None
        if self._ch is not None:
            self._ch.disconnect()
            self._ch = None

    # ------------------------------------------------------------------
    # Public query interface — same as the old DuckDB wrapper
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: list | None = None) -> None:
        """Execute a write statement (INSERT / UPDATE / DELETE / TRUNCATE)."""
        params = params or []
        if _is_clickhouse(sql):
            self._ch_execute(sql, params)
        else:
            self._pg_execute(sql, params)

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        """Run a SELECT and return every row as a list of dicts."""
        params = params or []
        if _is_clickhouse(sql):
            return self._ch_fetchall(sql, params)
        return self._pg_fetchall(sql, params)

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # PostgreSQL internals
    # ------------------------------------------------------------------

    @property
    def _pg_conn(self) -> psycopg2.extensions.connection:
        if self._pg is None:
            raise RuntimeError("Database not started — call startup() first.")
        return self._pg

    def _pg_execute(self, sql: str, params: list) -> None:
        with self._pg_conn.cursor() as cur:
            cur.execute(_to_pg_params(sql), params)

    def _pg_fetchall(self, sql: str, params: list) -> list[dict]:
        with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_to_pg_params(sql), params)
            return [dict(row) for row in cur.fetchall()]

    def _apply_pg_schema(self) -> None:
        if not _PG_SCHEMA.exists():
            _logger.warning("PostgreSQL schema not found: {}", _PG_SCHEMA)
            return
        with self._pg_conn.cursor() as cur:
            cur.execute(_PG_SCHEMA.read_text())
        _logger.info("PostgreSQL schema applied.")

    # ------------------------------------------------------------------
    # ClickHouse internals
    # ------------------------------------------------------------------

    @property
    def _ch_conn(self) -> ClickHouseClient:
        if self._ch is None:
            raise RuntimeError("Database not started — call startup() first.")
        return self._ch

    def _ch_execute(self, sql: str, params: list) -> None:
        stripped = sql.strip()

        # DELETE FROM <table> (no WHERE) → TRUNCATE.
        # ClickHouse mutations are asynchronous; TRUNCATE is immediate and is
        # the correct operation for resetting a full table (production_log reset).
        m = _BARE_DELETE_RE.match(stripped)
        if m:
            self._ch_conn.execute(f"TRUNCATE TABLE IF EXISTS {m.group(1)}")
            return

        # INSERT INTO <table> (...) VALUES (?, …)
        # clickhouse-driver wants the VALUES clause stripped; params are passed
        # as a list of rows: [[v1, v2, …]].
        if re.match(r'INSERT\s+INTO', stripped, re.IGNORECASE):
            prefix = _INSERT_VALUES_RE.sub('', stripped)
            self._ch_conn.execute(prefix + " VALUES", [params])
            return

        self._ch_conn.execute(stripped, params)

    def _ch_fetchall(self, sql: str, params: list) -> list[dict]:
        rows, col_types = self._ch_conn.execute(sql, params, with_column_types=True)
        columns = [name for name, _ in col_types]
        return [dict(zip(columns, row)) for row in rows]

    def _apply_ch_schema(self) -> None:
        if not _CH_SCHEMA.exists():
            _logger.warning("ClickHouse schema not found: {}", _CH_SCHEMA)
            return
        for stmt in _split_statements(_CH_SCHEMA.read_text()):
            self._ch_conn.execute(stmt)
        _logger.info("ClickHouse schema applied.")


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into individual statements, ignoring comment lines."""
    sql = re.sub(r'--[^\n]*', '', sql)
    return [s.strip() for s in sql.split(';') if s.strip()]


# Module-level singleton — imported as `from core.database import db` everywhere.
db = Database()