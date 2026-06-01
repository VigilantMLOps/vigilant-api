"""Dual-backend database client — PostgreSQL (OLTP) + ClickHouse (OLAP)."""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import clickhouse_connect
import psycopg2
import psycopg2.extras

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

# INSERT INTO <table> (c1, c2, …) VALUES (…) — capture table + column list so
# we can hand them to clickhouse-connect's bulk insert API.
_INSERT_RE = re.compile(
    r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(',
    re.IGNORECASE,
)


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
        self._ch_config: dict | None = None
        self._default_model_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Open the PostgreSQL connection, capture ClickHouse config, apply schemas.

        A long-lived PostgreSQL connection is fine (psycopg2 + autocommit is
        thread-safe for our usage). ClickHouse uses a fresh client per call —
        clickhouse-connect's HTTP client errors with 'concurrent queries within
        the same session' when shared across FastAPI's request threadpool.
        urllib3 pools the underlying socket so per-call clients are cheap.
        """
        self._pg = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "vigilant"),
            user=os.getenv("POSTGRES_USER", "vigilant"),
            password=os.getenv("POSTGRES_PASSWORD", "vigilant"),
        )
        self._pg.autocommit = True
        _logger.info("PostgreSQL connection established.")

        self._ch_config = {
            "host":     os.getenv("CLICKHOUSE_HOST", "localhost"),
            "port":     int(os.getenv("CLICKHOUSE_PORT", "8123")),
            "database": os.getenv("CLICKHOUSE_DB", "vigilant"),
            "username": os.getenv("CLICKHOUSE_USER", "default"),
            "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        }
        # Sanity-check the config by opening + closing a probe client.
        probe = clickhouse_connect.get_client(**self._ch_config)
        probe.close()
        _logger.info("ClickHouse connection verified.")

        self._apply_pg_schema()
        self._apply_ch_schema()
        self._resolve_default_model_id()

    def shutdown(self) -> None:
        if self._pg is not None:
            self._pg.close()
            self._pg = None
        self._ch_config = None
        self._default_model_id = None

    # ------------------------------------------------------------------
    # Default model resolution
    # ------------------------------------------------------------------

    DEFAULT_MODEL_NAME = "Malicious detector"
    DEFAULT_MODEL_VERSION = "v1"

    @property
    def default_model_id(self) -> str:
        """UUID of the canonical (name, version) model row. Resolved once at
        startup; raises if startup hasn't run or the seed row is missing."""
        if self._default_model_id is None:
            raise RuntimeError(
                "default_model_id not resolved — startup() must run first and "
                f"a models row with ({self.DEFAULT_MODEL_NAME!r}, "
                f"{self.DEFAULT_MODEL_VERSION!r}) must exist."
            )
        return self._default_model_id

    def _resolve_default_model_id(self) -> None:
        row = self.fetchone(
            "SELECT model_id FROM models WHERE model_name = ? AND model_version = ?",
            [self.DEFAULT_MODEL_NAME, self.DEFAULT_MODEL_VERSION],
        )
        if row is None:
            _logger.warning(
                "No models row for ({}, {}) — default_model_id unresolved.",
                self.DEFAULT_MODEL_NAME, self.DEFAULT_MODEL_VERSION,
            )
            return
        self._default_model_id = row["model_id"]
        _logger.info("Resolved default_model_id = {}", self._default_model_id)

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

    @contextmanager
    def _ch_client(self):
        """Construct a fresh ClickHouse client for one operation, then close.

        clickhouse-connect's HTTP client is not safe for concurrent use across
        threads/requests — it raises 'concurrent queries within the same
        session'. Creating a per-call client side-steps that; urllib3 keeps
        the underlying TCP socket warm via its connection pool.
        """
        if self._ch_config is None:
            raise RuntimeError("Database not started — call startup() first.")
        client = clickhouse_connect.get_client(**self._ch_config)
        try:
            yield client
        finally:
            client.close()

    def _ch_execute(self, sql: str, params: list) -> None:
        stripped = sql.strip()

        # DELETE FROM <table> (no WHERE) → TRUNCATE.
        # ClickHouse mutations are asynchronous; TRUNCATE is immediate and is
        # the correct operation for resetting a full table (production_log reset).
        m = _BARE_DELETE_RE.match(stripped)
        if m:
            with self._ch_client() as ch:
                ch.command(f"TRUNCATE TABLE IF EXISTS {m.group(1)}")
            return

        # INSERT INTO <table> (c1, c2, …) VALUES (?, …)
        # clickhouse-connect uses the bulk insert API: table name + column
        # list + a list of rows. Each call is one HTTP POST, no shared state.
        m = _INSERT_RE.match(stripped)
        if m:
            table = m.group(1)
            columns = [c.strip() for c in m.group(2).split(",")]
            with self._ch_client() as ch:
                ch.insert(table, [params], column_names=columns)
            return

        with self._ch_client() as ch:
            ch.command(stripped)

    def _ch_fetchall(self, sql: str, params: list) -> list[dict]:
        # No call site currently passes parameters to a ClickHouse SELECT,
        # so we forward the SQL as-is. Add `parameters={...}` here later if
        # that changes — clickhouse-connect uses {name:Type} placeholders.
        with self._ch_client() as ch:
            result = ch.query(sql)
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

    def _apply_ch_schema(self) -> None:
        if not _CH_SCHEMA.exists():
            _logger.warning("ClickHouse schema not found: {}", _CH_SCHEMA)
            return
        with self._ch_client() as ch:
            for stmt in _split_statements(_CH_SCHEMA.read_text()):
                ch.command(stmt)
        _logger.info("ClickHouse schema applied.")


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file into individual statements, ignoring comment lines."""
    sql = re.sub(r'--[^\n]*', '', sql)
    return [s.strip() for s in sql.split(';') if s.strip()]


# Module-level singleton — imported as `from core.database import db` everywhere.
db = Database()