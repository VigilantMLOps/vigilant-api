"""DuckDB persistence layer — connection, schema bootstrap, and query helpers."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

from core.logger import get_logger

_logger = get_logger("vigilant.database")
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB_PATH = "core/database/vigilant.db"


class Database:
    """
    Thin wrapper around a persistent DuckDB connection.

    Lifecycle
    ---------
    Call startup() inside the FastAPI lifespan (opens the file and applies
    the schema); call shutdown() on teardown to flush and close.

    All query helpers use positional ? parameters — never format SQL strings
    directly with user-supplied values.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        resolved = db_path or os.getenv("VIGILANT_DB_PATH", str(_DEFAULT_DB_PATH))
        self._path = str(resolved)
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Open DuckDB file, load JSON extension, and apply schema DDL."""
        os.makedirs("core/database/", exist_ok=True)
        self._conn = duckdb.connect(self._path)
        _logger.info("Database directory verified and connection initialized.")
        try:
            self._conn.execute("LOAD json")
        except Exception:
            pass  # JSON is built-in from DuckDB 1.0+; safe to skip
        self._conn.execute(_SCHEMA_PATH.read_text())

    def shutdown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def _active(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("Database not started — call startup() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: list | None = None) -> None:
        """Execute a write statement with optional positional parameters."""
        self._active.execute(sql, params or [])

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        """Run a SELECT and return every row as a dict."""
        cursor = self._active.execute(sql, params or [])
        if cursor.description is None:
            return []
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        """Run a SELECT and return the first row as a dict, or None."""
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None


# Module-level singleton — import `db` everywhere else
db = Database()
