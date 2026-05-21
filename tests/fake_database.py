"""SQLite-backed in-memory database for unit tests.

Implements the same interface as core.database.Database so services and routes
can be tested without running PostgreSQL or ClickHouse.

All tables — PostgreSQL and ClickHouse alike — are stored in a single SQLite
in-memory instance. No routing is needed for tests; every INSERT/SELECT/UPDATE/
DELETE is executed directly against SQLite using the same ? positional params
that DuckDB used, which SQLite also natively supports.
"""
from __future__ import annotations

import sqlite3


class FakeDatabase:
    """Drop-in test replacement for core.database.Database.

    Usage in fixtures:
        db = FakeDatabase()
        db.startup()
        yield db
        db.shutdown()
    """

    # Minimal schema: keeps column names that existing service SQL references.
    # Types are all TEXT/INTEGER/REAL; SQLite ignores unknown type names gracefully.
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS reports (
        report_id       TEXT    PRIMARY KEY,
        timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
        report_type     TEXT    NOT NULL,
        model_version   TEXT,
        metrics         TEXT,
        artifacts       TEXT,
        accuracy        REAL,
        precision_score REAL,
        recall          REAL,
        f1_score        REAL,
        roc_auc         REAL,
        avg_precision   REAL,
        split           TEXT,
        stage           TEXT,
        n_rows          INTEGER,
        n_features      INTEGER,
        imbalance_ratio REAL,
        duplicate_rows  INTEGER,
        missing_cells   INTEGER,
        class_distribution TEXT
    );

    CREATE TABLE IF NOT EXISTS incidents (
        incident_id   TEXT    PRIMARY KEY,
        timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
        severity      TEXT    NOT NULL,
        incident_type TEXT    NOT NULL,
        description   TEXT,
        status        TEXT    NOT NULL DEFAULT 'TRIGGERED'
    );

    CREATE TABLE IF NOT EXISTS production_log (
        log_id      TEXT    PRIMARY KEY,
        received_at TEXT    NOT NULL DEFAULT (datetime('now')),
        features    TEXT    NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS production_log_buffer (
        log_id      TEXT    PRIMARY KEY,
        received_at TEXT    NOT NULL DEFAULT (datetime('now')),
        features    TEXT    NOT NULL DEFAULT ''
    );

    CREATE TRIGGER IF NOT EXISTS trg_buffer_to_log
    AFTER INSERT ON production_log_buffer
    BEGIN
        INSERT INTO production_log (log_id, received_at, features)
        VALUES (NEW.log_id, NEW.received_at, NEW.features);
    END;

    CREATE TABLE IF NOT EXISTS alerts (
        alert_id   TEXT    PRIMARY KEY,
        timestamp  TEXT    NOT NULL DEFAULT (datetime('now')),
        level      TEXT    NOT NULL,
        event_type TEXT    NOT NULL DEFAULT '',
        message    TEXT    NOT NULL,
        metadata   TEXT    DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS feature_stats (
        model_id     TEXT    NOT NULL DEFAULT 'default',
        feature_name TEXT    NOT NULL,
        stats_json   TEXT    NOT NULL DEFAULT '{}',
        updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (model_id, feature_name)
    );

    CREATE TABLE IF NOT EXISTS report_metrics (
        report_id       TEXT    PRIMARY KEY,
        timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
        report_type     TEXT    NOT NULL DEFAULT '',
        model_id        TEXT    NOT NULL DEFAULT '',
        model_version   TEXT    NOT NULL DEFAULT '',
        accuracy        REAL    DEFAULT 0,
        precision_score REAL    DEFAULT 0,
        recall          REAL    DEFAULT 0,
        f1_score        REAL    DEFAULT 0,
        roc_auc         REAL    DEFAULT 0,
        avg_precision   REAL    DEFAULT 0,
        n_rows          INTEGER DEFAULT 0,
        n_features      INTEGER DEFAULT 0,
        imbalance_ratio REAL    DEFAULT 0,
        duplicate_rows  INTEGER DEFAULT 0,
        missing_cells   INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS drift_results (
        drift_id            TEXT    PRIMARY KEY,
        checked_at          TEXT    NOT NULL DEFAULT (datetime('now')),
        model_id            TEXT    NOT NULL DEFAULT '',
        model_version       TEXT    NOT NULL DEFAULT '',
        feature_name        TEXT    NOT NULL,
        method              TEXT    NOT NULL DEFAULT '',
        psi_score           REAL    NOT NULL DEFAULT 0,
        pvalue              REAL,
        status              TEXT    NOT NULL DEFAULT '',
        n_production_rows   INTEGER DEFAULT 0
    );
    """

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle (same as Database)
    # ------------------------------------------------------------------

    def startup(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)

    def shutdown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Query interface (same as Database)
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: list | None = None) -> None:
        assert self._conn is not None, "FakeDatabase not started."
        self._conn.execute(sql, params or [])
        self._conn.commit()

    def fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        assert self._conn is not None, "FakeDatabase not started."
        cursor = self._conn.execute(sql, params or [])
        return [dict(row) for row in cursor.fetchall()]

    def fetchone(self, sql: str, params: list | None = None) -> dict | None:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None
