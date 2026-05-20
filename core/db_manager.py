"""Database migration manager for VigilantMLOps.

Usage (from apps/backend/):
    python -m core.db_manager           # apply pending migrations
    python -m core.db_manager init      # same as above
    python -m core.db_manager reset     # drop all tables and re-init
    python -m core.db_manager status    # show applied migration history
    python -m core.db_manager --db /path/to/other.db init
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

# ── Path resolution ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB = Path(os.path.join(BASE_DIR, "database", "vigilant.db"))


def _resolve_db_path() -> Path:
    raw = os.getenv("VIGILANT_DB_PATH")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else Path.cwd() / p
    return _DEFAULT_DB


# ── Migration registry ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


# To add a new migration, append a new Migration entry with the next version
# number.  Statements are executed in order and the version is recorded in
# schema_version only after all statements succeed.
#
# Column additions example:
#   Migration(
#       version=2,
#       description="Add source_ip column to alerts",
#       statements=(
#           "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS source_ip VARCHAR",
#       ),
#   ),
MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        description="Initial schema: reports, incidents, production_log, alerts",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id     VARCHAR PRIMARY KEY,
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                report_type   VARCHAR NOT NULL,
                model_version VARCHAR,
                metrics       JSON,
                artifacts     JSON
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id   VARCHAR PRIMARY KEY,
                timestamp     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                severity      VARCHAR NOT NULL,
                incident_type VARCHAR NOT NULL,
                description   TEXT,
                status        VARCHAR NOT NULL DEFAULT 'TRIGGERED'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS production_log (
                log_id      VARCHAR PRIMARY KEY,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                features    JSON NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id  VARCHAR PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                level     VARCHAR NOT NULL,
                message   TEXT NOT NULL,
                metadata  JSON
            )
            """,
        ),
    ),
    Migration(
        version=2,
        description="Add feature_stats table for drift baseline statistics",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS feature_stats (
                feature_name VARCHAR PRIMARY KEY,
                stats_json   JSON NOT NULL,
                updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
        ),
    ),
]

# ── Internal helpers ──────────────────────────────────────────────────────────

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description VARCHAR NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


SEED_PATH = os.path.join(BASE_DIR, "database", "seed_data.sql")
_SEED_SQL = Path(SEED_PATH)


def _connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("LOAD json")
    except Exception:
        pass  # built-in since DuckDB 1.0; safe to ignore
    return conn


def _current_version(conn: duckdb.DuckDBPyConnection) -> int:
    conn.execute(_SCHEMA_VERSION_DDL)
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version"
    ).fetchone()
    return row[0] if row else 0


def _apply_seed(conn: duckdb.DuckDBPyConnection) -> None:
    """Load seed_data.sql when either feature_stats or reports is empty.

    All statements use INSERT OR REPLACE so re-running is safe.
    """
    if not _SEED_SQL.exists():
        return

    n_stats = conn.execute("SELECT COUNT(*) FROM feature_stats").fetchone()[0]
    n_reports = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    if n_stats > 0 and n_reports > 0:
        print(f"  feature_stats ({n_stats}) and reports ({n_reports}) already populated — skipping seed.")
        return

    inserted = 0
    for line in _SEED_SQL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("--"):
            conn.execute(line)
            inserted += 1

    print(f"  Seeded {inserted} statements from {_SEED_SQL.name}.")


# ── Public API ────────────────────────────────────────────────────────────────

def init(db_path: Path) -> None:
    """Apply all pending migrations in ascending version order, then seed if fresh."""
    conn = _connect(db_path)
    try:
        current = _current_version(conn)
        pending = [m for m in MIGRATIONS if m.version > current]

        if not pending:
            print(f"Database is up to date (version {current}).")
        else:
            for migration in pending:
                print(f"Applying v{migration.version}: {migration.description}")
                for stmt in migration.statements:
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                    [migration.version, migration.description],
                )
                print(f"  v{migration.version} applied.")

            final = _current_version(conn)
            print(f"Database ready at version {final}.")

        _apply_seed(conn)
    finally:
        conn.close()


def reset(db_path: Path) -> None:
    """Drop every table (including schema_version) then re-apply all migrations."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        for (table,) in rows:
            conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            print(f"Dropped: {table}")
    finally:
        conn.close()

    init(db_path)


def status(db_path: Path) -> None:
    """Print applied migration history and list any pending versions."""
    if not db_path.exists():
        print(f"Database does not exist yet: {db_path}")
        return

    conn = _connect(db_path)
    try:
        current = _current_version(conn)
        rows = conn.execute(
            "SELECT version, description, applied_at"
            " FROM schema_version ORDER BY version"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No migrations applied yet.")
    else:
        header = f"{'Ver':>4}  {'Applied At':<28}  Description"
        print(header)
        print("-" * len(header))
        for version, description, applied_at in rows:
            print(f"{version:>4}  {str(applied_at):<28}  {description}")

    pending = [m for m in MIGRATIONS if m.version > current]
    if pending:
        print(f"\nPending: {[m.version for m in pending]}")
    else:
        print(f"\nAll {len(MIGRATIONS)} migration(s) applied.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m core.db_manager",
        description="VigilantMLOps database migration manager",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="init",
        choices=["init", "reset", "status"],
        help="init (default) | reset | status",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        help="Override DB path (default: VIGILANT_DB_PATH env or core/database/vigilant.db)",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db) if args.db else _resolve_db_path()
    print(f"DB: {db_path}")

    if args.command == "init":
        init(db_path)
    elif args.command == "reset":
        reset(db_path)
    elif args.command == "status":
        status(db_path)


if __name__ == "__main__":
    main()
