#!/usr/bin/env bash
# Snapshot / restore the PostgreSQL + ClickHouse docker volumes as one unit.
#
#   scripts/db_snapshot.sh dump    [SNAPSHOT_DIR]
#   scripts/db_snapshot.sh restore [SNAPSHOT_DIR]   (set FORCE=1 to overwrite non-empty volumes)
#
# The two databases form a single logical dataset — model UUIDs minted in
# PostgreSQL are referenced by rows in ClickHouse — so they are always
# snapshotted and restored together. Services are stopped during the copy so
# ClickHouse flushes cleanly (a hot copy of its data dir can corrupt parts).
set -euo pipefail

cmd="${1:-}"
SNAPSHOT_DIR="${2:-./snapshots}"
HELPER_IMAGE="alpine"

log() { printf '\033[36m%s\033[0m\n' "$*"; }
err() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; }

# Resolve the real volume names from compose itself — no guessing at how the
# project name is normalized.
_resolve_volumes() {
    local json project shortnames
    if ! json="$(docker compose config --format json 2>/dev/null)"; then
        err "could not run 'docker compose config' — run from the repo root with a valid docker-compose.yml"
        exit 1
    fi
    project="$(printf '%s' "$json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("name",""))')"
    PG_VOL="${project}_postgres_data"
    CH_VOL="${project}_clickhouse_data"
}

_archive() { # volume, filename
    docker run --rm -v "$1":/from:ro -v "$(cd "$SNAPSHOT_DIR" && pwd)":/to \
        "$HELPER_IMAGE" tar czf "/to/$2" -C /from .
}

_extract() { # volume, filename
    docker run --rm -v "$1":/to -v "$(cd "$SNAPSHOT_DIR" && pwd)":/from:ro \
        "$HELPER_IMAGE" sh -c "rm -rf /to/* /to/..?* /to/.[!.]* 2>/dev/null; tar xzf /from/$2 -C /to"
}

_volume_nonempty() { # volume
    docker volume inspect "$1" >/dev/null 2>&1 || return 1
    [ -n "$(docker run --rm -v "$1":/v:ro "$HELPER_IMAGE" sh -c 'ls -A /v 2>/dev/null')" ]
}

dump() {
    _resolve_volumes
    for v in "$PG_VOL" "$CH_VOL"; do
        docker volume inspect "$v" >/dev/null 2>&1 || { err "volume '$v' not found — has the stack been started and seeded?"; exit 1; }
    done

    mkdir -p "$SNAPSHOT_DIR"
    log "Stopping services for a consistent copy..."
    docker compose stop

    log "Archiving $PG_VOL -> $SNAPSHOT_DIR/pg.tgz"
    _archive "$PG_VOL" pg.tgz
    log "Archiving $CH_VOL -> $SNAPSHOT_DIR/ch.tgz"
    _archive "$CH_VOL" ch.tgz

    log "Restarting services..."
    docker compose start

    log "Snapshot ready. Copy these to the target host's repo, then run 'make db-restore':"
    log "  $SNAPSHOT_DIR/pg.tgz"
    log "  $SNAPSHOT_DIR/ch.tgz"
}

restore() {
    _resolve_volumes
    [ -f "$SNAPSHOT_DIR/pg.tgz" ] || { err "missing $SNAPSHOT_DIR/pg.tgz"; exit 1; }
    [ -f "$SNAPSHOT_DIR/ch.tgz" ] || { err "missing $SNAPSHOT_DIR/ch.tgz"; exit 1; }

    if [ "${FORCE:-0}" != "1" ]; then
        if _volume_nonempty "$PG_VOL" || _volume_nonempty "$CH_VOL"; then
            err "target volumes already contain data. Refusing to overwrite — re-run with FORCE=1 to wipe and restore."
            exit 1
        fi
    fi

    docker volume create "$PG_VOL" >/dev/null
    docker volume create "$CH_VOL" >/dev/null

    log "Stopping services..."
    docker compose stop 2>/dev/null || true

    log "Restoring $PG_VOL <- pg.tgz"
    _extract "$PG_VOL" pg.tgz
    log "Restoring $CH_VOL <- ch.tgz"
    _extract "$CH_VOL" ch.tgz

    log "Starting services..."
    docker compose up -d

    log "Done. Verify: curl -s localhost:8000/api/v1/reports/latest"
}

case "$cmd" in
    dump)    dump ;;
    restore) restore ;;
    *) err "usage: $(basename "$0") {dump|restore} [SNAPSHOT_DIR]"; exit 1 ;;
esac
