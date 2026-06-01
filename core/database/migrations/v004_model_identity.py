"""Migration v4 Python runner — propagate the newly-minted model UUID into
ClickHouse tables that weren't truncated by the SQL portion.

ClickHouse forbids ALTER UPDATE on columns that are part of the table's
ORDER BY sort key. Tables where model_id IS in the sort key
(production_log / drift_results / report_metrics / llm_traces) are
truncated in v004_model_identity_ch.sql instead; their data is derived from
PostgreSQL events and gets repopulated by the next /reporter/evaluate-*
run with the new UUID baked in.

`alerts` is the only table where model_id is a regular column (sort key is
(timestamp, level, alert_id)), so we can backfill it in place.
"""
from __future__ import annotations


def run(pg, ch) -> None:
    """Read the new model UUID from PG and backfill alerts.model_id."""
    with pg.cursor() as cur:
        cur.execute(
            "SELECT model_id FROM models"
            " WHERE model_name = %s AND model_version = %s",
            ["Malicious detector", "v1"],
        )
        row = cur.fetchone()

    if row is None:
        print("    (no Malicious detector v1 row found; skipping CH backfill)")
        return

    new_id = row[0]
    print(f"    backfilling alerts.model_id = {new_id}")

    # ALTER UPDATE in ClickHouse is async — the mutation queues and runs in
    # the background. Subsequent SELECTs may briefly show old values until
    # the mutation completes.
    ch.command(
        "ALTER TABLE alerts UPDATE model_id = %(id)s"
        " WHERE model_id = '' OR model_id = 'default'",
        parameters={"id": new_id},
    )
    print("      ✓ alerts")
