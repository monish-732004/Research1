"""Stage C — Left/right truncation flagging.

Definitions
-----------
left-truncated  : first appearance at rel_hour == TRACE_START_HOUR (0)
                  AND state at that hour is RUNNING (not PENDING).
                  The pod was already running when the trace began — its
                  full lifetime is unknown.

right-truncated : last appearance at rel_hour == TRACE_END_HOUR (4439)
                  AND state at that hour is RUNNING (never terminal).
                  The pod was still running when the trace ended.

Outputs
-------
  data/interim/pod_truncation_flags.parquet
    pod_uid, left_truncated, right_truncated,
    first_rel_hour, last_rel_hour, terminal_state

  data/interim/pod_survival.parquet   (survival-analysis–ready)
    pod_uid, duration_hours_observed, is_censored

Usage
-----
    python -m src.preprocess.truncation --trace v2026
    python -m src.preprocess.truncation --trace v2026 --force
"""

from __future__ import annotations

import argparse
from typing import Any

import duckdb

from src.config import (
    INTERIM_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    TRACE_START_HOUR, TRACE_END_HOUR,
)


def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_flags    = INTERIM_DIR / "pod_truncation_flags.parquet"
    out_survival = INTERIM_DIR / "pod_survival.parquet"

    if out_flags.exists() and not force:
        rows = duckdb.execute(
            f"SELECT count(*) FROM read_parquet('{out_flags}')"
        ).fetchone()[0]
        print(f"  [truncation] output exists ({rows:,} rows) — skipping")
        return {"rows_out": rows, "skipped": True}

    pod_src = INTERIM_DIR / "pod_hourly_canonical.parquet"
    if not pod_src.exists():
        raise FileNotFoundError(f"{pod_src} not found. Run canonicalize first.")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # ── Pod lifecycle summary ─────────────────────────────────────────────────
    # DuckDB's `arg_min` / `arg_max` return the column value at the row where
    # the ordering column is minimum/maximum — we use it to get state at
    # first and last appearance.
    sql_flags = f"""
    WITH summary AS (
        SELECT
            pod_uid,
            min(rel_hour)                          AS first_rel_hour,
            max(rel_hour)                          AS last_rel_hour,
            arg_min(state, rel_hour)               AS first_state,
            arg_max(state, rel_hour)               AS last_state,
            count(*)                               AS total_hours
        FROM read_parquet('{pod_src}')
        GROUP BY pod_uid
    )
    SELECT
        pod_uid,
        (first_rel_hour = {TRACE_START_HOUR} AND first_state = 'RUNNING') AS left_truncated,
        (last_rel_hour  = {TRACE_END_HOUR}   AND last_state  = 'RUNNING') AS right_truncated,
        first_rel_hour,
        last_rel_hour,
        last_state AS terminal_state,
        total_hours
    FROM summary
    ORDER BY pod_uid
    """

    con.execute(f"COPY ({sql_flags}) TO '{out_flags}' (FORMAT PARQUET)")
    n_flags = con.execute(f"SELECT count(*) FROM read_parquet('{out_flags}')").fetchone()[0]

    # ── Survival table ────────────────────────────────────────────────────────
    # A pod is censored if it is right-truncated (outcome not observed) or
    # left-truncated (start not observed — treat as censored for standard
    # right-censored survival analysis; use left-truncated flag for LTRC models).
    sql_survival = f"""
    SELECT
        pod_uid,
        (last_rel_hour - first_rel_hour + 1)  AS duration_hours_observed,
        right_truncated                        AS is_censored,
        left_truncated,
        first_rel_hour,
        last_rel_hour
    FROM read_parquet('{out_flags}')
    """
    con.execute(f"COPY ({sql_survival}) TO '{out_survival}' (FORMAT PARQUET)")

    # Summary stats
    stats = con.execute(f"""
        SELECT
            sum(left_truncated::INT)  AS n_left,
            sum(right_truncated::INT) AS n_right,
            count(*)                  AS n_total
        FROM read_parquet('{out_flags}')
    """).fetchone()

    print(
        f"  [truncation] {n_flags:,} pods  "
        f"| left-trunc: {stats[0]:,}  right-trunc: {stats[1]:,}"
    )
    return {"rows_out": n_flags, "skipped": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Flag left/right truncated pods.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
