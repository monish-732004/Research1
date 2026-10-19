"""Stage B — Partition completeness and gap-filling.

For each pod that is active (state ∈ ACTIVE_STATES), builds the expected
hourly timeline and flags hours where the pod disappears from the data.

Outputs
-------
  data/interim/pod_activity_mask.parquet
    pod_uid, rel_hour, is_present, is_gap_filled

  outputs/logs/completeness_report.json
    partition-level and gap statistics

Usage
-----
    python -m src.preprocess.completeness --trace v2026
    python -m src.preprocess.completeness --trace v2026 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

from src.config import (
    INTERIM_DIR, LOG_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    TRACE_DAYS, TRACE_HOURS_PER_DAY,
    TRACE_END_HOUR, NETWORK_DAYS,
)

EXPECTED_POD_PARTITIONS    = TRACE_DAYS * TRACE_HOURS_PER_DAY        # 4440
EXPECTED_NETWORK_PARTITIONS = NETWORK_DAYS * TRACE_HOURS_PER_DAY     # 168


def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    out_mask    = INTERIM_DIR / "pod_activity_mask.parquet"
    out_report  = LOG_DIR / "completeness_report.json"

    if out_mask.exists() and not force:
        rows = duckdb.execute(
            f"SELECT count(*) FROM read_parquet('{out_mask}')"
        ).fetchone()[0]
        print(f"  [completeness] output exists ({rows:,} rows) — skipping")
        return {"rows_out": rows, "skipped": True}

    pod_src = INTERIM_DIR / "pod_hourly_canonical.parquet"
    if not pod_src.exists():
        raise FileNotFoundError(
            f"{pod_src} not found. Run canonicalize first."
        )

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # ── 1. Partition completeness (global) ────────────────────────────────────
    actual_partitions = con.execute(
        f"SELECT count(DISTINCT (day, hour)) FROM read_parquet('{pod_src}')"
    ).fetchone()[0]

    partition_report = {
        "pod_hourly": {
            "expected": EXPECTED_POD_PARTITIONS,
            "actual": actual_partitions,
            "missing": EXPECTED_POD_PARTITIONS - actual_partitions,
        },
    }

    # ── 2. Per-pod activity mask ──────────────────────────────────────────────
    # Build expected rows: for every pod, generate all rel_hour values in
    # [first_active_hour, last_active_hour] (inclusive).
    # Then left-join with actual to mark presence / gap.
    sql_mask = f"""
    WITH active AS (
        SELECT pod_uid, rel_hour
        FROM read_parquet('{pod_src}')
        WHERE state IN ('PENDING', 'RUNNING')
    ),
    pod_windows AS (
        SELECT
            pod_uid,
            min(rel_hour) AS first_hour,
            max(rel_hour) AS last_hour
        FROM active
        GROUP BY pod_uid
    ),
    -- Generate all hours in [first_hour, last_hour] per pod
    expected AS (
        SELECT p.pod_uid, h.rel_hour
        FROM pod_windows p,
             LATERAL (
                 SELECT unnest(range(p.first_hour, p.last_hour + 1)) AS rel_hour
             ) h
    ),
    actual_distinct AS (
        SELECT DISTINCT pod_uid, rel_hour FROM active
    )
    SELECT
        e.pod_uid,
        e.rel_hour,
        (a.pod_uid IS NOT NULL)        AS is_present,
        (a.pod_uid IS NULL)            AS is_gap_filled
    FROM expected e
    LEFT JOIN actual_distinct a USING (pod_uid, rel_hour)
    ORDER BY e.pod_uid, e.rel_hour
    """

    con.execute(f"COPY ({sql_mask}) TO '{out_mask}' (FORMAT PARQUET)")
    mask_rows = con.execute(f"SELECT count(*) FROM read_parquet('{out_mask}')").fetchone()[0]

    # ── 3. Gap statistics ─────────────────────────────────────────────────────
    gap_stats = con.execute(f"""
        SELECT
            count(DISTINCT pod_uid) FILTER (WHERE NOT is_present) AS pods_with_gaps,
            sum(CASE WHEN NOT is_present THEN 1 ELSE 0 END)        AS total_gap_hours,
            count(*) FILTER (WHERE is_present)                     AS present_hours,
            count(*)                                               AS total_expected_hours
        FROM read_parquet('{out_mask}')
    """).fetchone()

    report = {
        "trace": trace,
        "partitions": partition_report,
        "gaps": {
            "pods_with_gaps":     gap_stats[0],
            "total_gap_hours":    gap_stats[1],
            "present_hours":      gap_stats[2],
            "total_expected_hours": gap_stats[3],
            "gap_rate": round(gap_stats[1] / max(gap_stats[3], 1), 6),
        },
    }
    out_report.write_text(json.dumps(report, indent=2))

    print(
        f"  [completeness] mask: {mask_rows:,} rows  "
        f"| gaps: {gap_stats[1]:,} hours across {gap_stats[0]:,} pods"
    )
    return {"rows_out": mask_rows, "skipped": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check trace partition completeness.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
