"""Stage C — Fleet capacity + job summary + allocation efficiency.

Outputs (data/processed/):
  hourly_fleet_capacity.parquet      — per-hour total GPUs available + server counts
  job_type_summary.parquet           — job counts + GPU-hours by type/priority/spec
  server_allocation_efficiency.parquet — requested vs capacity per server

Usage
-----
    python -m src.preprocess.aggregate --trace v2026
    python -m src.preprocess.aggregate --trace v2026 --force
"""

from __future__ import annotations

import argparse
from typing import Any

import duckdb

from src.config import (
    INTERIM_DIR, PROCESSED_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    return con


def _s(p) -> str:
    return str(p).replace("\\", "/")


# ── C1: per-hour fleet capacity ───────────────────────────────────────────────

def aggregate_fleet_capacity(con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = PROCESSED_DIR / "hourly_fleet_capacity.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    src = _s(INTERIM_DIR / "server_hourly_canonical.parquet")
    out_s = _s(out)

    print("  [fleet_capacity] aggregating …")
    con.execute(f"""
        COPY (
            SELECT
                day,
                hour_int,
                rel_hour,
                SUM(gpu_count)              AS total_gpus_available,
                COUNT(DISTINCT server_id)   AS active_servers,
                COUNT(DISTINCT cluster_id)  AS active_clusters
            FROM read_parquet('{src}')
            GROUP BY day, hour_int, rel_hour
            ORDER BY rel_hour
        ) TO '{out_s}' (FORMAT PARQUET)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [fleet_capacity] {rows:,} hours → {out}")
    return {"output": str(out), "rows": rows}


# ── C2: job-level GPU-hours by type/priority/spec ────────────────────────────

def aggregate_job_type_summary(con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = PROCESSED_DIR / "job_type_summary.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    src = _s(INTERIM_DIR / "job_summary_canonical.parquet")
    out_s = _s(out)

    print("  [job_type_summary] aggregating …")
    con.execute(f"""
        COPY (
            SELECT
                job_type_public,
                priority_class,
                gpu_spec_public,
                is_genai_request,
                is_deferrable,
                COUNT(*)                                                AS job_count,
                SUM(gpu_hours_requested)                               AS total_gpu_hours_requested,
                AVG(gpu_request)                                       AS mean_gpu_request,
                PERCENTILE_CONT(0.5) WITHIN GROUP
                    (ORDER BY gpu_request)                             AS p50_gpu_request,
                PERCENTILE_CONT(0.95) WITHIN GROUP
                    (ORDER BY gpu_request)                             AS p95_gpu_request,
                AVG(duration_hours)                                    AS mean_duration_hours,
                AVG(schedule_delay_sec)                                AS mean_schedule_delay_sec,
                SUM(CASE WHEN ready_status    THEN 1 ELSE 0 END)      AS ready_count,
                SUM(CASE WHEN schedule_status THEN 1 ELSE 0 END)      AS scheduled_count
            FROM read_parquet('{src}')
            GROUP BY
                job_type_public, priority_class, gpu_spec_public,
                is_genai_request, is_deferrable
        ) TO '{out_s}' (FORMAT PARQUET)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [job_type_summary] {rows:,} groups → {out}")
    return {"output": str(out), "rows": rows}


# ── C3: server allocation efficiency ─────────────────────────────────────────

def aggregate_server_efficiency(con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = PROCESSED_DIR / "server_allocation_efficiency.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    srv = _s(INTERIM_DIR / "server_hourly_canonical.parquet")
    job = _s(INTERIM_DIR / "job_summary_canonical.parquet")
    out_s = _s(out)

    print("  [server_efficiency] joining server capacity + job requests …")
    con.execute(f"""
        COPY (
            WITH capacity AS (
                SELECT
                    server_id,
                    ANY_VALUE(gpu_spec_public)          AS gpu_spec_public,
                    ANY_VALUE(server_tier)              AS server_tier,
                    -- each row = 1 GPU-hour per gpu on that server that hour
                    SUM(gpu_count * 1.0)                AS total_capacity_gpu_hours
                FROM read_parquet('{srv}')
                GROUP BY server_id
            ),
            requested AS (
                SELECT
                    server_id,
                    SUM(gpu_hours_requested)            AS total_requested_gpu_hours,
                    COUNT(*)                            AS job_count
                FROM read_parquet('{job}')
                WHERE server_id IS NOT NULL
                GROUP BY server_id
            )
            SELECT
                c.server_id,
                c.gpu_spec_public,
                c.server_tier,
                c.total_capacity_gpu_hours,
                COALESCE(r.total_requested_gpu_hours, 0.0) AS total_requested_gpu_hours,
                COALESCE(r.job_count, 0)                    AS job_count,
                CASE
                    WHEN c.total_capacity_gpu_hours = 0 THEN NULL
                    ELSE COALESCE(r.total_requested_gpu_hours, 0.0)
                         / c.total_capacity_gpu_hours
                END                                         AS allocation_efficiency
            FROM capacity c
            LEFT JOIN requested r USING (server_id)
        ) TO '{out_s}' (FORMAT PARQUET)
    """)

    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [server_efficiency] {rows:,} servers → {out}")
    return {"output": str(out), "rows": rows}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    print(f"  [aggregate] trace={trace}")
    con = _con()
    results: dict[str, Any] = {}

    results["fleet_capacity"]    = aggregate_fleet_capacity(con, force=force)
    results["job_type_summary"]  = aggregate_job_type_summary(con, force=force)
    results["server_efficiency"] = aggregate_server_efficiency(con, force=force)

    skipped = all(r.get("skipped") for r in results.values())
    return {"skipped": skipped, "outputs": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate fleet capacity and job summaries.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
