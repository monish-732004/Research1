"""Stage A — Schema canonicalization.

Reads the three confirmed v2026 tables and writes canonical Parquet files
to data/interim/:

  server_hourly_canonical.parquet   — server capacity per hour (143M rows)
  job_summary_canonical.parquet     — one row per job / pod (40M rows)
  network_hourly_canonical.parquet  — network I/O per server per hour (5M rows)

All heavy lifting is DuckDB SQL — no Python row loops.

Usage
-----
    python -m src.preprocess.canonicalize --trace v2026
    python -m src.preprocess.canonicalize --trace v2026 --force
"""

from __future__ import annotations

import argparse
from typing import Any

import duckdb

from src.config import (
    INTERIM_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    table_glob,
)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    return con


def _glob(table: str, trace: str) -> str:
    return table_glob(table, trace).replace("\\", "/")


# ── A1: server_hourly ─────────────────────────────────────────────────────────

def canonicalize_server_hourly(trace: str, con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = INTERIM_DIR / "server_hourly_canonical.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    glob = _glob("asi_opensource_server_hourly", trace)
    out_s = str(out).replace("\\", "/")

    print(f"  [server_hourly] reading {glob} …")
    con.execute(f"""
        COPY (
            SELECT
                server_id,
                cluster_id,
                asw_id,
                gpu_spec_public,
                gpu_count,
                cpu_capacity_cores,
                day,
                CAST(hour AS INTEGER)                      AS hour_int,
                day * 24 + CAST(hour AS INTEGER)           AS rel_hour,
                CASE
                    WHEN gpu_count <= 2 THEN 'small'
                    WHEN gpu_count <= 8 THEN 'medium'
                    ELSE 'large'
                END                                        AS server_tier
            FROM read_parquet('{glob}')
        ) TO '{out_s}' (FORMAT PARQUET, ROW_GROUP_SIZE 200000)
    """)

    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [server_hourly] wrote {row_count:,} rows → {out}")
    return {"output": str(out), "rows": row_count}


# ── A2: job_execution_summary ─────────────────────────────────────────────────

def canonicalize_job_summary(trace: str, con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = INTERIM_DIR / "job_summary_canonical.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    glob = _glob("asi_opensource_job_execution_summary", trace)
    out_s = str(out).replace("\\", "/")

    print(f"  [job_summary] reading {glob} …")
    con.execute(f"""
        COPY (
            SELECT
                pod_id,
                workload_id,
                server_id,
                gpu_spec_public,
                priority_class,
                job_type_public,
                model_type_public,
                is_genai_request,
                gpu_request,
                duration_hours,
                schedule_delay_sec,
                ready_delay_sec,
                ready_status,
                schedule_status,
                -- derived columns
                gpu_request * duration_hours                AS gpu_hours_requested,
                (
                    job_type_public NOT IN ('online_inference')
                    AND priority_class = 'LP'
                )                                           AS is_batch,
                (
                    job_type_public NOT IN ('online_inference')
                    AND priority_class = 'LP'
                    AND ready_status = TRUE
                )                                           AS is_deferrable,
                -- truncation flags (Task D)
                (schedule_delay_sec = 0 AND ready_delay_sec = 0) AS left_truncated,
                (duration_hours > 4000)                     AS right_truncated
            FROM read_parquet('{glob}')
        ) TO '{out_s}' (FORMAT PARQUET, ROW_GROUP_SIZE 200000)
    """)

    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [job_summary] wrote {row_count:,} rows → {out}")
    return {"output": str(out), "rows": row_count}


# ── A3: network_hourly ────────────────────────────────────────────────────────

def canonicalize_network_hourly(trace: str, con: duckdb.DuckDBPyConnection, *, force: bool = False) -> dict:
    out = INTERIM_DIR / "network_hourly_canonical.parquet"
    if out.exists() and not force:
        return {"skipped": True, "output": str(out)}

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    glob = _glob("asi_opensource_network_hourly", trace)
    out_s = str(out).replace("\\", "/")

    print(f"  [network_hourly] reading {glob} …")
    con.execute(f"""
        COPY (
            SELECT
                server_id,
                rx_gibps_avg,
                tx_gibps_avg,
                day,
                CAST(hour AS INTEGER)              AS hour_int,
                day * 24 + CAST(hour AS INTEGER)   AS rel_hour,
                TRUE                               AS in_network_window
            FROM read_parquet('{glob}')
        ) TO '{out_s}' (FORMAT PARQUET, ROW_GROUP_SIZE 200000)
    """)

    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_s}')").fetchone()[0]
    print(f"  [network_hourly] wrote {row_count:,} rows → {out}")
    return {"output": str(out), "rows": row_count}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    con = _con()
    results: dict[str, Any] = {}

    results["server_hourly"] = canonicalize_server_hourly(trace, con, force=force)
    results["job_summary"]   = canonicalize_job_summary(trace, con, force=force)
    results["network_hourly"]= canonicalize_network_hourly(trace, con, force=force)

    skipped = all(r.get("skipped") for r in results.values())
    return {"skipped": skipped, "outputs": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Canonicalize raw trace tables.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true",
                        help="Rewrite outputs even if they already exist.")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
