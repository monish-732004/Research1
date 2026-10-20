"""RQ1 — GPU resource fragmentation analysis.

Computes allocation-based fragmentation (capacity never assigned to any job)
from the three processed tables produced by Phase 2:

  server_allocation_efficiency.parquet  — per-server capacity vs requested
  job_type_summary.parquet              — per (type × priority × spec) group
  hourly_fleet_capacity.parquet         — per-hour fleet totals

NOTE: Utilization-based fragmentation (GPU allocated but underutilised)
requires pod_hourly data (Phase 3b when the 351 GB archive arrives).
The numbers here are a lower bound on true waste.

Metric definitions
------------------
  frag_rate  =  1 - (total_requested_gpu_hours / total_capacity_gpu_hours)
               per server; fleet-level is aggregate of all servers.

All functions return small pandas DataFrames safe to hold in RAM.

Usage
-----
    python -m src.analysis.fragmentation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.config import (
    PROCESSED_DIR, TABLE_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
)


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


def _s(p: Path) -> str:
    return str(p).replace("\\", "/")


_EFF_SRC = PROCESSED_DIR / "server_allocation_efficiency.parquet"
_JTS_SRC = PROCESSED_DIR / "job_type_summary.parquet"
_FLC_SRC = PROCESSED_DIR / "hourly_fleet_capacity.parquet"


# ── Core: per-server allocation gap ──────────────────────────────────────────

def compute_frag_rate(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
) -> pd.DataFrame:
    """Per-server allocation-gap fragmentation rate.

    Returns: server_id, server_tier, gpu_spec_public,
             total_capacity_gpu_hours, total_requested_gpu_hours,
             frag_gpu_hours, frag_rate.
    """
    c = con or _con()
    return c.execute(f"""
        SELECT
            server_id,
            server_tier,
            gpu_spec_public,
            total_capacity_gpu_hours,
            total_requested_gpu_hours,
            job_count,
            total_capacity_gpu_hours - total_requested_gpu_hours  AS frag_gpu_hours,
            CASE WHEN total_capacity_gpu_hours > 0
                 THEN 1.0 - (total_requested_gpu_hours / total_capacity_gpu_hours)
                 ELSE NULL
            END                                                   AS frag_rate
        FROM read_parquet('{_s(eff_src)}')
        WHERE total_capacity_gpu_hours > 0
        ORDER BY frag_rate DESC
    """).df()


# ── Aggregations ──────────────────────────────────────────────────────────────

_FRAG_PCTILE = """
    round(avg(frag_rate),                        4) AS mean_frag_rate,
    round(quantile_cont(frag_rate, 0.50),        4) AS p50_frag_rate,
    round(quantile_cont(frag_rate, 0.95),        4) AS p95_frag_rate,
    round(quantile_cont(frag_rate, 0.99),        4) AS p99_frag_rate,
    round(min(frag_rate),                        4) AS min_frag_rate,
    round(max(frag_rate),                        4) AS max_frag_rate,
    count(*)                                        AS n_servers
"""


def aggregate_by_server_tier(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
) -> pd.DataFrame:
    """Frag stats grouped by server tier (small / medium / large)."""
    c = con or _con()
    return c.execute(f"""
        WITH base AS (
            SELECT server_tier,
                   CASE WHEN total_capacity_gpu_hours > 0
                        THEN 1.0 - total_requested_gpu_hours / total_capacity_gpu_hours
                        ELSE NULL END AS frag_rate
            FROM read_parquet('{_s(eff_src)}')
            WHERE total_capacity_gpu_hours > 0
        )
        SELECT server_tier, {_FRAG_PCTILE}
        FROM base
        WHERE frag_rate IS NOT NULL
        GROUP BY server_tier
        ORDER BY server_tier
    """).df()


def aggregate_by_gpu_spec(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
) -> pd.DataFrame:
    """Frag stats grouped by GPU spec (A100, H100, L20, …)."""
    c = con or _con()
    return c.execute(f"""
        WITH base AS (
            SELECT gpu_spec_public,
                   CASE WHEN total_capacity_gpu_hours > 0
                        THEN 1.0 - total_requested_gpu_hours / total_capacity_gpu_hours
                        ELSE NULL END AS frag_rate
            FROM read_parquet('{_s(eff_src)}')
            WHERE total_capacity_gpu_hours > 0
        )
        SELECT gpu_spec_public, {_FRAG_PCTILE}
        FROM base
        WHERE frag_rate IS NOT NULL
        GROUP BY gpu_spec_public
        ORDER BY mean_frag_rate DESC
    """).df()


def aggregate_by_priority(
    con: duckdb.DuckDBPyConnection | None = None,
    jts_src: Path = _JTS_SRC,
) -> pd.DataFrame:
    """Job GPU-hours and job counts grouped by priority class (HP / LP / …).

    NOTE: This is job-allocation distribution, not per-server frag_rate,
    because job_summary does not carry server-level utilization.
    """
    c = con or _con()
    return c.execute(f"""
        SELECT
            priority_class,
            sum(job_count)                  AS total_jobs,
            sum(total_gpu_hours_requested)  AS total_gpu_hours,
            round(avg(mean_gpu_request), 4) AS mean_gpu_request,
            round(avg(mean_duration_hours), 2) AS mean_duration_hours,
            round(sum(total_gpu_hours_requested) * 100.0
                  / sum(sum(total_gpu_hours_requested)) OVER (), 2) AS pct_gpu_hours
        FROM read_parquet('{_s(jts_src)}')
        GROUP BY priority_class
        ORDER BY total_gpu_hours DESC
    """).df()


def aggregate_by_workload_bucket(
    con: duckdb.DuckDBPyConnection | None = None,
    jts_src: Path = _JTS_SRC,
) -> pd.DataFrame:
    """Job GPU-hours grouped by job type (online_inference, offline_inference, …)."""
    c = con or _con()
    return c.execute(f"""
        SELECT
            job_type_public,
            sum(job_count)                     AS total_jobs,
            sum(total_gpu_hours_requested)     AS total_gpu_hours,
            round(avg(mean_gpu_request), 4)    AS mean_gpu_request,
            round(avg(p50_gpu_request), 4)     AS p50_gpu_request,
            round(avg(p95_gpu_request), 4)     AS p95_gpu_request,
            round(avg(mean_duration_hours), 2) AS mean_duration_hours,
            round(sum(total_gpu_hours_requested) * 100.0
                  / sum(sum(total_gpu_hours_requested)) OVER (), 2) AS pct_gpu_hours
        FROM read_parquet('{_s(jts_src)}')
        GROUP BY job_type_public
        ORDER BY total_gpu_hours DESC
    """).df()


def aggregate_by_hour_of_day(
    con: duckdb.DuckDBPyConnection | None = None,
    flc_src: Path = _FLC_SRC,
) -> pd.DataFrame:
    """24-row diurnal profile of fleet GPU availability.

    Returns: hour_int, mean/p50/p95 total_gpus_available, mean active_servers.
    """
    c = con or _con()
    return c.execute(f"""
        SELECT
            hour_int,
            round(avg(total_gpus_available), 0)                    AS mean_gpus,
            round(quantile_cont(total_gpus_available, 0.50), 0)    AS p50_gpus,
            round(quantile_cont(total_gpus_available, 0.95), 0)    AS p95_gpus,
            round(avg(active_servers), 0)                          AS mean_servers
        FROM read_parquet('{_s(flc_src)}')
        GROUP BY hour_int
        ORDER BY hour_int
    """).df()


def aggregate_by_day(
    con: duckdb.DuckDBPyConnection | None = None,
    flc_src: Path = _FLC_SRC,
) -> pd.DataFrame:
    """185-row longitudinal profile of fleet GPU availability."""
    c = con or _con()
    return c.execute(f"""
        SELECT
            day,
            round(avg(total_gpus_available), 0) AS mean_gpus,
            max(total_gpus_available)            AS max_gpus,
            min(total_gpus_available)            AS min_gpus,
            round(avg(active_servers), 0)        AS mean_servers
        FROM read_parquet('{_s(flc_src)}')
        GROUP BY day
        ORDER BY day
    """).df()


def day_hour_heatmap(
    con: duckdb.DuckDBPyConnection | None = None,
    flc_src: Path = _FLC_SRC,
) -> pd.DataFrame:
    """185 × 24 pivot of total_gpus_available.

    Index = day (0–184), columns = hour_int (0–23).
    Values = total GPUs available at that (day, hour) cell.
    """
    c = con or _con()
    df = c.execute(f"""
        SELECT day, hour_int, total_gpus_available
        FROM read_parquet('{_s(flc_src)}')
        ORDER BY day, hour_int
    """).df()
    return df.pivot(index="day", columns="hour_int", values="total_gpus_available")


# ── Fleet-level headline ──────────────────────────────────────────────────────

def fleet_headline(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
) -> dict[str, float]:
    """Single-row fleet totals: capacity, requested, unallocated GPU-hours."""
    c = con or _con()
    row = c.execute(f"""
        SELECT
            sum(total_capacity_gpu_hours)   AS total_capacity,
            sum(total_requested_gpu_hours)  AS total_requested,
            sum(total_capacity_gpu_hours - total_requested_gpu_hours) AS unallocated,
            count(DISTINCT server_id)       AS n_servers
        FROM read_parquet('{_s(eff_src)}')
        WHERE total_capacity_gpu_hours > 0
    """).fetchone()
    cap  = float(row[0] or 0)
    req  = float(row[1] or 0)
    unalloc = float(row[2] or 0)
    return {
        "total_capacity_gpu_hours": cap,
        "total_requested_gpu_hours": req,
        "unallocated_gpu_hours": unalloc,
        "frag_rate": unalloc / cap if cap else 0.0,
        "n_servers": int(row[3] or 0),
    }


# ── Convenience runner ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RQ1 fragmentation analysis.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], default="v2026")
    parser.parse_args(argv)  # validate args; trace unused until v2023 support added

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    con = _con()

    headline = fleet_headline(con)
    print("\n=== Fleet Headline ===")
    for k, v in headline.items():
        print(f"  {k:<35} {v:,.2f}" if isinstance(v, float) else f"  {k:<35} {v:,}")

    print("\n=== By Server Tier ===")
    print(aggregate_by_server_tier(con).to_string(index=False))

    print("\n=== By Workload Bucket ===")
    wb = aggregate_by_workload_bucket(con)
    print(wb.to_string(index=False))
    wb.to_csv(TABLE_DIR / "table1_frag_by_workload.csv", index=False)

    print("\n=== By Priority Class ===")
    print(aggregate_by_priority(con).to_string(index=False))


if __name__ == "__main__":
    main()
