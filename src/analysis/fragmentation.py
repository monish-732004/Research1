"""RQ1 — GPU resource fragmentation analysis.

All heavy aggregation runs in DuckDB against Parquet files; every public
function returns a small pandas DataFrame safe to hold in RAM.

Functions
---------
compute_frag_rate          per (rel_hour, server_tier) frag_rate
aggregate_by_priority      priority class × frag stats
aggregate_by_workload_bucket  workload type × frag stats
aggregate_by_hour_of_day   24-row diurnal profile
aggregate_by_day           185-row longitudinal profile
aggregate_by_server_tier   tier × frag stats
per_pod_frag_rate          per-pod mean frag rate (for CDF figure)

Usage
-----
    python -m src.analysis.fragmentation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.config import (
    INTERIM_DIR, PROCESSED_DIR, TABLE_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
)

# ── Connection helper ─────────────────────────────────────────────────────────

def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


_FRAG_SRC   = PROCESSED_DIR / "hourly_frag.parquet"
_POD_SRC    = INTERIM_DIR   / "pod_hourly_canonical.parquet"

# Percentile helper (DuckDB quantile_cont)
_PCTILE_EXPRS = """
    round(avg(frag_rate), 4)                  AS mean_frag_rate,
    round(quantile_cont(frag_rate, 0.50), 4)  AS p50_frag_rate,
    round(quantile_cont(frag_rate, 0.95), 4)  AS p95_frag_rate,
    round(quantile_cont(frag_rate, 0.99), 4)  AS p99_frag_rate,
    round(min(frag_rate), 4)                  AS min_frag_rate,
    round(max(frag_rate), 4)                  AS max_frag_rate,
    count(*)                                  AS n_hours
"""


# ── Core computation ──────────────────────────────────────────────────────────

def compute_frag_rate(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> pd.DataFrame:
    """Add frag_rate column to hourly_frag data.

    Returns DataFrame: rel_hour, server_tier, total_gpus_requested,
    total_gpus_used, total_frag_gpu_hours, pod_count, frag_rate.
    """
    c = con or _con()
    return c.execute(f"""
        SELECT *,
               CASE WHEN total_gpus_requested > 0
                    THEN total_frag_gpu_hours / total_gpus_requested
                    ELSE NULL
               END AS frag_rate
        FROM read_parquet('{frag_src}')
        ORDER BY rel_hour, server_tier
    """).df()


# ── Pod-level frag (for CDF, priority, workload aggregations) ─────────────────

def per_pod_frag_rate(
    con: duckdb.DuckDBPyConnection | None = None,
    pod_src: Path = _POD_SRC,
) -> pd.DataFrame:
    """Mean fragmentation rate per pod over all its RUNNING hours.

    Returns: pod_uid, gpu_requested_mean, mean_frag_rate, n_running_hours,
             priority, workload_type.
    """
    c = con or _con()
    return c.execute(f"""
        SELECT
            pod_uid,
            avg(gpu_requested)                       AS gpu_requested_mean,
            avg(CASE WHEN gpu_util IS NOT NULL
                     THEN 1.0 - gpu_util ELSE NULL END) AS mean_frag_rate,
            count(*)                                 AS n_running_hours,
            first(priority)                          AS priority,
            first(workload_type)                     AS workload_type
        FROM read_parquet('{pod_src}')
        WHERE state = 'RUNNING'
          AND gpu_requested > 0
        GROUP BY pod_uid
        ORDER BY pod_uid
    """).df()


# ── Aggregations ──────────────────────────────────────────────────────────────

def aggregate_by_priority(
    con: duckdb.DuckDBPyConnection | None = None,
    pod_src: Path = _POD_SRC,
) -> pd.DataFrame:
    """Frag stats grouped by priority class.

    Returns: priority, mean/p50/p95/p99 frag_rate, n_pods.
    """
    c = con or _con()
    return c.execute(f"""
        WITH pod_frag AS (
            SELECT
                pod_uid,
                first(priority)                              AS priority,
                avg(CASE WHEN gpu_util IS NOT NULL
                         THEN 1.0 - gpu_util ELSE NULL END) AS frag_rate
            FROM read_parquet('{pod_src}')
            WHERE state = 'RUNNING' AND gpu_requested > 0
            GROUP BY pod_uid
        )
        SELECT
            priority,
            {_PCTILE_EXPRS.replace('n_hours', 'n_pods')},
            sum(n_pods) OVER () AS total_pods
        FROM (
            SELECT priority, {_PCTILE_EXPRS.replace('count(*)', 'count(pod_uid)')}
            FROM pod_frag
            GROUP BY priority
        )
        ORDER BY priority NULLS LAST
    """).df()


def aggregate_by_workload_bucket(
    con: duckdb.DuckDBPyConnection | None = None,
    pod_src: Path = _POD_SRC,
) -> pd.DataFrame:
    """Frag stats grouped by workload type.

    If workload_type is absent (all NULL), returns a single 'unknown' bucket.
    Returns: workload_type, mean/p50/p95/p99 frag_rate, n_pods, pct_pods.
    """
    c = con or _con()
    return c.execute(f"""
        WITH pod_frag AS (
            SELECT
                pod_uid,
                coalesce(first(workload_type), 'unknown') AS workload_type,
                avg(CASE WHEN gpu_util IS NOT NULL
                         THEN 1.0 - gpu_util ELSE NULL END) AS frag_rate
            FROM read_parquet('{pod_src}')
            WHERE state = 'RUNNING' AND gpu_requested > 0
            GROUP BY pod_uid
        ),
        agg AS (
            SELECT
                workload_type,
                round(avg(frag_rate), 4)                  AS mean_frag_rate,
                round(quantile_cont(frag_rate, 0.50), 4)  AS p50_frag_rate,
                round(quantile_cont(frag_rate, 0.95), 4)  AS p95_frag_rate,
                round(quantile_cont(frag_rate, 0.99), 4)  AS p99_frag_rate,
                round(min(frag_rate), 4)                  AS min_frag_rate,
                round(max(frag_rate), 4)                  AS max_frag_rate,
                count(pod_uid)                            AS n_pods
            FROM pod_frag
            GROUP BY workload_type
        )
        SELECT *,
               round(n_pods * 100.0 / sum(n_pods) OVER (), 2) AS pct_pods
        FROM agg
        ORDER BY n_pods DESC
    """).df()


def aggregate_by_hour_of_day(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> pd.DataFrame:
    """24-row diurnal profile: hour_of_day → mean/p50/p95 frag_rate."""
    c = con or _con()
    return c.execute(f"""
        WITH hourly AS (
            SELECT
                rel_hour % 24 AS hour_of_day,
                CASE WHEN total_gpus_requested > 0
                     THEN total_frag_gpu_hours / total_gpus_requested
                     ELSE NULL END AS frag_rate
            FROM read_parquet('{frag_src}')
        )
        SELECT
            hour_of_day,
            {_PCTILE_EXPRS}
        FROM hourly
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """).df()


def aggregate_by_day(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> pd.DataFrame:
    """185-row longitudinal profile: day → mean/p50/p95 frag_rate.

    Useful for detecting drift, campaign events, and weekend dips.
    """
    c = con or _con()
    return c.execute(f"""
        WITH daily AS (
            SELECT
                rel_hour / 24 AS day,
                CASE WHEN total_gpus_requested > 0
                     THEN total_frag_gpu_hours / total_gpus_requested
                     ELSE NULL END AS frag_rate
            FROM read_parquet('{frag_src}')
        )
        SELECT
            day,
            {_PCTILE_EXPRS}
        FROM daily
        GROUP BY day
        ORDER BY day
    """).df()


def aggregate_by_server_tier(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> pd.DataFrame:
    """Per-tier frag stats over the full trace."""
    c = con or _con()
    return c.execute(f"""
        WITH hourly AS (
            SELECT
                server_tier,
                CASE WHEN total_gpus_requested > 0
                     THEN total_frag_gpu_hours / total_gpus_requested
                     ELSE NULL END AS frag_rate,
                total_frag_gpu_hours,
                total_gpus_requested
            FROM read_parquet('{frag_src}')
        )
        SELECT
            server_tier,
            {_PCTILE_EXPRS},
            sum(total_frag_gpu_hours)  AS total_frag_gpu_hours,
            sum(total_gpus_requested)  AS total_gpus_requested,
            round(sum(total_frag_gpu_hours) * 100.0
                  / nullif(sum(total_gpus_requested), 0), 2) AS overall_frag_pct
        FROM hourly
        GROUP BY server_tier
        ORDER BY overall_frag_pct DESC NULLS LAST
    """).df()


def day_hour_heatmap(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> pd.DataFrame:
    """185 × 24 heatmap table: day, hour_of_day → mean_frag_rate.

    Pivoted into a wide DataFrame (index=day, columns=hour_of_day) by
    the caller (e.g., seaborn heatmap expects this shape).
    """
    c = con or _con()
    long = c.execute(f"""
        SELECT
            rel_hour / 24      AS day,
            rel_hour % 24      AS hour_of_day,
            avg(CASE WHEN total_gpus_requested > 0
                     THEN total_frag_gpu_hours / total_gpus_requested
                     ELSE NULL END) AS mean_frag_rate
        FROM read_parquet('{frag_src}')
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    return long.pivot(index="day", columns="hour_of_day", values="mean_frag_rate")


# ── Runner (save key tables for paper) ───────────────────────────────────────

def run(trace: str = "v2026") -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    con = _con()

    print("[fragmentation] computing frag rate …")
    frag = compute_frag_rate(con)
    frag.to_csv(TABLE_DIR / "frag_rate_hourly.csv", index=False)

    for name, fn in [
        ("by_workload",    lambda: aggregate_by_workload_bucket(con)),
        ("by_priority",    lambda: aggregate_by_priority(con)),
        ("by_hour_of_day", lambda: aggregate_by_hour_of_day(con)),
        ("by_day",         lambda: aggregate_by_day(con)),
        ("by_tier",        lambda: aggregate_by_server_tier(con)),
    ]:
        print(f"  {name} …")
        df = fn()
        df.to_csv(TABLE_DIR / f"frag_{name}.csv", index=False)

    print(f"[fragmentation] done — tables in {TABLE_DIR}")


if __name__ == "__main__":
    run()
