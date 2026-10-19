"""Hourly machine-level GPU utilisation aggregates.

Produces one row per (machine_id, hour_bucket) with:
  - mean/p50/p95 GPU utilisation
  - idle_fraction: share of minutes below GPU_IDLE_THRESHOLD
  - allocated_gpus: GPUs assigned to running tasks that hour

Usage:
    python -m src.preprocess.aggregate
"""

import duckdb
from src.config import (
    INTERIM_DIR, PROCESSED_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GPU_IDLE_THRESHOLD,
)


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: read interim/machine_metric_clean.parquet, aggregate to hourly buckets
    raise NotImplementedError("hourly aggregation not yet implemented")


if __name__ == "__main__":
    run()
