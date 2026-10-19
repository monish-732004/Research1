"""GPU fragmentation metric computation.

Fragmentation is defined as the gap between allocated GPU capacity and
measured GPU utilisation.  Two complementary metrics are computed:

  frag_alloc  = 1 - (mean_gpu_util / allocated_gpu_fraction)
  frag_idle   = fraction of GPU-minutes below GPU_IDLE_THRESHOLD

Usage:
    python -m src.analysis.fragmentation
"""

import duckdb
from src.config import (
    PROCESSED_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GPU_IDLE_THRESHOLD, FRAG_WINDOW_HOURS,
)


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: read processed/hourly_machine.parquet, compute frag_alloc + frag_idle
    raise NotImplementedError("fragmentation metrics not yet implemented")


if __name__ == "__main__":
    run()
