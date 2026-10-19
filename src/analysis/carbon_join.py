"""Join fragmentation aggregates with grid carbon intensity data.

For each (region, hour_bucket) row in the fragmentation table, look up
the grid carbon intensity (gCO₂eq/kWh) from the Electricity Maps API
(or from the offline fallback constants in config.py).

Carbon waste (gCO₂eq) per hour =
    wasted_gpu_hours * GPU_TDP_WATTS/1000 * carbon_intensity

Usage:
    python -m src.analysis.carbon_join
"""

import os
import requests
import duckdb
from src.config import (
    PROCESSED_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    ELECTRICITY_MAPS_API_KEY, GRID_INTENSITY_DEFAULTS, ALIBABA_REGIONS,
)

# Approximate GPU TDP for the mix of cards in the Alibaba cluster (Watts).
# Refine once machine_spec data is loaded.
GPU_TDP_WATTS = 300.0


def fetch_intensity(region: str, iso_zone: str, timestamp: int) -> float:
    """Return gCO₂eq/kWh for the given zone and Unix timestamp."""
    if ELECTRICITY_MAPS_API_KEY is None:
        return GRID_INTENSITY_DEFAULTS.get(region, 550.0)

    # TODO: call Electricity Maps historical API
    raise NotImplementedError("live Electricity Maps fetch not yet implemented")


def run(con: duckdb.DuckDBPyConnection | None = None) -> None:
    if con is None:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={DUCKDB_THREADS}")

    # TODO: join fragmentation table with carbon intensity; write carbon_waste.parquet
    raise NotImplementedError("carbon_join not yet implemented")


if __name__ == "__main__":
    run()
