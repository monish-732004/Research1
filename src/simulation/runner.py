"""Simulation driver: loads the trace, runs baseline or CADS, collects metrics.

Usage:
    python -m src.simulation.runner --mode baseline
    python -m src.simulation.runner --mode cads
"""

import argparse
import simpy
import duckdb
from src.config import (
    PROCESSED_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    SIMPY_SEED, TRACE_N_GPUS,
)
from src.simulation.baseline import BaselineScheduler
from src.simulation.cads import CADSScheduler


def run(mode: str = "baseline") -> dict:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    env = simpy.Environment()

    if mode == "baseline":
        sched: BaselineScheduler | CADSScheduler = BaselineScheduler(env, TRACE_N_GPUS)
    elif mode == "cads":
        # TODO: load intensity forecast from processed/carbon_intensity.parquet
        sched = CADSScheduler(env, TRACE_N_GPUS, intensity_forecast={})
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # TODO: load jobs from processed/, feed into scheduler, run env
    raise NotImplementedError(f"simulation runner ({mode}) not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "cads"], default="baseline")
    args = parser.parse_args()
    run(args.mode)
