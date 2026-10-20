"""GPU wasted-hour and energy calculations (RQ1 — magnitude).

Computes allocation-based waste (GPU capacity never assigned to any job)
from server_allocation_efficiency.parquet and job_type_summary.parquet.

NOTE: This is a lower bound on true waste. Utilization-based waste
(GPUs allocated but underutilised within jobs) requires pod_hourly data.

Functions
---------
total_unallocated_gpu_hours    fleet-level headline waste figure
wasted_kwh                     GPU-hours → kWh (configurable TDP + overhead)
wasted_co2eq                   kWh × grid intensity → gCO₂eq
waste_report                   WasteReport for all TDP scenarios
breakdown_by_priority          waste split by priority class
breakdown_by_workload          waste split by workload type

GPU TDP scenarios
-----------------
  A100 = 400 W,  H100 = 700 W,  mixed = weighted-average from fleet spec mix
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from src.config import (
    PROCESSED_DIR, TABLE_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GRID_INTENSITY_DEFAULTS,
)

# ── GPU TDP scenarios ─────────────────────────────────────────────────────────

A100_TDP_W  = 400.0
H100_TDP_W  = 700.0
MIXED_TDP_W = 550.0   # fleet-weighted default (A100/H100 mix)

TDP_SCENARIOS: dict[str, float] = {
    "A100":  A100_TDP_W,
    "H100":  H100_TDP_W,
    "mixed": MIXED_TDP_W,
}

# PUE × GPU memory overhead (typical hyperscale PUE ≈ 1.15, memory ≈ 10%)
POWER_OVERHEAD = 1.15 * 1.10


# ── Connection helper ─────────────────────────────────────────────────────────

def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


def _s(p: Path) -> str:
    return str(p).replace("\\", "/")


_EFF_SRC = PROCESSED_DIR / "server_allocation_efficiency.parquet"
_JTS_SRC = PROCESSED_DIR / "job_type_summary.parquet"


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class WasteReport:
    tdp_scenario: str
    tdp_watts: float
    unallocated_gpu_hours: float
    wasted_kwh: float
    wasted_co2eq_g: float
    region: str

    def __str__(self) -> str:
        return (
            f"[{self.tdp_scenario:5s} {self.tdp_watts:.0f}W] "
            f"{self.unallocated_gpu_hours:>14,.0f} GPU-h  "
            f"{self.wasted_kwh:>12,.1f} kWh  "
            f"{self.wasted_co2eq_g/1e6:>10,.2f} tCO₂eq  "
            f"({self.region})"
        )


# ── Core calculations ─────────────────────────────────────────────────────────

def total_unallocated_gpu_hours(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
) -> float:
    """Total GPU-hours of fleet capacity never allocated to any job."""
    c = con or _con()
    row = c.execute(f"""
        SELECT sum(total_capacity_gpu_hours - total_requested_gpu_hours)
        FROM read_parquet('{_s(eff_src)}')
        WHERE total_capacity_gpu_hours > 0
    """).fetchone()
    return float(row[0] or 0.0)


def wasted_kwh(
    unallocated_gpu_hours: float,
    tdp_watts: float = MIXED_TDP_W,
    overhead: float = POWER_OVERHEAD,
) -> float:
    """Convert unallocated GPU-hours to kWh.

    kWh = gpu_hours × TDP_W × overhead / 1000
    """
    return unallocated_gpu_hours * tdp_watts * overhead / 1000.0


def wasted_co2eq(
    kwh: float,
    region: str = "cn-beijing",
    intensity_g_per_kwh: float | None = None,
) -> float:
    """Convert kWh to gCO₂eq using grid carbon intensity.

    Returns grams of CO₂-equivalent.
    """
    intensity = intensity_g_per_kwh or GRID_INTENSITY_DEFAULTS.get(region, 580.0)
    return kwh * intensity


def waste_report(
    con: duckdb.DuckDBPyConnection | None = None,
    eff_src: Path = _EFF_SRC,
    region: str = "cn-beijing",
) -> list[WasteReport]:
    """Generate WasteReport for all three TDP scenarios."""
    c = con or _con()
    gpu_hours = total_unallocated_gpu_hours(c, eff_src)
    reports = []
    for name, tdp in TDP_SCENARIOS.items():
        kwh = wasted_kwh(gpu_hours, tdp)
        co2 = wasted_co2eq(kwh, region)
        reports.append(WasteReport(
            tdp_scenario=name,
            tdp_watts=tdp,
            unallocated_gpu_hours=gpu_hours,
            wasted_kwh=kwh,
            wasted_co2eq_g=co2,
            region=region,
        ))
    return reports


# ── Breakdowns ────────────────────────────────────────────────────────────────

def breakdown_by_priority(
    con: duckdb.DuckDBPyConnection | None = None,
    jts_src: Path = _JTS_SRC,
    tdp_watts: float = MIXED_TDP_W,
) -> pd.DataFrame:
    """GPU-hours requested and estimated wasted energy by priority class."""
    c = con or _con()
    df = c.execute(f"""
        SELECT
            priority_class,
            sum(job_count)                  AS total_jobs,
            sum(total_gpu_hours_requested)  AS total_gpu_hours_requested
        FROM read_parquet('{_s(jts_src)}')
        GROUP BY priority_class
        ORDER BY total_gpu_hours_requested DESC
    """).df()
    df["wasted_kwh_est"] = df["total_gpu_hours_requested"] * tdp_watts * POWER_OVERHEAD / 1000.0
    return df


def breakdown_by_workload(
    con: duckdb.DuckDBPyConnection | None = None,
    jts_src: Path = _JTS_SRC,
    tdp_watts: float = MIXED_TDP_W,
) -> pd.DataFrame:
    """GPU-hours requested and estimated wasted energy by workload type."""
    c = con or _con()
    df = c.execute(f"""
        SELECT
            job_type_public,
            sum(job_count)                  AS total_jobs,
            sum(total_gpu_hours_requested)  AS total_gpu_hours_requested,
            round(avg(mean_gpu_request), 4) AS mean_gpu_request
        FROM read_parquet('{_s(jts_src)}')
        GROUP BY job_type_public
        ORDER BY total_gpu_hours_requested DESC
    """).df()
    df["wasted_kwh_est"] = df["total_gpu_hours_requested"] * tdp_watts * POWER_OVERHEAD / 1000.0
    return df


# ── Convenience runner ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GPU waste energy calculation.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], default="v2026")
    parser.parse_args(argv)

    con = _con()
    print("=== Waste Report (allocation-based lower bound) ===")
    for r in waste_report(con):
        print(r)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== Breakdown by Workload ===")
    bw = breakdown_by_workload(con)
    print(bw.to_string(index=False))
    bw.to_csv(TABLE_DIR / "waste_by_workload.csv", index=False)

    print("\n=== Breakdown by Priority ===")
    print(breakdown_by_priority(con).to_string(index=False))


if __name__ == "__main__":
    main()
