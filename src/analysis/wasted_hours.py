"""GPU wasted-hour and energy calculations (RQ1 — magnitude).

Functions
---------
total_wasted_gpu_hours      headline waste figure
wasted_kwh                  convert GPU-hours to kWh given TDP
wasted_co2eq                kWh × grid intensity → gCO₂eq
breakdown_by_priority       waste split by priority class
breakdown_by_workload       waste split by workload type

GPU TDP scenarios
-----------------
A100_TDP  = 400 W  (SXM4 TDP)
H100_TDP  = 700 W  (SXM5 TDP)
MIXED_TDP = 550 W  (assumed 50/50 fleet)

Usage
-----
    python -m src.analysis.wasted_hours
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import duckdb
import pandas as pd

from src.config import (
    INTERIM_DIR, PROCESSED_DIR, TABLE_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GRID_INTENSITY_DEFAULTS,
)

# ── GPU TDP scenarios ─────────────────────────────────────────────────────────

A100_TDP_W  = 400.0
H100_TDP_W  = 700.0
MIXED_TDP_W = 550.0   # fleet-weighted default

TDP_SCENARIOS: dict[str, float] = {
    "A100":  A100_TDP_W,
    "H100":  H100_TDP_W,
    "mixed": MIXED_TDP_W,
}

# Overhead multiplier: PUE × memory power fraction
# Typical hyperscale PUE ≈ 1.15; GPU memory draws ≈ 10% extra
POWER_OVERHEAD = 1.15 * 1.10


# ── Connection helper ─────────────────────────────────────────────────────────

def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


_FRAG_SRC = PROCESSED_DIR / "hourly_frag.parquet"
_POD_SRC  = INTERIM_DIR   / "pod_hourly_canonical.parquet"


# ── Core calculations ─────────────────────────────────────────────────────────

def total_wasted_gpu_hours(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
) -> float:
    """Return total fragmented GPU-hours over the full trace."""
    c = con or _con()
    row = c.execute(
        f"SELECT sum(total_frag_gpu_hours) FROM read_parquet('{frag_src}')"
    ).fetchone()
    return float(row[0] or 0.0)


def wasted_kwh(
    gpu_hours: float,
    *,
    tdp_watts: float = MIXED_TDP_W,
    include_overhead: bool = True,
) -> float:
    """Convert fragmented GPU-hours to wasted kWh.

    Parameters
    ----------
    gpu_hours:
        Total wasted GPU-hours.
    tdp_watts:
        GPU thermal design power in watts.
    include_overhead:
        If True, multiply by POWER_OVERHEAD (PUE + memory).
    """
    multiplier = POWER_OVERHEAD if include_overhead else 1.0
    return gpu_hours * (tdp_watts / 1000.0) * multiplier


def wasted_co2eq(
    kwh: float,
    region: str = "cn-beijing",
    *,
    intensity_g_per_kwh: float | None = None,
) -> float:
    """Convert kWh to gCO₂eq using grid carbon intensity.

    Parameters
    ----------
    kwh:
        Energy in kWh.
    region:
        Alibaba region key (looked up in GRID_INTENSITY_DEFAULTS).
    intensity_g_per_kwh:
        Override the default intensity (gCO₂eq/kWh).
    """
    if intensity_g_per_kwh is None:
        intensity_g_per_kwh = GRID_INTENSITY_DEFAULTS.get(region, 550.0)
    return kwh * intensity_g_per_kwh


@dataclass
class WasteReport:
    """Summary of wasted GPU energy and carbon for one scenario."""
    scenario: str
    tdp_watts: float
    wasted_gpu_hours: float
    wasted_kwh: float
    wasted_kwh_with_overhead: float
    wasted_co2eq_kg: float          # kg CO₂eq (÷1000 from g)
    wasted_co2eq_tonnes: float

    def __str__(self) -> str:
        return (
            f"[{self.scenario}] "
            f"{self.wasted_gpu_hours:,.0f} GPU-hours  "
            f"{self.wasted_kwh_with_overhead:,.1f} kWh  "
            f"{self.wasted_co2eq_tonnes:,.2f} tCO₂eq"
        )


def waste_report(
    con: duckdb.DuckDBPyConnection | None = None,
    frag_src: Path = _FRAG_SRC,
    region: str = "cn-beijing",
) -> list[WasteReport]:
    """Generate WasteReport for all TDP scenarios."""
    c = con or _con()
    gpu_hours = total_wasted_gpu_hours(c, frag_src)
    reports = []
    for name, tdp in TDP_SCENARIOS.items():
        kwh_raw  = wasted_kwh(gpu_hours, tdp_watts=tdp, include_overhead=False)
        kwh_over = wasted_kwh(gpu_hours, tdp_watts=tdp, include_overhead=True)
        co2_g    = wasted_co2eq(kwh_over, region=region)
        reports.append(WasteReport(
            scenario=name,
            tdp_watts=tdp,
            wasted_gpu_hours=gpu_hours,
            wasted_kwh=kwh_raw,
            wasted_kwh_with_overhead=kwh_over,
            wasted_co2eq_kg=co2_g / 1000.0,
            wasted_co2eq_tonnes=co2_g / 1_000_000.0,
        ))
    return reports


# ── Breakdowns ────────────────────────────────────────────────────────────────

def breakdown_by_priority(
    con: duckdb.DuckDBPyConnection | None = None,
    pod_src: Path = _POD_SRC,
    tdp_watts: float = MIXED_TDP_W,
) -> pd.DataFrame:
    """Wasted GPU-hours and kWh split by pod priority class."""
    c = con or _con()
    df = c.execute(f"""
        SELECT
            first(priority)                                    AS priority,
            sum(gpu_requested * (1.0 - coalesce(gpu_util, 0))) AS wasted_gpu_hours,
            count(DISTINCT pod_uid)                            AS n_pods
        FROM read_parquet('{pod_src}')
        WHERE state = 'RUNNING' AND gpu_requested > 0
        GROUP BY pod_uid
    """).df()   # pod-level first, then aggregate

    # DuckDB doesn't directly support two-level aggregation in one query here
    result = c.execute(f"""
        WITH pod_waste AS (
            SELECT
                pod_uid,
                first(priority)                                    AS priority,
                sum(gpu_requested * (1.0 - coalesce(gpu_util, 0))) AS wasted_gpu_hours
            FROM read_parquet('{pod_src}')
            WHERE state = 'RUNNING' AND gpu_requested > 0
            GROUP BY pod_uid
        )
        SELECT
            priority,
            sum(wasted_gpu_hours)             AS wasted_gpu_hours,
            count(pod_uid)                    AS n_pods,
            round(sum(wasted_gpu_hours) * {tdp_watts / 1000.0} * {POWER_OVERHEAD}, 1)
                                              AS wasted_kwh
        FROM pod_waste
        GROUP BY priority
        ORDER BY wasted_gpu_hours DESC
    """).df()
    return result


def breakdown_by_workload(
    con: duckdb.DuckDBPyConnection | None = None,
    pod_src: Path = _POD_SRC,
    tdp_watts: float = MIXED_TDP_W,
) -> pd.DataFrame:
    """Wasted GPU-hours and kWh split by workload type."""
    c = con or _con()
    return c.execute(f"""
        WITH pod_waste AS (
            SELECT
                pod_uid,
                coalesce(first(workload_type), 'unknown')          AS workload_type,
                sum(gpu_requested * (1.0 - coalesce(gpu_util, 0))) AS wasted_gpu_hours
            FROM read_parquet('{pod_src}')
            WHERE state = 'RUNNING' AND gpu_requested > 0
            GROUP BY pod_uid
        )
        SELECT
            workload_type,
            sum(wasted_gpu_hours)             AS wasted_gpu_hours,
            count(pod_uid)                    AS n_pods,
            round(sum(wasted_gpu_hours) * {tdp_watts / 1000.0} * {POWER_OVERHEAD}, 1)
                                              AS wasted_kwh,
            round(sum(wasted_gpu_hours) * 100.0
                  / sum(sum(wasted_gpu_hours)) OVER (), 2) AS pct_of_total
        FROM pod_waste
        GROUP BY workload_type
        ORDER BY wasted_gpu_hours DESC
    """).df()


# ── Runner ────────────────────────────────────────────────────────────────────

def run(trace: str = "v2026") -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    con = _con()

    print("[wasted_hours] computing waste reports …")
    for r in waste_report(con):
        print(f"  {r}")

    print("[wasted_hours] breakdown by priority …")
    breakdown_by_priority(con).to_csv(TABLE_DIR / "waste_by_priority.csv", index=False)

    print("[wasted_hours] breakdown by workload …")
    breakdown_by_workload(con).to_csv(TABLE_DIR / "waste_by_workload.csv", index=False)

    print(f"[wasted_hours] done — tables in {TABLE_DIR}")


if __name__ == "__main__":
    run()
