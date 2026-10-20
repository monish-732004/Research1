"""Phase 4 carbon join: compute hourly idle CO₂ cost and deferrable savings.

⚠️  Uses SYNTHETIC carbon intensity profiles (see carbon_fetch.py).
    All numbers are illustrative until real Electricity Maps data arrives.

Approach
--------
Per-hour requested GPUs are unavailable without pod_hourly data.
We approximate: estimated_requested(h) = total_gpus_available(h) × fleet_alloc_efficiency
where fleet_alloc_efficiency = sum(requested) / sum(capacity) over all servers.
This preserves the 57% period-average allocation gap and distributes it uniformly
across all hours in proportion to fleet size — a conservative lower bound.

Outputs (per region × anchor combo, 12 total)
----------------------------------------------
data/processed/hourly_carbon_{region}_{anchor}.parquet
data/processed/deferrable_carbon_savings.parquet  (anchor-independent)

Usage
-----
    python -m src.analysis.carbon_join
    python -m src.analysis.carbon_join --region cn-north --anchor 2025-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.analysis.carbon_fetch import get_carbon_profile
from src.analysis.wasted_hours import MIXED_TDP_W, POWER_OVERHEAD
from src.config import (
    CALENDAR_ANCHORS, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GRID_PROFILES, PROCESSED_DIR,
)

_FLC   = PROCESSED_DIR / "hourly_fleet_capacity.parquet"
_EFF   = PROCESSED_DIR / "server_allocation_efficiency.parquet"
_JTS   = PROCESSED_DIR / "job_type_summary.parquet"

MEAN_FLEET_TDP_KW = MIXED_TDP_W / 1000.0   # 0.55 kW per GPU slot


def _s(p: Path) -> str:
    return str(p).replace("\\", "/")


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


# ── Fleet allocation efficiency (period average) ──────────────────────────────

def fleet_alloc_efficiency(con: duckdb.DuckDBPyConnection | None = None) -> float:
    """Fleet-wide average: total_requested / total_capacity over 185 days."""
    c = con or _con()
    row = c.execute(f"""
        SELECT
            sum(total_requested_gpu_hours) / sum(total_capacity_gpu_hours)
        FROM read_parquet('{_s(_EFF)}')
        WHERE total_capacity_gpu_hours > 0
    """).fetchone()
    return float(row[0] or 0.0)


# ── Fleet-level hourly carbon cost ────────────────────────────────────────────

def compute_hourly_carbon(
    region: str,
    anchor: str,
    alloc_eff: float | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Join fleet capacity with a synthetic carbon profile to compute idle CO₂.

    For each rel_hour:
      idle_gpus      = total_gpus_available × (1 − alloc_efficiency)
      idle_kwh       = idle_gpus × MEAN_FLEET_TDP_KW × POWER_OVERHEAD
      idle_co2_kg    = idle_kwh × carbon_intensity_gco2_per_kwh / 1_000

    Parameters
    ----------
    region, anchor : grid region and calendar anchor (see GRID_PROFILES / CALENDAR_ANCHORS)
    alloc_eff      : precomputed fleet allocation efficiency (0–1); computed if None
    force          : recompute even if cached output exists

    Returns
    -------
    DataFrame[rel_hour, day, hour_int, total_gpus_available, active_servers,
              timestamp, carbon_intensity_gco2_per_kwh, idle_gpus, idle_kwh,
              idle_co2_kg, carbon_data_source]
    """
    out = PROCESSED_DIR / f"hourly_carbon_{region}_{anchor}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    c = con or _con()
    eff = alloc_eff if alloc_eff is not None else fleet_alloc_efficiency(c)
    frag = 1.0 - eff

    carbon_df = get_carbon_profile(region, anchor)

    fleet_df = c.execute(f"""
        SELECT rel_hour, day, hour_int, total_gpus_available, active_servers
        FROM read_parquet('{_s(_FLC)}')
        ORDER BY rel_hour
    """).df()

    df = fleet_df.merge(carbon_df[["rel_hour", "timestamp",
                                    "carbon_intensity_gco2_per_kwh",
                                    "carbon_data_source"]],
                        on="rel_hour", how="inner")

    df["idle_gpus"]    = df["total_gpus_available"] * frag
    df["idle_kwh"]     = df["idle_gpus"] * MEAN_FLEET_TDP_KW * POWER_OVERHEAD
    df["idle_co2_kg"]  = df["idle_kwh"] * df["carbon_intensity_gco2_per_kwh"] / 1_000.0
    df["region"]       = region
    df["anchor"]       = anchor

    df.to_parquet(out, index=False)
    return df


# ── Deferrable job carbon savings ─────────────────────────────────────────────

def compute_deferrable_savings(
    anchor: str = "2025-01-01",
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Upper-bound CO₂ saving from shifting deferrable jobs to low-carbon windows.

    Method (per region):
      1. Load carbon profile → compute 12-hour rolling minimum intensity.
      2. For each job_type × priority group (is_deferrable=True):
           mean_intensity     = mean carbon over the full 4 440-hour trace
           mean_min12h        = mean of the 12h rolling minimum at each hour
           carbon_saving_pct  = (mean_intensity − mean_min12h) / mean_intensity
           co2_saving_kg      = total_gpu_hours × TDP_kw × overhead
                                × mean_intensity × carbon_saving_pct / 1_000
    This is a theoretical ceiling; real savings depend on scheduler flexibility.

    The deferrable_carbon_savings.parquet is written once (anchor-independent
    for the savings %, though absolute kg depend on carbon intensity level).
    """
    out = PROCESSED_DIR / "deferrable_carbon_savings.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    c = con or _con()

    defer_df = c.execute(f"""
        SELECT
            job_type_public,
            priority_class,
            gpu_spec_public,
            sum(job_count)                    AS total_jobs,
            sum(total_gpu_hours_requested)    AS total_gpu_hours_requested,
            sum(total_gpu_hours_requested) FILTER (WHERE is_deferrable)
                                              AS deferrable_gpu_hours
        FROM read_parquet('{_s(_JTS)}')
        GROUP BY job_type_public, priority_class, gpu_spec_public
        HAVING deferrable_gpu_hours > 0
        ORDER BY deferrable_gpu_hours DESC
    """).df()

    rows = []
    for region in GRID_PROFILES:
        cp = get_carbon_profile(region, anchor)
        intensity = cp["carbon_intensity_gco2_per_kwh"].values

        # 12h rolling minimum (forward-looking)
        min12h = pd.Series(intensity).rolling(12, min_periods=1).min().values
        mean_int  = intensity.mean()
        mean_min  = min12h.mean()
        saving_pct = (mean_int - mean_min) / mean_int if mean_int > 0 else 0.0

        for _, row in defer_df.iterrows():
            dgh = float(row["deferrable_gpu_hours"])
            if dgh <= 0:
                continue
            co2_baseline_kg = dgh * MEAN_FLEET_TDP_KW * POWER_OVERHEAD * mean_int / 1_000.0
            co2_saving_kg   = co2_baseline_kg * saving_pct
            rows.append({
                "region":                       region,
                "job_type_public":              row["job_type_public"],
                "priority_class":               row["priority_class"],
                "gpu_spec_public":              row["gpu_spec_public"],
                "total_jobs":                   int(row["total_jobs"]),
                "total_gpu_hours_requested":    float(row["total_gpu_hours_requested"]),
                "deferrable_gpu_hours":         dgh,
                "mean_carbon_intensity":        round(mean_int, 2),
                "mean_min12h_intensity":        round(mean_min, 2),
                "carbon_saving_pct":            round(saving_pct * 100, 2),
                "co2_baseline_kg":              round(co2_baseline_kg, 1),
                "co2_saving_kg_upper_bound":    round(co2_saving_kg, 1),
                "carbon_data_source":           "synthetic_diurnal_model",
                "anchor":                       anchor,
            })

    result = pd.DataFrame(rows)
    result.to_parquet(out, index=False)
    return result


# ── Summary helpers ───────────────────────────────────────────────────────────

def headline_table(
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """3 × 4 grid of total idle CO₂ (tonnes) by (anchor × region)."""
    rows = []
    alloc_eff = fleet_alloc_efficiency(con)
    for anchor in CALENDAR_ANCHORS:
        for region in GRID_PROFILES:
            df = compute_hourly_carbon(region, anchor, alloc_eff, con, force=force)
            total_t = df["idle_co2_kg"].sum() / 1_000.0
            rows.append({"anchor": anchor, "region": region,
                         "idle_co2_tonnes": round(total_t, 1)})
    return pd.DataFrame(rows).pivot(index="anchor", columns="region",
                                    values="idle_co2_tonnes")


# ── Convenience runner ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 carbon join.")
    parser.add_argument("--region", choices=list(GRID_PROFILES), default=None)
    parser.add_argument("--anchor", choices=list(CALENDAR_ANCHORS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    regions = [args.region] if args.region else list(GRID_PROFILES)
    anchors = [args.anchor] if args.anchor else list(CALENDAR_ANCHORS)

    con = _con()
    eff = fleet_alloc_efficiency(con)
    print(f"Fleet allocation efficiency: {eff:.4f}  (frag rate: {1-eff:.4f})")

    print("\n=== Hourly carbon profiles ===")
    for anchor in anchors:
        for region in regions:
            df = compute_hourly_carbon(region, anchor, eff, con, force=args.force)
            total = df["idle_co2_kg"].sum() / 1_000.0
            print(f"  {region:12s}  {anchor}  total idle CO₂: {total:,.1f} tonnes")

    print("\n=== Deferrable savings (cn-north, 2025-01-01 anchor) ===")
    sav = compute_deferrable_savings(force=args.force)
    by_region = (sav.groupby("region")[["co2_saving_kg_upper_bound", "co2_baseline_kg"]]
                    .sum().assign(
                        saving_pct=lambda d: d["co2_saving_kg_upper_bound"] / d["co2_baseline_kg"] * 100
                    ).round(1))
    print(by_region.to_string())

    print("\n=== 3×4 headline table (idle CO₂ tonnes) ===")
    print(headline_table(con).to_string())


if __name__ == "__main__":
    main()
