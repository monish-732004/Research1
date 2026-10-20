"""Synthetic hourly carbon intensity profiles for Chinese grid regions.

⚠️  All profiles are SYNTHETIC — generated from a diurnal sinusoidal model
    calibrated to IEA/CEA regional annual averages.  Replace with real
    Electricity Maps historical data when the API key arrives (Phase 4.5).

Usage
-----
    python -m src.analysis.carbon_fetch           # generate + cache all 12 combos
    python -m src.analysis.carbon_fetch --region cn-north --anchor 2025-01-01
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    CARBON_RAW_DIR, CALENDAR_ANCHORS, ELECTRICITY_MAPS_API_KEY,
    GRID_PROFILES, TRACE_DAYS,
)

N_HOURS = TRACE_DAYS * 24   # 4440


# ── Diurnal model ─────────────────────────────────────────────────────────────

def _hourly_multipliers(peak_factor: float) -> list[float]:
    """Return 24 normalised diurnal multipliers (mean = 1.0).

    Night (0–5): 0.85 (grid runs cleaner overnight).
    Day (6–23):  sinusoidal ramp from 1.0 to peak_factor.
    """
    raw = []
    for h in range(24):
        if h < 6:
            raw.append(0.85)
        else:
            v = 1.0 + (peak_factor - 1.0) * math.sin((h - 6) * math.pi / 18)
            raw.append(max(0.85, min(peak_factor, v)))

    mean = sum(raw) / 24
    return [r / mean for r in raw]


# ── Core generator ────────────────────────────────────────────────────────────

def generate_synthetic_carbon_profile(
    region: str,
    anchor_date: date | str,
    n_hours: int = N_HOURS,
) -> pd.DataFrame:
    """Build a synthetic hourly carbon intensity time series.

    Parameters
    ----------
    region      : key in GRID_PROFILES
    anchor_date : date (or 'YYYY-MM-DD') whose midnight UTC = rel_hour 0
    n_hours     : trace length in hours (default 185 × 24 = 4 440)

    Returns
    -------
    DataFrame[rel_hour: int, timestamp: datetime, carbon_intensity_gco2_per_kwh: float]
    with metadata column carbon_data_source = "synthetic_diurnal_model".
    """
    if region not in GRID_PROFILES:
        raise ValueError(f"Unknown region '{region}'. Choose from {list(GRID_PROFILES)}")
    if isinstance(anchor_date, str):
        anchor_date = date.fromisoformat(anchor_date)

    profile = GRID_PROFILES[region]
    annual_avg = profile["annual_avg"]
    peak_factor = profile["peak_factor"]
    mults = _hourly_multipliers(peak_factor)

    base = datetime(anchor_date.year, anchor_date.month, anchor_date.day,
                    tzinfo=timezone.utc)
    rows = []
    for h in range(n_hours):
        ts = base + timedelta(hours=h)
        intensity = annual_avg * mults[ts.hour]
        rows.append({
            "rel_hour": h,
            "timestamp": ts,
            "carbon_intensity_gco2_per_kwh": round(intensity, 2),
            "carbon_data_source": "synthetic_diurnal_model",
        })
    return pd.DataFrame(rows)


# ── Cache wrapper ─────────────────────────────────────────────────────────────

def get_carbon_profile(
    region: str,
    anchor_date: date | str,
    n_hours: int = N_HOURS,
    force: bool = False,
) -> pd.DataFrame:
    """Return a synthetic carbon profile, loading from cache if present.

    Cache location: data/raw/carbon/{region}_{anchor}.parquet.
    """
    anchor_str = anchor_date if isinstance(anchor_date, str) else anchor_date.isoformat()
    cache = CARBON_RAW_DIR / f"{region}_{anchor_str}.parquet"

    if cache.exists() and not force:
        return pd.read_parquet(cache)

    df = generate_synthetic_carbon_profile(region, anchor_str, n_hours)
    CARBON_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ── Electricity Maps stub ─────────────────────────────────────────────────────

def fetch_electricity_maps(
    region: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    """Attempt to fetch real carbon intensity from Electricity Maps API.

    Returns None when no API key is configured.
    Phase 4.5: implement when academic key arrives.
    """
    if not ELECTRICITY_MAPS_API_KEY:
        print("  [carbon_fetch] API key not set — using synthetic profile")
        return None
    print("  [carbon_fetch] Electricity Maps API key present but integration "
          "not yet implemented — using synthetic profile")
    return None


# ── Convenience runner ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate / cache synthetic carbon profiles.")
    parser.add_argument("--region", choices=list(GRID_PROFILES), default=None)
    parser.add_argument("--anchor", choices=list(CALENDAR_ANCHORS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    regions = [args.region] if args.region else list(GRID_PROFILES)
    anchors = [args.anchor] if args.anchor else list(CALENDAR_ANCHORS)

    for region in regions:
        for anchor in anchors:
            df = get_carbon_profile(region, anchor, force=args.force)
            print(
                f"  {region:12s}  {anchor}  "
                f"mean={df['carbon_intensity_gco2_per_kwh'].mean():.1f}  "
                f"min={df['carbon_intensity_gco2_per_kwh'].min():.1f}  "
                f"max={df['carbon_intensity_gco2_per_kwh'].max():.1f}  "
                f"gCO₂/kWh"
            )

    print(f"\nCached to {CARBON_RAW_DIR}")


if __name__ == "__main__":
    main()
