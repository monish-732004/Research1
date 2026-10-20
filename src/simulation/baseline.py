"""Phase 5 — Baseline CO₂ replay.

For each job in job_summary_canonical (non-truncated):
  start_rel_hour  = clamp(schedule_delay_sec / 3600, 0, TRACE_END_HOUR - ceil(duration_hours))
  energy_kwh(h)   = gpu_request × tdp_kw × POWER_OVERHEAD × 1h
  co2_kg(h)       = energy_kwh(h) × carbon_intensity[h] / 1000
  job_co2         = Σ co2_kg over [start_hour … start_hour + ceil(duration_hours))

Run for all 12 (region × anchor) combinations.

Outputs
-------
outputs/sim_runs/baseline_{region}_{anchor}.json
  {region, anchor, total_co2_kg, total_jobs, total_gpu_hours,
   deferrable_jobs, deferrable_gpu_hours, uncertain_starts,
   uncertain_pct}

Usage
-----
    python -m src.simulation.baseline
    python -m src.simulation.baseline --region cn-north --anchor 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.wasted_hours import POWER_OVERHEAD
from src.config import (
    CARBON_RAW_DIR, CALENDAR_ANCHORS, GRID_PROFILES,
    INTERIM_DIR, PROCESSED_DIR, TABLE_DIR, TRACE_END_HOUR,
)
from src.preprocess.tdp import get_tdp

SIM_DIR = Path("outputs/sim_runs")
_JOB_SRC = INTERIM_DIR / "job_summary_canonical.parquet"

DEFAULT_TDP_W = 350   # fallback watts per GPU slot


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_carbon(region: str, anchor: str) -> np.ndarray:
    """Return float64 array shape (4440,) of gCO₂/kWh indexed by rel_hour."""
    path = CARBON_RAW_DIR / f"{region}_{anchor}.parquet"
    df   = pd.read_parquet(path).sort_values("rel_hour")
    return df["carbon_intensity_gco2_per_kwh"].values.astype(np.float64)


def _load_jobs() -> pd.DataFrame:
    """Load non-truncated jobs with columns needed for CO₂ replay."""
    cols = [
        "pod_id", "gpu_spec_public", "gpu_request", "duration_hours",
        "schedule_delay_sec", "is_deferrable", "job_type_public",
        "left_truncated", "right_truncated",
    ]
    df = pd.read_parquet(_JOB_SRC, columns=cols)
    df["left_truncated"]  = df["left_truncated"].fillna(False).astype(bool)
    df["right_truncated"] = df["right_truncated"].fillna(False).astype(bool)
    df = df[~df["left_truncated"] & ~df["right_truncated"]].copy()
    df["tdp_w"]   = df["gpu_spec_public"].map(
        lambda s: get_tdp(s) if isinstance(s, str) else DEFAULT_TDP_W
    ).astype(np.float32)
    df["tdp_kw"]  = df["tdp_w"] / 1000.0
    df["dur_h"]   = np.ceil(df["duration_hours"].fillna(1.0)).astype(np.int32).clip(1)
    df["start_h"] = (df["schedule_delay_sec"].fillna(0.0) / 3600.0).clip(lower=0)
    df["start_h"] = np.floor(df["start_h"]).astype(np.int32)
    # clamp so job fits inside trace window
    df["start_h"] = np.minimum(df["start_h"], TRACE_END_HOUR - df["dur_h"] + 1).clip(lower=0)
    df["start_uncertain"] = df["schedule_delay_sec"] == 0
    return df.reset_index(drop=True)


def _job_co2_vectorised(
    starts: np.ndarray,   # (N,) int32
    durs:   np.ndarray,   # (N,) int32
    gpus:   np.ndarray,   # (N,) float32
    tdp_kw: np.ndarray,   # (N,) float32
    ci_cumsum: np.ndarray,# (4441,) float64  — prepended 0
) -> np.ndarray:
    """Return per-job CO₂ in kg using cumsum for O(1) range sum queries."""
    ends   = np.minimum(starts + durs, len(ci_cumsum) - 1)
    ci_sum = ci_cumsum[ends] - ci_cumsum[starts]     # gCO₂/kWh × hours
    # energy_kwh = gpus × tdp_kw × POWER_OVERHEAD × hours (already in ci_sum unit)
    # co2_kg     = gpus × tdp_kw × POWER_OVERHEAD × ci_sum / 1000
    return gpus * tdp_kw * POWER_OVERHEAD * ci_sum / 1000.0


# ── main simulation function ──────────────────────────────────────────────────

def run_baseline(
    region: str,
    anchor: str,
    jobs: pd.DataFrame | None = None,
    force: bool = False,
) -> dict:
    """Run baseline CO₂ replay for one (region, anchor) combination.

    Returns the result dict and writes it to outputs/sim_runs/.
    """
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SIM_DIR / f"baseline_{region}_{anchor}.json"

    if out_path.exists() and not force:
        with out_path.open() as fh:
            return json.load(fh)

    ci      = _load_carbon(region, anchor)
    ci_cs   = np.concatenate([[0.0], np.cumsum(ci)])   # (4441,)

    if jobs is None:
        print(f"  [baseline] Loading jobs …")
        jobs = _load_jobs()

    print(f"  [baseline] {region} × {anchor}  ({len(jobs):,} jobs) …")

    co2 = _job_co2_vectorised(
        starts   = jobs["start_h"].values,
        durs     = jobs["dur_h"].values,
        gpus     = jobs["gpu_request"].values.astype(np.float32),
        tdp_kw   = jobs["tdp_kw"].values,
        ci_cumsum= ci_cs,
    )

    n_uncertain   = int(jobs["start_uncertain"].sum())
    n_deferrable  = int(jobs["is_deferrable"].sum())
    def_gpu_hours = float((jobs.loc[jobs["is_deferrable"], "gpu_request"] *
                           jobs.loc[jobs["is_deferrable"], "duration_hours"]).sum())

    result = {
        "region":                 region,
        "anchor":                 anchor,
        "total_co2_kg":           float(co2.sum()),
        "total_jobs":             len(jobs),
        "total_gpu_hours":        float((jobs["gpu_request"] * jobs["duration_hours"]).sum()),
        "deferrable_jobs":        n_deferrable,
        "deferrable_gpu_hours":   def_gpu_hours,
        "uncertain_starts":       n_uncertain,
        "uncertain_pct":          round(n_uncertain / len(jobs) * 100, 2),
    }

    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  [baseline] CO₂={result['total_co2_kg']/1e6:.3f} Mt  → {out_path}")
    return result


def run_all_baselines(force: bool = False) -> list[dict]:
    jobs = _load_jobs()
    results = []
    for region in GRID_PROFILES:
        for anchor in CALENDAR_ANCHORS:
            results.append(run_baseline(region, anchor, jobs=jobs, force=force))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run baseline CO₂ replay.")
    parser.add_argument("--region", default=None)
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args(argv)

    if args.region and args.anchor:
        run_baseline(args.region, args.anchor, force=args.force)
    else:
        run_all_baselines(force=args.force)


if __name__ == "__main__":
    main()
