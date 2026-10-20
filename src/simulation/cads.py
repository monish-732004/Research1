"""Phase 5 — Carbon-Aware Deferral Scheduler (CADS).

For each deferrable job:
  Scan window [start_hour, start_hour + max_defer_hours]
  Pick the start hour whose mean carbon intensity over duration_hours is lowest.
  Shift that job's start to the best hour.

Non-deferrable jobs (is_deferrable=False) are never shifted.

Parameters
----------
carbon_threshold_gco2_per_kwh : float
    Jobs only considered for deferral when their original start-hour mean CI
    exceeds this threshold.  Set to 0 to always attempt deferral.
max_defer_hours : int  (1–12)
    Hard deadline — a job may not be pushed beyond this many hours.

Outputs
-------
outputs/sim_runs/cads_{region}_{anchor}_{threshold}_{defer}.json
  {region, anchor, carbon_threshold, max_defer_hours,
   total_co2_kg, baseline_co2_kg, reduction_kg, reduction_pct,
   jobs_deferred, jobs_total_eligible,
   mean_defer_hours, p50_defer_hours, p95_defer_hours}

Usage
-----
    python -m src.simulation.cads --region cn-north --anchor 2025-01-01 \
           --threshold 600 --max-defer 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import TRACE_END_HOUR
from src.simulation.baseline import (
    SIM_DIR, _load_carbon, _load_jobs, _job_co2_vectorised,
)


# ── CADS core ─────────────────────────────────────────────────────────────────

def _find_best_starts(
    starts:   np.ndarray,   # (N,) int32 — original start hours
    durs:     np.ndarray,   # (N,) int32 — ceil(duration_hours)
    ci_cs:    np.ndarray,   # (4441,) float64 cumsum
    max_defer: int,
) -> np.ndarray:
    """Vectorised best-start search.

    For each job returns the start hour within [start, start+max_defer]
    that minimises mean carbon intensity over the job's duration.
    Uses cumsum for O(1) range-mean queries.
    """
    n_cands = max_defer + 1
    # candidate starts: (N, n_cands)
    cands = starts[:, None] + np.arange(n_cands, dtype=np.int32)[None, :]
    # clip so candidates don't exceed trace end
    max_allowed = np.maximum(0, TRACE_END_HOUR - durs + 1)    # (N,)
    cands = np.minimum(cands, max_allowed[:, None])

    # ends: (N, n_cands)
    ends  = np.minimum(cands + durs[:, None], len(ci_cs) - 1)

    # mean CI for each candidate: (N, n_cands)
    ci_sum  = ci_cs[ends] - ci_cs[cands]
    widths  = np.maximum(ends - cands, 1)
    mean_ci = ci_sum / widths.astype(np.float64)

    # mask out candidates that would push start past max_allowed
    past_end = cands > max_allowed[:, None]
    mean_ci[past_end] = np.inf

    best_idx = np.argmin(mean_ci, axis=1)
    return cands[np.arange(len(starts)), best_idx]


def run_cads(
    region: str,
    anchor: str,
    carbon_threshold: float,
    max_defer_hours: int,
    jobs: pd.DataFrame | None = None,
    force: bool = False,
) -> dict:
    """Run CADS for one (region, anchor, threshold, max_defer) configuration.

    Returns result dict and writes it to outputs/sim_runs/.
    """
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    tag      = f"{region}_{anchor}_{int(carbon_threshold)}_{max_defer_hours}"
    out_path = SIM_DIR / f"cads_{tag}.json"

    if out_path.exists() and not force:
        with out_path.open() as fh:
            return json.load(fh)

    ci      = _load_carbon(region, anchor)
    ci_cs   = np.concatenate([[0.0], np.cumsum(ci)])

    if jobs is None:
        jobs = _load_jobs()

    # ── Identify eligible jobs ────────────────────────────────────────────────
    eligible_mask = jobs["is_deferrable"].values.astype(bool)
    starts_all    = jobs["start_h"].values.astype(np.int32)
    durs_all      = jobs["dur_h"].values.astype(np.int32)

    # mean CI at original start for each job
    ends_orig = np.minimum(starts_all + durs_all, len(ci_cs) - 1)
    orig_mean_ci = (ci_cs[ends_orig] - ci_cs[starts_all]) / np.maximum(durs_all, 1)

    # eligible = deferrable AND original mean CI > threshold
    above_threshold = orig_mean_ci > carbon_threshold
    defer_mask = eligible_mask & above_threshold   # (N,)

    # ── Find best starts for eligible jobs ────────────────────────────────────
    new_starts = starts_all.copy()
    if defer_mask.any():
        idx = np.where(defer_mask)[0]
        best = _find_best_starts(
            starts_all[idx], durs_all[idx], ci_cs, max_defer_hours
        )
        new_starts[idx] = best

    defer_shift = (new_starts - starts_all).astype(np.float32)
    actually_deferred = defer_mask & (defer_shift > 0)

    # ── Compute total CO₂ ─────────────────────────────────────────────────────
    gpus   = jobs["gpu_request"].values.astype(np.float32)
    tdp_kw = jobs["tdp_kw"].values.astype(np.float32)

    cads_co2     = _job_co2_vectorised(new_starts, durs_all, gpus, tdp_kw, ci_cs)
    baseline_co2 = _job_co2_vectorised(starts_all, durs_all, gpus, tdp_kw, ci_cs)

    total_cads     = float(cads_co2.sum())
    total_baseline = float(baseline_co2.sum())
    reduction_kg   = total_baseline - total_cads
    reduction_pct  = reduction_kg / total_baseline * 100 if total_baseline > 0 else 0.0

    shifts = defer_shift[actually_deferred]
    result = {
        "region":               region,
        "anchor":               anchor,
        "carbon_threshold":     carbon_threshold,
        "max_defer_hours":      max_defer_hours,
        "total_co2_kg":         total_cads,
        "baseline_co2_kg":      total_baseline,
        "reduction_kg":         reduction_kg,
        "reduction_pct":        round(reduction_pct, 4),
        "jobs_deferred":        int(actually_deferred.sum()),
        "jobs_total_eligible":  int(defer_mask.sum()),
        "mean_defer_hours":     float(shifts.mean()) if len(shifts) else 0.0,
        "p50_defer_hours":      float(np.percentile(shifts, 50)) if len(shifts) else 0.0,
        "p95_defer_hours":      float(np.percentile(shifts, 95)) if len(shifts) else 0.0,
    }

    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CADS simulation.")
    parser.add_argument("--region",    required=True)
    parser.add_argument("--anchor",    required=True)
    parser.add_argument("--threshold", type=float, default=600.0)
    parser.add_argument("--max-defer", type=int,   default=8)
    parser.add_argument("--force",     action="store_true")
    args = parser.parse_args(argv)
    result = run_cads(
        args.region, args.anchor,
        args.threshold, args.max_defer,
        force=args.force,
    )
    print(f"  reduction={result['reduction_pct']:.2f}%  "
          f"deferred={result['jobs_deferred']:,}  "
          f"p95={result['p95_defer_hours']:.1f}h")


if __name__ == "__main__":
    main()


# ── Legacy SimPy stub (kept for runner.py compatibility) ──────────────────────

class CADSScheduler:
    """Legacy SimPy-based stub — superseded by run_cads() in Phase 5."""

    def __init__(
        self,
        env,
        n_gpus: int,
        intensity_forecast: dict[int, float],
        predictive_mode: bool = False,
    ) -> None:
        import simpy as _simpy
        self.env = env
        self.n_gpus = n_gpus
        self.gpu_pool = _simpy.Resource(env, capacity=n_gpus)
        self.intensity_forecast = intensity_forecast
        self.predictive_mode = predictive_mode
        self.metrics: list[dict] = []
        self._prod_model = None
        self._jt_mean_gpu_hours: dict[str, float] = {}
        if predictive_mode:
            self._load_production_model()

    def _load_production_model(self) -> None:
        try:
            from src.modeling.choose import load_production_model
            self._prod_model = load_production_model()
        except FileNotFoundError:
            self.predictive_mode = False

    def submit_job(self, *args, **kwargs):
        raise NotImplementedError("Use run_cads() for Phase 5 simulation.")
