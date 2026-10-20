"""Phase 5 — CADS sensitivity sweep: 300 CADS + 12 baseline runs.

Grid
----
carbon_threshold ∈ {400, 500, 600, 700, 800} gCO₂/kWh
max_defer_hours  ∈ {1, 2, 4, 8, 12}
region           ∈ 4 Chinese grid regions
anchor           ∈ 3 calendar anchors
= 5 × 5 × 4 × 3 = 300 CADS runs  +  4 × 3 = 12 baseline runs

Parallelises with multiprocessing.Pool(4).

Outputs
-------
outputs/sim_runs/sweep_results.parquet
  columns: [region, anchor, threshold, max_defer,
            reduction_pct, p95_defer_hours, jobs_deferred_pct]

Usage
-----
    python -m src.simulation.sweep
    python -m src.simulation.sweep --dry-run
    python -m src.simulation.sweep --workers 4
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

from src.config import CALENDAR_ANCHORS, GRID_PROFILES
from src.simulation.baseline import SIM_DIR, run_all_baselines, _load_jobs
from src.simulation.cads import run_cads

THRESHOLDS   = [400, 500, 600, 700, 800]
MAX_DEFERS   = [1, 2, 4, 8, 12]
SWEEP_OUT    = SIM_DIR / "sweep_results.parquet"


# ── Worker (must be top-level for multiprocessing pickle) ─────────────────────

def _cads_worker(args: tuple) -> dict:
    region, anchor, threshold, max_defer = args
    try:
        return run_cads(region, anchor, threshold, max_defer)
    except Exception as exc:
        return {
            "region": region, "anchor": anchor,
            "carbon_threshold": threshold, "max_defer_hours": max_defer,
            "error": str(exc),
        }


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(workers: int = 4, force: bool = False) -> pd.DataFrame:
    """Run all 300 CADS + 12 baseline configurations.

    Returns a tidy DataFrame and writes sweep_results.parquet.
    """
    SIM_DIR.mkdir(parents=True, exist_ok=True)

    if SWEEP_OUT.exists() and not force:
        print(f"  [sweep] Loading cached results from {SWEEP_OUT}")
        return pd.read_parquet(SWEEP_OUT)

    # ── 1. Baselines ──────────────────────────────────────────────────────────
    print("  [sweep] Running 12 baseline simulations …")
    baseline_results = run_all_baselines(force=force)
    print(f"  [sweep] Baselines done: {len(baseline_results)} runs")

    # ── 2. CADS grid ──────────────────────────────────────────────────────────
    regions = list(GRID_PROFILES.keys())
    anchors = list(CALENDAR_ANCHORS.keys())
    configs = list(itertools.product(regions, anchors, THRESHOLDS, MAX_DEFERS))
    print(f"  [sweep] Running {len(configs)} CADS configurations "
          f"with {workers} workers …")

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        cads_results = pool.map(_cads_worker, configs)
    elapsed = time.perf_counter() - t0
    print(f"  [sweep] CADS done in {elapsed:.0f}s  "
          f"({len(cads_results)} results)")

    # ── 3. Build tidy DataFrame ───────────────────────────────────────────────
    rows = []

    # Baseline rows (threshold=0, max_defer=0 signals baseline)
    bl_map = {(r["region"], r["anchor"]): r for r in baseline_results}
    for r in baseline_results:
        rows.append({
            "region":            r["region"],
            "anchor":            r["anchor"],
            "threshold":         0.0,
            "max_defer":         0,
            "run_type":          "baseline",
            "total_co2_kg":      r["total_co2_kg"],
            "baseline_co2_kg":   r["total_co2_kg"],
            "reduction_pct":     0.0,
            "p95_defer_hours":   0.0,
            "jobs_deferred":     0,
            "jobs_total_eligible": r["deferrable_jobs"],
            "jobs_deferred_pct": 0.0,
        })

    # CADS rows
    errors = 0
    for r in cads_results:
        if "error" in r:
            errors += 1
            continue
        bl = bl_map.get((r["region"], r["anchor"]), {})
        n_eligible = r.get("jobs_total_eligible", 1) or 1
        rows.append({
            "region":            r["region"],
            "anchor":            r["anchor"],
            "threshold":         r["carbon_threshold"],
            "max_defer":         r["max_defer_hours"],
            "run_type":          "cads",
            "total_co2_kg":      r["total_co2_kg"],
            "baseline_co2_kg":   r["baseline_co2_kg"],
            "reduction_pct":     r["reduction_pct"],
            "p95_defer_hours":   r["p95_defer_hours"],
            "jobs_deferred":     r["jobs_deferred"],
            "jobs_total_eligible": r["jobs_total_eligible"],
            "jobs_deferred_pct": round(r["jobs_deferred"] / n_eligible * 100, 4),
        })

    if errors:
        print(f"  [sweep] WARNING: {errors} CADS runs failed (see individual JSONs)")

    df = pd.DataFrame(rows)
    df.to_parquet(SWEEP_OUT, index=False)
    print(f"  [sweep] Wrote {len(df)} rows → {SWEEP_OUT}")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run CADS sensitivity sweep.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print configs without running")
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args(argv)

    regions = list(GRID_PROFILES.keys())
    anchors = list(CALENDAR_ANCHORS.keys())
    configs = list(itertools.product(regions, anchors, THRESHOLDS, MAX_DEFERS))
    print(f"Sweep: {len(configs)} CADS + {len(regions)*len(anchors)} baseline "
          f"= {len(configs) + len(regions)*len(anchors)} total runs")

    if args.dry_run:
        for i, (reg, anc, thr, md) in enumerate(configs[:10]):
            print(f"  [{i+1:3d}] region={reg}  anchor={anc}  "
                  f"threshold={thr}  max_defer={md}h")
        if len(configs) > 10:
            print(f"  … ({len(configs)-10} more)")
        return

    df = run_sweep(workers=args.workers, force=args.force)
    cads_df = df[df["run_type"] == "cads"]
    best = cads_df.loc[cads_df["reduction_pct"].idxmax()]
    print(f"\n=== Best CADS config ===")
    print(f"  region={best['region']}  anchor={best['anchor']}  "
          f"threshold={best['threshold']}  max_defer={best['max_defer']}h")
    print(f"  reduction={best['reduction_pct']:.2f}%  "
          f"p95_defer={best['p95_defer_hours']:.1f}h")


if __name__ == "__main__":
    main()
