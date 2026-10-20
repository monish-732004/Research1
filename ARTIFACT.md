# Phase 5 Artifact Notes

## Start-time estimation

Start times are estimated from `schedule_delay_sec / 3600`.

**Jobs with `schedule_delay_sec = 0`** (N = TBD, Y% of deferrable jobs) have uncertain
start times assigned to `rel_hour = 0` and flagged `start_uncertain = True`.

**Sensitivity:** rerunning with random uniform start assignment for these jobs (sampled
from `[0, TRACE_END_HOUR - dur_h]`) changes aggregate CO₂ reduction results by < Z%.

> N, Y, Z to be filled in after the sweep runs (`python -m src.simulation.sweep`).
> Grep `baseline_*.json` for `"uncertain_starts"` and `"uncertain_pct"` once complete.

---

## Carbon profiles

Synthetic diurnal sinusoidal profiles are used in Phase 5.  Regional grid mix (CN-North,
CN-East, CN-South, CN-West) is parameterised by `GRID_PROFILES` in `src/config.py`.

**Pending:** rerun sweep after real Electricity Maps API data arrives.

---

## Data provenance

| File | Source | Notes |
|---|---|---|
| `job_summary_canonical.parquet` | Alibaba 2022 GPU trace | Deduplicated, non-truncated jobs only |
| `carbon_raw/{region}_{anchor}.parquet` | Synthetic (Phase 5) | Replace with Electricity Maps export |
| `outputs/sim_runs/sweep_results.parquet` | `src.simulation.sweep` | 300 CADS + 12 baseline runs |

---

## Reproducibility

```bash
# 1. Generate carbon profiles (synthetic)
python -m src.preprocess.carbon

# 2. Run 300-config sweep (~20–30 min, 4 workers)
python -m src.simulation.sweep

# 3. Open notebook
jupyter nbconvert --to notebook --execute notebooks/05_simulation_results.ipynb
```

All random seeds are fixed; sweep results are deterministic given the same trace and
carbon profiles.
