"""Phase 4 correlation analysis: allocation gap vs. carbon intensity.

Computes Pearson + Spearman correlations and lag correlations for each
(anchor × region) combination, with block-bootstrap 95% CIs.

⚠️  Uses SYNTHETIC carbon profiles. Correlations will be driven by diurnal
    patterns in both signals. Results should be re-run with real Electricity
    Maps data when available (Phase 4.5).

Output
------
outputs/tables/carbon_correlation.csv
  Columns: anchor, region, lag_hours, pearson_r, spearman_r,
           pearson_ci_lo, pearson_ci_hi, spearman_ci_lo, spearman_ci_hi, p_value

Usage
-----
    python -m src.analysis.correlation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.carbon_fetch import get_carbon_profile
from src.config import (
    CALENDAR_ANCHORS, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    GRID_PROFILES, PROCESSED_DIR, TABLE_DIR,
)

import duckdb

_FLC = PROCESSED_DIR / "hourly_fleet_capacity.parquet"
_EFF = PROCESSED_DIR / "server_allocation_efficiency.parquet"

LAG_HOURS       = [0, 1, 2, 4, 8, 12, -1, -2, -4, -8, -12]
BOOTSTRAP_ITERS = 500
BLOCK_SIZE      = 24   # daily blocks to respect autocorrelation


def _s(p: Path) -> str:
    return str(p).replace("\\", "/")


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")
    return c


# ── Load per-hour allocation gap ──────────────────────────────────────────────

def hourly_allocation_gap(con: duckdb.DuckDBPyConnection | None = None) -> pd.Series:
    """Return fleet allocation gap (idle GPUs) per rel_hour, indexed by rel_hour."""
    c = con or _con()

    # fleet allocation efficiency (period average)
    eff_row = c.execute(f"""
        SELECT sum(total_requested_gpu_hours) / sum(total_capacity_gpu_hours)
        FROM read_parquet('{_s(_EFF)}')
        WHERE total_capacity_gpu_hours > 0
    """).fetchone()
    alloc_eff = float(eff_row[0] or 0.0)
    frag = 1.0 - alloc_eff

    df = c.execute(f"""
        SELECT rel_hour, total_gpus_available
        FROM read_parquet('{_s(_FLC)}')
        ORDER BY rel_hour
    """).df()
    df["alloc_gap_gpus"] = df["total_gpus_available"] * frag
    return df.set_index("rel_hour")["alloc_gap_gpus"]


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def _block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_iter: int = BOOTSTRAP_ITERS,
    block_size: int = BLOCK_SIZE,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Block-bootstrap 95% confidence interval for stat_fn(x, y)."""
    rng = rng or np.random.default_rng(42)
    n = len(x)
    n_blocks = max(1, n // block_size)
    stats_boot = []
    for _ in range(n_iter):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        try:
            stats_boot.append(stat_fn(x[idx], y[idx]))
        except Exception:
            continue
    if not stats_boot:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats_boot, [2.5, 97.5])
    return float(lo), float(hi)


# ── Core correlation for one (anchor, region, lag) ───────────────────────────

def _corr_at_lag(
    gap: np.ndarray,
    intensity: np.ndarray,
    lag: int,
) -> tuple[float, float, float]:
    """Return (pearson_r, spearman_r, p_value) for a given lag.

    Positive lag: carbon intensity leads allocation gap.
    """
    n = len(gap)
    if lag >= 0:
        g = gap[lag:]
        ci = intensity[:n - lag] if lag > 0 else intensity
    else:
        abs_lag = abs(lag)
        g = gap[:n - abs_lag]
        ci = intensity[abs_lag:]

    if len(g) < 10:
        return float("nan"), float("nan"), float("nan")

    pr, pp = stats.pearsonr(g, ci)
    sr, sp = stats.spearmanr(g, ci)
    # Use Spearman p-value as representative (more robust to outliers)
    return float(pr), float(sr), float(sp)


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_correlation_analysis(
    con: duckdb.DuckDBPyConnection | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Compute lag correlations for all 12 (anchor × region) combinations.

    Returns a DataFrame with one row per (anchor, region, lag_hours).
    Also saves to outputs/tables/carbon_correlation.csv.
    """
    out_csv = TABLE_DIR / "carbon_correlation.csv"
    if out_csv.exists() and not force:
        return pd.read_csv(out_csv)

    c = con or _con()
    gap = hourly_allocation_gap(c)
    rng = np.random.default_rng(42)
    rows = []

    for anchor in CALENDAR_ANCHORS:
        for region in GRID_PROFILES:
            cp = get_carbon_profile(region, anchor)
            intensity = cp.set_index("rel_hour")["carbon_intensity_gco2_per_kwh"]
            # Align to the hours present in both series
            common = gap.index.intersection(intensity.index)
            g_arr = gap.loc[common].values.astype(float)
            i_arr = intensity.loc[common].values.astype(float)

            for lag in LAG_HOURS:
                pr, sr, pval = _corr_at_lag(g_arr, i_arr, lag)

                # Bootstrap CI only for lag=0 to save time (±0 is the headline)
                if lag == 0:
                    def pearson_stat(x, y):
                        return stats.pearsonr(x, y)[0]
                    def spearman_stat(x, y):
                        return stats.spearmanr(x, y)[0]
                    p_lo, p_hi = _block_bootstrap_ci(g_arr, i_arr, pearson_stat,
                                                     rng=rng)
                    s_lo, s_hi = _block_bootstrap_ci(g_arr, i_arr, spearman_stat,
                                                     rng=rng)
                else:
                    n = len(g_arr)
                    if lag >= 0:
                        gx, ix = g_arr[lag:], i_arr[:n - lag] if lag > 0 else i_arr
                    else:
                        al = abs(lag)
                        gx, ix = g_arr[:n - al], i_arr[al:]
                    p_lo, p_hi = _block_bootstrap_ci(gx, ix, pearson_stat, rng=rng)
                    s_lo, s_hi = _block_bootstrap_ci(gx, ix, spearman_stat, rng=rng)

                rows.append({
                    "anchor":          anchor,
                    "region":          region,
                    "lag_hours":       lag,
                    "pearson_r":       round(pr, 4) if not np.isnan(pr) else None,
                    "spearman_r":      round(sr, 4) if not np.isnan(sr) else None,
                    "pearson_ci_lo":   round(p_lo, 4) if not np.isnan(p_lo) else None,
                    "pearson_ci_hi":   round(p_hi, 4) if not np.isnan(p_hi) else None,
                    "spearman_ci_lo":  round(s_lo, 4) if not np.isnan(s_lo) else None,
                    "spearman_ci_hi":  round(s_hi, 4) if not np.isnan(s_hi) else None,
                    "p_value":         round(pval, 6) if not np.isnan(pval) else None,
                    "carbon_data_source": "synthetic_diurnal_model",
                })

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    return result


# ── Convenience runner ────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 correlation analysis.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print("Running lag correlation analysis (12 combos × 11 lags) …")
    df = run_correlation_analysis(force=args.force)

    # Print lag=0 headline
    lag0 = df[df["lag_hours"] == 0].copy()
    print("\n=== Lag-0 Spearman correlation (gap vs. carbon intensity) ===")
    print(lag0[["anchor", "region", "spearman_r", "spearman_ci_lo",
                "spearman_ci_hi", "p_value"]].to_string(index=False))

    best = lag0.loc[lag0["spearman_r"].abs().idxmax()]
    print(f"\nBest: {best['region']}  anchor={best['anchor']}  "
          f"ρ={best['spearman_r']:.4f}  p={best['p_value']:.4e}")
    print(f"Saved to {TABLE_DIR / 'carbon_correlation.csv'}")


if __name__ == "__main__":
    main()
