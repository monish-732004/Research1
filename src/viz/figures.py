"""Publication-quality figures for the gpu-carbon-waste paper.

All functions accept a small pandas DataFrame (the aggregated result from
src/analysis/fragmentation.py) and an output Path, produce an IEEE-compliant
PDF, and return the Path.

Figures
-------
  fig_frag_cdf_by_workload    Fig 1 — CDF of per-pod frag rate by workload type
  fig_frag_heatmap_day_hour   Fig 2 — 185×24 heatmap (weekly cycles)
  fig_frag_by_priority        Fig 3 — box plot by priority class
  fig_frag_by_tier            Fig 4 — box plot by server tier
  generate_all                Convenience runner for all four

Usage
-----
    python -m src.viz.figures
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIG_DIR, PROCESSED_DIR, INTERIM_DIR
from src.viz.style import (
    set_ieee_style, figure_single, figure_double,
    PALETTE, WORKLOAD_COLORS, TIER_COLORS,
)

set_ieee_style()

# ── Default data sources ──────────────────────────────────────────────────────

_FRAG_SRC = PROCESSED_DIR / "hourly_frag.parquet"
_POD_SRC  = INTERIM_DIR   / "pod_hourly_canonical.parquet"


# ── Fig 1 — CDF of per-pod fragmentation rate by workload type ────────────────

def fig_frag_cdf_by_workload(
    pod_frag_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """CDF of per-pod mean fragmentation rates, overlaid by workload type.

    Parameters
    ----------
    pod_frag_df:
        Output of ``fragmentation.per_pod_frag_rate()``.
        Required columns: mean_frag_rate, workload_type.
    """
    out_path = out_path or FIG_DIR / "fig1_frag_cdf_by_workload.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = figure_single(height_ratio=1.1)

    df = pod_frag_df.dropna(subset=["mean_frag_rate"])
    workload_order = (
        df.groupby("workload_type")["mean_frag_rate"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for wt in workload_order:
        subset = df[df["workload_type"] == wt]["mean_frag_rate"].sort_values()
        n = len(subset)
        if n == 0:
            continue
        cdf = np.arange(1, n + 1) / n
        label = wt.replace("_", " ").title() if wt else "Unknown"
        color = WORKLOAD_COLORS.get(wt, "#888888")
        ax.plot(subset.values, cdf, label=label, color=color, lw=1.0)

    ax.axvline(0.5, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Per-pod fragmentation rate")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right", handlelength=1.2)
    ax.set_title("Fragmentation rate CDF by workload type")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 2 — Heatmap of mean frag by (day × hour-of-day) ──────────────────────

def fig_frag_heatmap_day_hour(
    pivot_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """185 × 24 heatmap of mean fragmentation rate.

    Parameters
    ----------
    pivot_df:
        Output of ``fragmentation.day_hour_heatmap()``.
        Index = day (0–184), columns = hour_of_day (0–23).
    """
    out_path = out_path or FIG_DIR / "fig2_frag_heatmap_day_hour.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = figure_double(height_ratio=1.8)

    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        xticklabels=4,
        yticklabels=20,
        cbar_kws={"label": "Mean frag rate", "shrink": 0.6},
        linewidths=0,
        rasterized=True,   # keep PDF small for a 185×24 grid
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Trace day")
    ax.set_title("GPU fragmentation rate: day × hour-of-day")

    # Weekly grid lines (every 7 days)
    for d in range(0, pivot_df.shape[0], 7):
        ax.axhline(d, color="white", lw=0.3, alpha=0.6)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 3 — Fragmentation by priority class (box plot) ───────────────────────

def fig_frag_by_priority(
    pod_frag_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Box plot of per-pod frag rate by priority class.

    Parameters
    ----------
    pod_frag_df:
        Output of ``fragmentation.per_pod_frag_rate()``.
        Required columns: mean_frag_rate, priority.
    """
    out_path = out_path or FIG_DIR / "fig3_frag_by_priority.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pod_frag_df.dropna(subset=["mean_frag_rate", "priority"]).copy()
    df["priority"] = df["priority"].astype(int)
    priority_order = sorted(df["priority"].unique())

    fig, ax = figure_single(height_ratio=1.1)

    data_by_priority = [
        df[df["priority"] == p]["mean_frag_rate"].values
        for p in priority_order
    ]
    bp = ax.boxplot(
        data_by_priority,
        labels=priority_order,
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "lw": 1.5},
        flierprops={"marker": ".", "ms": 2, "alpha": 0.3},
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], [PALETTE["blue"]] * len(priority_order)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Priority class")
    ax.set_ylabel("Per-pod frag rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Fragmentation by priority class")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 4 — Fragmentation by server tier (box plot) ──────────────────────────

def fig_frag_by_tier(
    hourly_frag_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Box plot of hourly frag rate by server tier.

    Parameters
    ----------
    hourly_frag_df:
        Output of ``fragmentation.compute_frag_rate()``.
        Required columns: frag_rate, server_tier.
    """
    out_path = out_path or FIG_DIR / "fig4_frag_by_tier.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = hourly_frag_df.dropna(subset=["frag_rate", "server_tier"])
    tier_order = ["small", "medium", "large", "unknown"]
    tier_order = [t for t in tier_order if t in df["server_tier"].unique()]

    fig, ax = figure_single(height_ratio=1.1)

    data_by_tier = [
        df[df["server_tier"] == tier]["frag_rate"].values
        for tier in tier_order
    ]
    bp = ax.boxplot(
        data_by_tier,
        labels=tier_order,
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "lw": 1.5},
        flierprops={"marker": ".", "ms": 2, "alpha": 0.3},
        showfliers=True,
    )
    for patch, tier in zip(bp["boxes"], tier_order):
        patch.set_facecolor(TIER_COLORS.get(tier, "#888888"))
        patch.set_alpha(0.7)

    ax.set_xlabel("Server tier (by GPU count)")
    ax.set_ylabel("Hourly frag rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Fragmentation by server tier")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Convenience runner ────────────────────────────────────────────────────────

def generate_all() -> None:
    """Load data from standard paths and render all four paper figures."""
    import duckdb
    from src.config import DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS
    from src.analysis.fragmentation import (
        compute_frag_rate, per_pod_frag_rate, day_hour_heatmap,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    print("Loading data …")
    hourly = compute_frag_rate(con)
    pods   = per_pod_frag_rate(con)
    hmap   = day_hour_heatmap(con)

    for name, fn, data in [
        ("Fig 1 CDF by workload",   fig_frag_cdf_by_workload,   pods),
        ("Fig 2 heatmap day×hour",  fig_frag_heatmap_day_hour,  hmap),
        ("Fig 3 by priority",       fig_frag_by_priority,       pods),
        ("Fig 4 by server tier",    fig_frag_by_tier,           hourly),
    ]:
        path = fn(data)
        print(f"  {name} → {path}")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    generate_all()
