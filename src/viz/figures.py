"""Publication-quality figures for the gpu-carbon-waste paper.

All functions accept a small pandas DataFrame and return the saved Path.

Figures
-------
  fig_frag_cdf_by_spec        Fig 1 — CDF of server frag_rate by GPU spec
  fig_frag_heatmap_day_hour   Fig 2 — 185×24 heatmap of fleet GPU availability
  fig_frag_by_priority        Fig 3 — GPU-hours by priority class (bar)
  fig_frag_by_tier            Fig 4 — box plot of frag_rate by server tier
  generate_all                Convenience runner

Usage
-----
    python -m src.viz.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIG_DIR
from src.viz.style import (
    set_ieee_style, figure_single, figure_double,
    PALETTE, TIER_COLORS,
)

set_ieee_style()

_TOP_N_SPECS = 8   # GPU spec lines to show on CDF


# ── Fig 1 — CDF of server allocation-gap by GPU spec ─────────────────────────

def fig_frag_cdf_by_spec(
    frag_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """CDF of per-server frag_rate overlaid by GPU spec.

    Parameters
    ----------
    frag_df:
        Output of ``fragmentation.compute_frag_rate()``.
        Required columns: frag_rate, gpu_spec_public.
    """
    out_path = out_path or FIG_DIR / "fig1_frag_cdf_by_spec.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = frag_df.dropna(subset=["frag_rate"])
    top_specs = (
        df.groupby("gpu_spec_public")["frag_rate"]
        .count()
        .sort_values(ascending=False)
        .head(_TOP_N_SPECS)
        .index.tolist()
    )

    colors = list(PALETTE.values())
    fig, ax = figure_single(height_ratio=1.1)

    for i, spec in enumerate(top_specs):
        vals = df[df["gpu_spec_public"] == spec]["frag_rate"].sort_values()
        n = len(vals)
        if n == 0:
            continue
        cdf = np.arange(1, n + 1) / n
        ax.plot(vals.values, cdf, label=spec,
                color=colors[i % len(colors)], lw=1.0)

    ax.axvline(0.5, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Server allocation gap (frag rate)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(loc="lower right", handlelength=1.2, fontsize=6)
    ax.set_title("Allocation-gap CDF by GPU spec")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 2 — Heatmap of fleet GPU availability (day × hour-of-day) ─────────────

def fig_frag_heatmap_day_hour(
    pivot_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """185 × 24 heatmap of total GPUs available.

    Parameters
    ----------
    pivot_df:
        Output of ``fragmentation.day_hour_heatmap()``.
        Index = day (0–184), columns = hour_int (0–23).
    """
    out_path = out_path or FIG_DIR / "fig2_fleet_heatmap_day_hour.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Normalise to thousands for readability
    plot_df = pivot_df / 1000.0

    fig, ax = figure_double(height_ratio=1.8)

    sns.heatmap(
        plot_df,
        ax=ax,
        cmap="Blues",
        xticklabels=4,
        yticklabels=20,
        cbar_kws={"label": "GPUs available (K)", "shrink": 0.6},
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Trace day")
    ax.set_title("Fleet GPU availability: day × hour-of-day")

    for d in range(0, pivot_df.shape[0], 7):
        ax.axhline(d, color="white", lw=0.3, alpha=0.6)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 3 — GPU-hours by priority class (bar chart) ──────────────────────────

def fig_frag_by_priority(
    priority_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Horizontal bar chart of total GPU-hours by priority class.

    Parameters
    ----------
    priority_df:
        Output of ``fragmentation.aggregate_by_priority()``.
        Required columns: priority_class, total_gpu_hours.
    """
    out_path = out_path or FIG_DIR / "fig3_gpu_hours_by_priority.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = priority_df.copy()
    df["gpu_hours_M"] = df["total_gpu_hours"] / 1e6

    fig, ax = figure_single(height_ratio=1.0)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
              PALETTE["red"], PALETTE["purple"]]
    bars = ax.barh(
        df["priority_class"],
        df["gpu_hours_M"],
        color=[colors[i % len(colors)] for i in range(len(df))],
    )

    for bar, pct in zip(bars, df.get("pct_gpu_hours", [None]*len(df))):
        if pct is not None:
            ax.text(bar.get_width() + df["gpu_hours_M"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=6)

    ax.set_xlabel("GPU-hours requested (M)")
    ax.set_title("GPU-hours by priority class")
    ax.invert_yaxis()

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Fig 4 — Frag rate by server tier (box plot) ───────────────────────────────

def fig_frag_by_tier(
    frag_df: pd.DataFrame,
    out_path: Path | None = None,
) -> Path:
    """Box plot of per-server allocation-gap frag_rate by server tier.

    Parameters
    ----------
    frag_df:
        Output of ``fragmentation.compute_frag_rate()``.
        Required columns: frag_rate, server_tier.
    """
    out_path = out_path or FIG_DIR / "fig4_frag_by_tier.pdf"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = frag_df.dropna(subset=["frag_rate", "server_tier"])
    tier_order = [t for t in ["small", "medium", "large", "unknown"]
                  if t in df["server_tier"].unique()]

    fig, ax = figure_single(height_ratio=1.1)
    data_by_tier = [df[df["server_tier"] == t]["frag_rate"].values for t in tier_order]
    bp = ax.boxplot(
        data_by_tier,
        labels=tier_order,
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "lw": 1.5},
        flierprops={"marker": ".", "ms": 2, "alpha": 0.3},
    )
    for patch, tier in zip(bp["boxes"], tier_order):
        patch.set_facecolor(TIER_COLORS.get(tier, "#888888"))
        patch.set_alpha(0.7)

    ax.set_xlabel("Server tier (by GPU count)")
    ax.set_ylabel("Allocation gap (frag rate)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Allocation gap by server tier")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ── Convenience runner ────────────────────────────────────────────────────────

def generate_all() -> None:
    """Load data from standard paths and render all four paper figures."""
    import duckdb
    from src.config import DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS
    from src.analysis.fragmentation import (
        compute_frag_rate, aggregate_by_priority, day_hour_heatmap,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    print("Loading data …")
    frag_df     = compute_frag_rate(con)
    priority_df = aggregate_by_priority(con)
    hmap_df     = day_hour_heatmap(con)

    for name, fn, data in [
        ("Fig 1 CDF by GPU spec",     fig_frag_cdf_by_spec,       frag_df),
        ("Fig 2 heatmap day×hour",    fig_frag_heatmap_day_hour,  hmap_df),
        ("Fig 3 by priority class",   fig_frag_by_priority,       priority_df),
        ("Fig 4 by server tier",      fig_frag_by_tier,           frag_df),
    ]:
        path = fn(data)
        print(f"  {name} → {path}")

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    generate_all()
