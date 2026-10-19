"""Publication-quality figures for the gpu-carbon-waste paper.

Each public function produces and saves one figure to FIG_DIR.
Call generate_all() to produce every figure in sequence.

Usage:
    python -m src.viz.figures
"""

from pathlib import Path
import matplotlib.pyplot as plt

from src.config import FIG_DIR, PROCESSED_DIR
from src.viz.style import apply_ieee_style, figure_single, figure_double

apply_ieee_style()


def fig1_fragmentation_timeseries() -> Path:
    """Fig 1: hourly GPU fragmentation over the 6-month trace window."""
    fig, ax = figure_double(height_ratio=0.8)
    # TODO: load processed/fragmentation.parquet and plot frag_idle over time
    ax.set_xlabel("Date")
    ax.set_ylabel("Fragmentation (idle fraction)")
    ax.set_title("GPU Fragmentation Over Time")
    out = FIG_DIR / "fig1_frag_timeseries.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_carbon_waste_heatmap() -> Path:
    """Fig 2: carbon waste by region × hour-of-day heatmap."""
    fig, ax = figure_double()
    # TODO: load processed/carbon_waste.parquet, pivot to region × hour matrix
    ax.set_title("Carbon Waste (gCO₂eq) by Region and Hour of Day")
    out = FIG_DIR / "fig2_carbon_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_cads_savings() -> Path:
    """Fig 3: CDF of per-job carbon savings under CADS vs baseline."""
    fig, ax = figure_single(height_ratio=1.1)
    # TODO: load outputs from simulation runner, plot CDF
    ax.set_xlabel("Carbon saving per job (gCO₂eq)")
    ax.set_ylabel("CDF")
    ax.set_title("CADS Carbon Savings Distribution")
    out = FIG_DIR / "fig3_cads_savings_cdf.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_all() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fn in [fig1_fragmentation_timeseries, fig2_carbon_waste_heatmap, fig3_cads_savings]:
        path = fn()
        print(f"Saved: {path}")


if __name__ == "__main__":
    generate_all()
