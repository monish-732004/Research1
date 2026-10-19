"""IEEE-compliant matplotlib style for publication figures.

IEEE double-column papers use 3.5-inch single-column or 7-inch double-column
figures.  Text size must match the body font (9–10 pt).

Apply at module top level:
    from src.viz.style import apply_ieee_style
    apply_ieee_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE double-column body text is 9 pt; match axis labels/ticks to body size.
_IEEE_RC = {
    # Figure size
    "figure.figsize":        (3.5, 2.4),   # single-column default (inches)
    "figure.dpi":            300,
    # Font
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":             9,
    "axes.titlesize":        9,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    # Lines
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.6,
    # Grid
    "axes.grid":             True,
    "grid.linestyle":        "--",
    "grid.linewidth":        0.4,
    "grid.alpha":            0.6,
    # Layout
    "figure.constrained_layout.use": True,
    # Output
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "pdf.fonttype":          42,   # embed fonts in PDF
    "ps.fonttype":           42,
}

# Accessible colour palette (colour-blind safe, prints in greyscale)
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}

PALETTE_LIST = list(PALETTE.values())


def apply_ieee_style() -> None:
    """Apply IEEE RC params to the global matplotlib state."""
    mpl.rcParams.update(_IEEE_RC)
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE_LIST)


def figure_single(height_ratio: float = 1.0) -> tuple:
    """Return (fig, ax) sized for a single IEEE column."""
    w, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(figsize=(w, h * height_ratio))


def figure_double(height_ratio: float = 1.0) -> tuple:
    """Return (fig, ax) sized for a full IEEE double column (7 in wide)."""
    _, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(figsize=(7.0, h * height_ratio))
