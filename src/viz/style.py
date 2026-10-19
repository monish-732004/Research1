"""IEEE-compliant matplotlib style for publication figures.

IEEE double-column papers use 3.5-inch single-column or 7-inch double-column
figures.  Text size should match the body font (8 pt for most IEEE conference
styles).

Usage
-----
    from src.viz.style import set_ieee_style
    set_ieee_style()
    # or the alias:
    from src.viz.style import apply_ieee_style
    apply_ieee_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── IEEE RC parameters ────────────────────────────────────────────────────────

_IEEE_RC: dict = {
    # Figure size — single column by default
    "figure.figsize":        (3.5, 2.5),
    "figure.dpi":            150,       # screen preview; savefig overrides to 300
    # Font — Times for the paper, fall back gracefully
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             8,
    "axes.titlesize":        8,
    "axes.labelsize":        8,
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "legend.fontsize":       7,
    "legend.framealpha":     0.85,
    # Lines
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.5,
    "patch.linewidth":       0.5,
    # Grid
    "axes.grid":             True,
    "grid.linestyle":        "--",
    "grid.linewidth":        0.35,
    "grid.alpha":            0.55,
    # Layout
    "figure.constrained_layout.use": True,
    # Output — always PDF-quality, embed fonts
    "savefig.dpi":           300,
    "savefig.format":        "pdf",
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
}

# ── Colour-blind-safe, print-safe palette ─────────────────────────────────────
# Based on Wong (2011) "Points of view: Color blindness", Nature Methods.
# All 8 colours are distinguishable under deuteranopia/protanopia and in
# greyscale print.

PALETTE: dict[str, str] = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}

# Ordered list for matplotlib colour cycle
PALETTE_LIST: list[str] = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["purple"],
    PALETTE["sky"],
    PALETTE["yellow"],
    PALETTE["black"],
]

# Workload-type colour mapping (consistent across all paper figures)
WORKLOAD_COLORS: dict[str, str] = {
    "training":         PALETTE["blue"],
    "inference":        PALETTE["orange"],
    "ad_hoc":           PALETTE["green"],
    "offline_inference":PALETTE["purple"],
    "unknown":          "#AAAAAA",
}

# Server tier colour mapping
TIER_COLORS: dict[str, str] = {
    "small":   PALETTE["sky"],
    "medium":  PALETTE["blue"],
    "large":   PALETTE["red"],
    "unknown": "#AAAAAA",
}


# ── Apply functions ───────────────────────────────────────────────────────────

def set_ieee_style() -> None:
    """Apply IEEE RC params to the global matplotlib state."""
    mpl.rcParams.update(_IEEE_RC)
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=PALETTE_LIST)


# Alias for backward compatibility
apply_ieee_style = set_ieee_style


# ── Convenience figure factories ──────────────────────────────────────────────

def figure_single(height_ratio: float = 1.0) -> tuple:
    """Return (fig, ax) sized for a single IEEE column (3.5 × 2.5 in)."""
    w, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(figsize=(w, h * height_ratio))


def figure_double(height_ratio: float = 1.0) -> tuple:
    """Return (fig, ax) sized for a full IEEE double column (7.0 × 2.5 in)."""
    _, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(figsize=(7.0, h * height_ratio))


def figure_1x2(height_ratio: float = 1.0) -> tuple:
    """Return (fig, (ax1, ax2)) for a side-by-side double-column layout."""
    _, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(1, 2, figsize=(7.0, h * height_ratio))


def figure_2x2(height_ratio: float = 1.0) -> tuple:
    """Return (fig, axes) for a 2×2 double-column layout."""
    _, h = _IEEE_RC["figure.figsize"]
    return plt.subplots(2, 2, figsize=(7.0, h * height_ratio * 2))
