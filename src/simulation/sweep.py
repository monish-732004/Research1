"""Parameter sweep for CADS simulation.

Doubles when predictive_mode is included: 300 baseline runs × 2 = 600 total.

Usage
-----
    python -m src.simulation.sweep --dry-run
"""

from __future__ import annotations

import argparse
import itertools

# ── Sweep grid ────────────────────────────────────────────────────────────────

SWEEP_GRID = {
    "mode":               ["baseline", "cads"],
    "predictive_mode":    [False, True],
    "green_percentile":   [0.15, 0.25, 0.35],
    "max_defer_minutes":  [15, 30, 60],
    "region":             ["cn-north", "cn-east", "cn-south", "cn-northwest"],
    "anchor":             ["2025-01-01", "2025-04-01", "2025-07-01"],
}


def enumerate_configs() -> list[dict]:
    """Return all (mode, predictive_mode, ...) combinations.

    Constraints:
      - predictive_mode=True only applies to mode=cads
      - baseline runs ignore green_percentile / max_defer_minutes
    """
    configs = []
    keys = list(SWEEP_GRID.keys())
    for vals in itertools.product(*SWEEP_GRID.values()):
        cfg = dict(zip(keys, vals))
        # Skip predictive=True with baseline (no-op)
        if cfg["mode"] == "baseline" and cfg["predictive_mode"]:
            continue
        # Baseline: deduplicate (region/anchor/defer settings don't affect it)
        if cfg["mode"] == "baseline":
            if cfg["green_percentile"] != 0.25:
                continue
            if cfg["max_defer_minutes"] != 30:
                continue
            if cfg["region"] != "cn-north":
                continue
            if cfg["anchor"] != "2025-01-01":
                continue
        configs.append(cfg)
    return configs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CADS parameter sweep.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    configs = enumerate_configs()
    print(f"Total sweep configurations: {len(configs)}")

    if args.dry_run:
        for i, cfg in enumerate(configs[:10]):
            print(f"  [{i+1:3d}] {cfg}")
        if len(configs) > 10:
            print(f"  ... ({len(configs) - 10} more)")
        return

    print("Simulation runner not yet implemented — use --dry-run to inspect the grid.")


if __name__ == "__main__":
    main()
