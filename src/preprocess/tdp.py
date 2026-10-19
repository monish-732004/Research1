"""GPU TDP lookup table (Stage B).

Maps gpu_spec_public strings (as they appear in the Alibaba trace) to
thermal design power in watts, using public Nvidia/AMD spec sheets.

Usage
-----
    python -m src.preprocess.tdp --trace v2026
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import duckdb

from src.config import (
    LOG_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    table_glob,
)

# Public spec-sheet TDP values (watts, board TDP / SXM TDP where applicable)
GPU_TDP: dict[str, int] = {
    "A10":     150,
    "A100":    400,
    "A800":    400,   # A100 variant for China market
    "H100":    700,
    "H800":    700,   # H100 variant for China market
    "L20":     275,
    "L40":     300,
    "L40S":    350,
    "V100":    300,
    "T4":       70,
    "A30":     165,
    "A40":     300,
    "RTX3090": 350,
    "RTX4090": 450,
    "unknown": 350,   # conservative fleet-average fallback
}


def get_tdp(gpu_spec_public: str) -> int:
    """Return TDP (W) for *gpu_spec_public*.

    Tries exact match, then prefix match (e.g. "A100-80GB" → "A100"),
    then falls back to the "unknown" sentinel.
    """
    if not gpu_spec_public or gpu_spec_public.strip().lower() in ("", "unknown", "none"):
        return GPU_TDP["unknown"]

    spec = gpu_spec_public.strip()

    if spec in GPU_TDP:
        return GPU_TDP[spec]

    for key in sorted(GPU_TDP, key=len, reverse=True):
        if key == "unknown":
            continue
        if spec.upper().startswith(key.upper()):
            return GPU_TDP[key]

    return GPU_TDP["unknown"]


def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    """Scan distinct gpu_spec_public values and report TDP coverage."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"tdp_coverage_{trace}.json"

    if out_path.exists() and not force:
        return {"skipped": True, "output": str(out_path)}

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    specs: set[str] = set()
    for table in ("asi_opensource_server_hourly", "asi_opensource_job_execution_summary"):
        glob = table_glob(table, trace).replace("\\", "/")
        try:
            rows = con.execute(
                f"SELECT DISTINCT gpu_spec_public FROM read_parquet('{glob}') "
                f"WHERE gpu_spec_public IS NOT NULL"
            ).fetchall()
            specs.update(r[0] for r in rows)
        except Exception as e:
            print(f"  Warning: could not scan {table}: {e}")

    coverage: list[dict] = []
    fallbacks: list[str] = []
    for spec in sorted(specs):
        tdp = get_tdp(spec)
        method = "exact" if spec in GPU_TDP else (
            "prefix" if any(spec.upper().startswith(k.upper())
                            for k in GPU_TDP if k != "unknown") else "fallback"
        )
        coverage.append({"gpu_spec_public": spec, "tdp_watts": tdp, "match": method})
        if method == "fallback":
            fallbacks.append(spec)

    report = {
        "trace": trace,
        "distinct_specs": len(specs),
        "fallback_count": len(fallbacks),
        "fallback_specs": fallbacks,
        "coverage": coverage,
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"  TDP coverage: {len(specs)} GPU specs, {len(fallbacks)} fallback(s)")
    if fallbacks:
        print(f"  Fallbacks: {fallbacks}")
    print(f"  Written: {out_path}")
    return {"output": str(out_path), "distinct_specs": len(specs), "fallbacks": fallbacks}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="GPU TDP coverage report.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
