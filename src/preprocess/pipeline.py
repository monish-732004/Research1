"""Preprocessing pipeline driver.

Runs stages A → B → C → D in order, times each stage, and writes
outputs/logs/preprocess_timings.json.

Usage
-----
    python -m src.preprocess.pipeline --trace v2026
    python -m src.preprocess.pipeline --trace v2026 --force
    python -m src.preprocess.pipeline --trace v2026 --stages canonicalize,truncation
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

from src.config import LOG_DIR
from src.preprocess import canonicalize, tdp, aggregate

# Stages: A → B → C  (truncation flags are embedded in canonicalize Stage A)
STAGE_ORDER: list[tuple[str, Callable]] = [
    ("canonicalize", canonicalize.run),
    ("tdp",          tdp.run),
    ("aggregate",    aggregate.run),
]


def run_pipeline(
    trace: str,
    *,
    force: bool = False,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    to_run = STAGE_ORDER
    if stages is not None:
        known = {name for name, _ in STAGE_ORDER}
        unknown = set(stages) - known
        if unknown:
            raise ValueError(f"Unknown stage(s): {unknown}. Known: {sorted(known)}")
        to_run = [(n, fn) for n, fn in STAGE_ORDER if n in stages]

    timings: dict[str, Any] = {}
    total_start = time.perf_counter()

    for name, fn in to_run:
        print(f"\n[pipeline] ── {name} ──")
        t0 = time.perf_counter()
        result = fn(trace, force=force)
        elapsed = round(time.perf_counter() - t0, 2)
        timings[name] = {**result, "elapsed_s": elapsed}
        status = "skipped" if result.get("skipped") else "done"
        print(f"[pipeline] {name}: {status} in {elapsed}s")

    total_elapsed = round(time.perf_counter() - total_start, 2)
    report = {
        "trace": trace,
        "total_elapsed_s": total_elapsed,
        "stages": timings,
    }

    out = LOG_DIR / f"preprocess_timings_{trace}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[pipeline] Complete in {total_elapsed}s  →  {out}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true",
                        help="Rerun stages even if outputs exist.")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated subset of stages to run, e.g. canonicalize,truncation")
    args = parser.parse_args(argv)

    stage_list = [s.strip() for s in args.stages.split(",")] if args.stages else None
    run_pipeline(args.trace, force=args.force, stages=stage_list)


if __name__ == "__main__":
    main()
