"""Thin adapter for Alibaba cluster-trace-gpu-v2023.

The v2023 trace (~6,200 GPUs) uses the same logical schema as v2026 but
may have different concrete column names, different value encodings, and
different table filenames.  This module:

  1. Defines v2023-specific column alias overrides (merged on top of the
     standard aliases in canonicalize.py).
  2. Defines known v2023 filename mappings.
  3. Runs the full canonicalize → completeness → truncation → aggregate
     pipeline with those overrides so that the same analysis code can
     consume both traces without modification.

Known schema differences (v2023 vs v2026)
------------------------------------------
- GPU utilisation is sometimes stored as a 0–100 integer (auto-normalised
  by the heuristic in canonicalize._build_pod_select).
- The trace spans ~90 days instead of 185; TRACE_END_HOUR is lower.
- Priority may be encoded as a string ('high', 'medium', 'low') rather
  than an integer; handled via a CASE expression below.

Usage
-----
    python -m src.ingest.v2023_adapter
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from src.config import (
    RAW_DIR, INTERIM_DIR, LOG_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
)
from src.preprocess import canonicalize


# ── v2023 alias overrides ─────────────────────────────────────────────────────
# These are *merged on top of* the default aliases in canonicalize._POD_ALIASES.
# Only columns that differ need to appear here.

V2023_POD_ALIAS_OVERRIDES: dict[str, list[str]] = {
    "pod_uid":       ["pod_name", "pod_uid", "pod_id", "task_id", "job_id",
                      "container_id"],
    "gpu_requested": ["gpu_num", "plan_gpu", "gpu_request", "gpu_requested"],
    "gpu_util":      ["gpu_util_avg", "gpu_util", "avg_gpu_util",
                      "gpu_utilization", "gpu_util_rate"],
    "state":         ["status", "pod_status", "state", "pod_state",
                      "task_status", "container_status"],
    "workload_type": ["task_type", "job_type", "workload_type", "workload",
                      "category", "model_category"],
}

# Known alternative filenames in v2023 releases
V2023_POD_FILENAMES = [
    "pod_list.csv.parquet", "pod_list.parquet",
    "pai_task_table.parquet", "task_table.parquet",
    "instance_events.parquet",
]

V2023_SERVER_FILENAMES = [
    "machine_list.parquet", "server_list.parquet",
    "machine_events.parquet",
]


# ── Priority string → integer mapping ────────────────────────────────────────

# If priority is stored as a string in v2023, coerce it to an integer tier.
# Lower integer = higher urgency (common convention).
PRIORITY_STRING_MAP = {
    "best_effort": 0, "besteffort": 0,
    "low":  1, "background": 1,
    "mid":  2, "medium": 2, "normal": 2,
    "high": 3, "latency_critical": 4, "critical": 4,
}


# ── Merged alias builder ──────────────────────────────────────────────────────

def _merged_aliases() -> dict[str, list[str]]:
    merged = dict(canonicalize._POD_ALIASES)   # start from standard aliases
    for key, override_list in V2023_POD_ALIAS_OVERRIDES.items():
        # Prepend v2023 candidates so they take priority
        existing = merged.get(key, [])
        merged[key] = list(dict.fromkeys(override_list + existing))
    return merged


# ── Canonical table builder ───────────────────────────────────────────────────

def _find_source(candidates: list[str], raw_dir: Path) -> Path | None:
    return next((raw_dir / f for f in candidates if (raw_dir / f).exists()), None)


def run(*, force: bool = False) -> dict[str, Any]:
    """Canonicalize v2023 trace with v2023-specific aliases.

    Writes the same canonical files as canonicalize.run('v2026') so the
    downstream analysis pipeline can be applied without changes.
    """
    trace = "v2023"
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    pod_out    = INTERIM_DIR / "pod_hourly_canonical.parquet"
    server_out = INTERIM_DIR / "server_hourly_canonical.parquet"

    if pod_out.exists() and not force:
        rows = duckdb.execute(
            f"SELECT count(*) FROM read_parquet('{pod_out}')"
        ).fetchone()[0]
        print(f"  [v2023_adapter] output exists ({rows:,} rows) — skipping")
        return {"rows_out": rows, "skipped": True}

    raw_dir = RAW_DIR / trace
    pod_src = _find_source(V2023_POD_FILENAMES, raw_dir)
    if pod_src is None:
        # Fall back to the standard canonicalize search
        return canonicalize.run(trace, force=force)

    server_src = _find_source(V2023_SERVER_FILENAMES, raw_dir)

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    merged = _merged_aliases()
    actual = canonicalize._actual_columns(pod_src)
    mapping = canonicalize._discover_mapping(actual, merged)

    # Priority might be a string in v2023 — check and handle
    priority_col = mapping.get("priority")
    if priority_col:
        sample = con.execute(
            f"SELECT \"{priority_col}\" FROM read_parquet('{pod_src}') LIMIT 100"
        ).df()[priority_col].dropna()
        if sample.dtype == object and len(sample) > 0:
            # Build CASE expression for string → int mapping
            cases = "\n".join(
                f"    WHEN lower(CAST(\"{priority_col}\" AS VARCHAR)) = '{k}' THEN {v}"
                for k, v in PRIORITY_STRING_MAP.items()
            )
            mapping["_priority_sql"] = (
                f"CASE\n{cases}\n    ELSE NULL END"
            )

    # Build and execute SQL using the standard helper but with merged mapping
    sql = canonicalize._build_pod_select(pod_src, mapping, con)

    # Patch priority expression if string-encoded
    if "_priority_sql" in mapping:
        priority_expr = mapping["_priority_sql"]
        sql = sql.replace(
            f'CAST("{priority_col}" AS INTEGER) AS priority',
            f'({priority_expr}) AS priority',
        )

    con.execute(f"COPY ({sql}) TO '{pod_out}' (FORMAT PARQUET)")
    pod_rows = con.execute(f"SELECT count(*) FROM read_parquet('{pod_out}')").fetchone()[0]
    print(f"  [v2023_adapter] pod_hourly_canonical: {pod_rows:,} rows")

    # Server table (optional)
    if server_src:
        actual_s = canonicalize._actual_columns(server_src)
        server_mapping = canonicalize._discover_mapping(actual_s, canonicalize._SERVER_ALIASES)
        sql_s = canonicalize._build_server_select(server_src, server_mapping)
        con.execute(f"COPY ({sql_s}) TO '{server_out}' (FORMAT PARQUET)")
        srv_rows = con.execute(f"SELECT count(*) FROM read_parquet('{server_out}')").fetchone()[0]
        print(f"  [v2023_adapter] server_hourly_canonical: {srv_rows:,} rows")

    canonicalize._write_discovered(trace, mapping, None)
    return {"rows_out": pod_rows, "skipped": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Canonicalize v2023 trace.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(force=args.force)


if __name__ == "__main__":
    main()
