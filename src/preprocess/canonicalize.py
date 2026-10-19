"""Stage A — Schema canonicalization.

Reads the raw Parquet files for a trace version, resolves actual column names
against known aliases, casts to stable types, and writes canonical tables:

  data/interim/pod_hourly_canonical.parquet
  data/interim/server_hourly_canonical.parquet  (if source exists)

All heavy lifting is DuckDB SQL — no Python row loops.

Usage
-----
    python -m src.preprocess.canonicalize --trace v2026
    python -m src.preprocess.canonicalize --trace v2026 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from src.config import (
    RAW_DIR, INTERIM_DIR, LOG_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
)

# ── Column alias maps ─────────────────────────────────────────────────────────
# Each entry maps canonical_name → [list of known source column names, in
# preference order].  The first alias present in the actual schema wins.
# If none match, the canonical name itself is tried as a fallback.

_POD_ALIASES: dict[str, list[str]] = {
    "pod_uid":       ["pod_uid", "pod_id", "container_id", "poduid",
                      "job_id", "task_id", "instance_id"],
    "day":           ["day", "day_id", "day_num", "date_day"],
    "hour":          ["hour", "hour_id", "hour_num"],
    "gpu_requested": ["gpu_num", "gpu_request", "gpu_request_num",
                      "num_gpu_request", "gpu_requested", "gpu_num_request",
                      "plan_gpu"],
    "gpu_util":      ["gpu_util", "avg_gpu_util", "gpu_utilization",
                      "avg_gpu_utilization", "gpu_used_frac",
                      "gpu_milli_usage", "gpu_usage"],
    "priority":      ["priority", "pri", "priority_class", "job_priority",
                      "task_priority"],
    "state":         ["state", "pod_state", "status", "job_state",
                      "container_state", "task_state"],
    # optional: pod → server mapping (may be absent)
    "server_uid":    ["server_uid", "machine_id", "node_id", "worker_id",
                      "machine_uid", "server_id"],
    # optional: workload category (training / inference / ad-hoc / offline-inference)
    "workload_type": ["workload_type", "job_type", "model_type", "task_type",
                      "workload", "category", "work_type", "app_type"],
}

_SERVER_ALIASES: dict[str, list[str]] = {
    "server_uid":       ["server_uid", "machine_id", "node_id", "worker_id",
                         "machine_uid", "server_id"],
    "day":              ["day", "day_id", "day_num"],
    "hour":             ["hour", "hour_id", "hour_num"],
    "gpu_count_total":  ["gpu_count", "num_gpus", "total_gpu",
                         "gpu_count_total", "gpu_total", "gpu_num"],
    "cpu_count_total":  ["cpu_count", "num_cpus", "total_cpu",
                         "cpu_count_total", "cpu_total", "cpu_num"],
    "asw_id":           ["asw_id", "asw", "availability_zone", "az",
                         "zone_id", "dc_id", "datacenter_id"],
    "rack_id":          ["rack_id", "rack", "rack_name"],
}

# SQL casts for each canonical column
_POD_CASTS: dict[str, str] = {
    "pod_uid":       "VARCHAR",
    "day":           "INTEGER",
    "hour":          "INTEGER",
    "gpu_requested": "INTEGER",
    "gpu_util":      "FLOAT",
    "priority":      "INTEGER",
    "state":         "VARCHAR",
    "server_uid":    "VARCHAR",
    "workload_type": "VARCHAR",
}

_SERVER_CASTS: dict[str, str] = {
    "server_uid":      "VARCHAR",
    "day":             "INTEGER",
    "hour":            "INTEGER",
    "gpu_count_total": "INTEGER",
    "cpu_count_total": "INTEGER",
    "asw_id":          "VARCHAR",
    "rack_id":         "VARCHAR",
}


# ── Schema helpers ────────────────────────────────────────────────────────────

def _actual_columns(parquet_path: Path) -> set[str]:
    schema = pq.read_schema(parquet_path)
    return {f.name.lower() for f in schema}


def _resolve(actual: set[str], canonical: str, aliases: list[str]) -> str | None:
    """Return the first alias present in *actual*, else None."""
    for alias in aliases:
        if alias.lower() in actual:
            return alias
    # exact canonical name as last resort
    if canonical.lower() in actual:
        return canonical
    return None


def _discover_mapping(
    actual: set[str],
    aliases: dict[str, list[str]],
) -> dict[str, str | None]:
    """Map canonical_name → actual_column_name (or None if absent)."""
    return {
        canonical: _resolve(actual, canonical, alias_list)
        for canonical, alias_list in aliases.items()
    }


def _gpu_util_needs_scale(parquet_path: Path, col: str, con: duckdb.DuckDBPyConnection) -> bool:
    """Return True if gpu_util values appear to be percentages (max > 1.0)."""
    row = con.execute(
        f"SELECT max(\"{col}\") FROM read_parquet('{parquet_path}')"
    ).fetchone()
    return row[0] is not None and row[0] > 1.5   # allow tiny float noise


def _build_pod_select(
    parquet_path: Path,
    mapping: dict[str, str | None],
    con: duckdb.DuckDBPyConnection,
) -> str:
    """Build SELECT ... AS ... SQL for pod_hourly_canonical."""
    parts: list[str] = []

    # Required columns — if absent write NULL with correct type
    for col in ("pod_uid", "day", "hour", "state"):
        src = mapping.get(col)
        cast = _POD_CASTS[col]
        if src:
            if col == "state":
                parts.append(f"UPPER(COALESCE(CAST(\"{src}\" AS {cast}), 'UNKNOWN')) AS {col}")
            else:
                parts.append(f"CAST(\"{src}\" AS {cast}) AS {col}")
        else:
            if col == "state":
                parts.append(f"'UNKNOWN' AS {col}")
            else:
                parts.append(f"NULL::{cast} AS {col}")

    # rel_hour derived
    parts.append("(CAST(day AS INTEGER) * 24 + CAST(hour AS INTEGER)) AS rel_hour")

    # gpu_requested
    src = mapping.get("gpu_requested")
    if src:
        parts.append(f"CAST(\"{src}\" AS INTEGER) AS gpu_requested")
    else:
        parts.append("NULL::INTEGER AS gpu_requested")

    # gpu_util (may need /100 normalisation)
    src = mapping.get("gpu_util")
    if src:
        if _gpu_util_needs_scale(parquet_path, src, con):
            parts.append(f"CAST(\"{src}\" AS FLOAT) / 100.0 AS gpu_util")
        else:
            parts.append(f"CAST(\"{src}\" AS FLOAT) AS gpu_util")
    else:
        parts.append("NULL::FLOAT AS gpu_util")

    # priority
    src = mapping.get("priority")
    if src:
        parts.append(f"CAST(\"{src}\" AS INTEGER) AS priority")
    else:
        parts.append("NULL::INTEGER AS priority")

    # server_uid (optional)
    src = mapping.get("server_uid")
    if src:
        parts.append(f"CAST(\"{src}\" AS VARCHAR) AS server_uid")
    else:
        parts.append("NULL::VARCHAR AS server_uid")

    # workload_type (optional) — normalise to lowercase known values
    src = mapping.get("workload_type")
    if src:
        parts.append(
            f"CASE lower(CAST(\"{src}\" AS VARCHAR))\n"
            f"        WHEN 'training'           THEN 'training'\n"
            f"        WHEN 'train'              THEN 'training'\n"
            f"        WHEN 'inference'          THEN 'inference'\n"
            f"        WHEN 'infer'              THEN 'inference'\n"
            f"        WHEN 'offline_inference'  THEN 'offline_inference'\n"
            f"        WHEN 'offline-inference'  THEN 'offline_inference'\n"
            f"        WHEN 'ad_hoc'             THEN 'ad_hoc'\n"
            f"        WHEN 'ad-hoc'             THEN 'ad_hoc'\n"
            f"        WHEN 'dev'                THEN 'ad_hoc'\n"
            f"        ELSE lower(CAST(\"{src}\" AS VARCHAR))\n"
            f"    END AS workload_type"
        )
    else:
        parts.append("NULL::VARCHAR AS workload_type")

    cols = ",\n    ".join(parts)
    return f"SELECT\n    {cols}\nFROM read_parquet('{parquet_path}')"


def _build_server_select(
    parquet_path: Path,
    mapping: dict[str, str | None],
) -> str:
    parts: list[str] = []
    for col in ("server_uid", "day", "hour"):
        src = mapping.get(col)
        cast = _SERVER_CASTS[col]
        parts.append(f"CAST(\"{src}\" AS {cast}) AS {col}" if src else f"NULL::{cast} AS {col}")

    parts.append("(CAST(day AS INTEGER) * 24 + CAST(hour AS INTEGER)) AS rel_hour")

    for col in ("gpu_count_total", "cpu_count_total"):
        src = mapping.get(col)
        cast = _SERVER_CASTS[col]
        parts.append(f"CAST(\"{src}\" AS {cast}) AS {col}" if src else f"NULL::{cast} AS {col}")

    for col in ("asw_id", "rack_id"):
        src = mapping.get(col)
        parts.append(f"CAST(\"{src}\" AS VARCHAR) AS {col}" if src else f"NULL::VARCHAR AS {col}")

    cols = ",\n    ".join(parts)
    return f"SELECT\n    {cols}\nFROM read_parquet('{parquet_path}')"


# ── Discovered-schema writer (feeds downstream tooling) ───────────────────────

def _write_discovered(trace: str, pod_mapping: dict, server_mapping: dict | None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"discovered_schema_{trace}.json"
    out.write_text(json.dumps(
        {"trace": trace, "pod": pod_mapping, "server": server_mapping},
        indent=2,
    ))


# ── Main run function ─────────────────────────────────────────────────────────

def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    """Canonicalize pod and server Parquet files for *trace*.

    Returns dict with keys: rows_out, skipped.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pod_out    = INTERIM_DIR / "pod_hourly_canonical.parquet"
    server_out = INTERIM_DIR / "server_hourly_canonical.parquet"

    if pod_out.exists() and not force:
        rows = duckdb.execute(
            f"SELECT count(*) FROM read_parquet('{pod_out}')"
        ).fetchone()[0]
        print(f"  [canonicalize] output exists ({rows:,} rows) — skipping (use --force to rerun)")
        return {"rows_out": rows, "skipped": True}

    raw_dir = RAW_DIR / trace
    # Locate pod and server source files
    pod_candidates = [
        "pod_hourly.parquet", "instance_events.parquet",
        "resource_usage.parquet", "resource_request.parquet",
    ]
    server_candidates = [
        "server_hourly.parquet", "machine_events.parquet",
        "machine_spec.parquet",
    ]

    pod_src = next((raw_dir / f for f in pod_candidates if (raw_dir / f).exists()), None)
    server_src = next((raw_dir / f for f in server_candidates if (raw_dir / f).exists()), None)

    if pod_src is None:
        raise FileNotFoundError(
            f"No pod-level source Parquet found in {raw_dir}. "
            f"Expected one of: {pod_candidates}"
        )

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # Pod table
    actual = _actual_columns(pod_src)
    pod_mapping = _discover_mapping(actual, _POD_ALIASES)
    sql = _build_pod_select(pod_src, pod_mapping, con)
    con.execute(f"COPY ({sql}) TO '{pod_out}' (FORMAT PARQUET)")
    pod_rows = con.execute(f"SELECT count(*) FROM read_parquet('{pod_out}')").fetchone()[0]
    print(f"  [canonicalize] pod_hourly_canonical: {pod_rows:,} rows")

    # Server table (optional)
    server_mapping = None
    if server_src:
        actual_s = _actual_columns(server_src)
        server_mapping = _discover_mapping(actual_s, _SERVER_ALIASES)
        sql_s = _build_server_select(server_src, server_mapping)
        con.execute(f"COPY ({sql_s}) TO '{server_out}' (FORMAT PARQUET)")
        srv_rows = con.execute(f"SELECT count(*) FROM read_parquet('{server_out}')").fetchone()[0]
        print(f"  [canonicalize] server_hourly_canonical: {srv_rows:,} rows")
    else:
        print("  [canonicalize] no server source found — server_hourly_canonical skipped")

    _write_discovered(trace, pod_mapping, server_mapping)
    return {"rows_out": pod_rows, "skipped": False}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Canonicalize trace Parquet schemas.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
