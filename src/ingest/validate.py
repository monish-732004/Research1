"""Validate ingested Parquet files against expected schemas.

Usage
-----
    python -m src.ingest.validate --trace v2026
    python -m src.ingest.validate --trace v2023 --out outputs/logs/validation_v2023.json

For each table the validator records:
  - file size on disk
  - row count
  - column schema (actual vs expected, any mismatches)
  - null rate per column
  - min / max timestamp (for tables that have a timestamp column)
  - overall pass / fail

Writes a JSON report to outputs/logs/validation_{trace}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.config import RAW_DIR, LOG_DIR, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS
from src.ingest.schema import SCHEMAS, schema_diff, SchemaMismatch

TIMESTAMP_COLUMNS = {"timestamp", "submit_time"}  # column names to look for


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _validate_table(
    table: str,
    parquet_path: Path,
    con: duckdb.DuckDBPyConnection,
) -> dict:
    result: dict = {
        "table": table,
        "path": str(parquet_path),
        "file_size": parquet_path.stat().st_size,
        "file_size_human": _human_size(parquet_path.stat().st_size),
        "pass": True,
        "errors": [],
        "warnings": [],
    }

    # ── Row count ─────────────────────────────────────────────────────────────
    row_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{parquet_path}')"
    ).fetchone()[0]
    result["row_count"] = row_count

    # ── Schema diff ───────────────────────────────────────────────────────────
    actual_schema = pq.read_schema(parquet_path)
    mismatches: list[SchemaMismatch] = schema_diff(
        table, actual_schema, strict_extra=False
    )
    result["schema_mismatches"] = [
        {"kind": m.kind, "column": m.column, "expected": m.expected, "actual": m.actual}
        for m in mismatches
    ]
    if mismatches:
        result["pass"] = False
        result["errors"].extend(str(m) for m in mismatches)

    # ── Actual columns present ────────────────────────────────────────────────
    result["columns"] = [
        {"name": f.name, "type": str(f.type)} for f in actual_schema
    ]

    # ── Null rates ────────────────────────────────────────────────────────────
    col_names = [f.name for f in actual_schema]
    null_rate_sql = ", ".join(
        f"round(count(*) filter (where \"{c}\" IS NULL) * 1.0 / count(*), 4) AS \"{c}\""
        for c in col_names
    )
    null_rates_row = con.execute(
        f"SELECT {null_rate_sql} FROM read_parquet('{parquet_path}')"
    ).fetchone()
    result["null_rates"] = dict(zip(col_names, null_rates_row))

    high_null = {c: v for c, v in result["null_rates"].items() if v > 0.5}
    if high_null:
        result["warnings"].append(f"High null rate (>50%) in: {list(high_null.keys())}")

    # ── Timestamp range ───────────────────────────────────────────────────────
    ts_cols = [c for c in col_names if c in TIMESTAMP_COLUMNS]
    for ts_col in ts_cols:
        row = con.execute(
            f"SELECT min(\"{ts_col}\"), max(\"{ts_col}\") "
            f"FROM read_parquet('{parquet_path}')"
        ).fetchone()
        ts_min, ts_max = row
        result[f"{ts_col}_min"] = ts_min
        result[f"{ts_col}_max"] = ts_max
        if ts_min is not None and ts_max is not None:
            result[f"{ts_col}_span_days"] = round((ts_max - ts_min) / 86400, 1)

    return result


def run(trace: str, out_path: Path | None = None) -> dict:
    raw_dir = RAW_DIR / trace
    if not raw_dir.exists():
        print(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = LOG_DIR / f"validation_{trace}.json"

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    report: dict = {
        "trace": trace,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": [],
        "overall_pass": True,
    }

    for table_name in SCHEMAS:
        parquet_path = raw_dir / f"{table_name}.parquet"
        if not parquet_path.exists():
            entry = {
                "table": table_name,
                "pass": False,
                "errors": [f"File not found: {parquet_path}"],
            }
            report["tables"].append(entry)
            report["overall_pass"] = False
            print(f"  MISSING  {table_name}.parquet")
            continue

        print(f"  Validating {table_name}.parquet …", end=" ", flush=True)
        entry = _validate_table(table_name, parquet_path, con)
        report["tables"].append(entry)
        if not entry["pass"]:
            report["overall_pass"] = False
        status = "PASS" if entry["pass"] else "FAIL"
        rows = f"{entry.get('row_count', '?'):,}"
        size = entry.get("file_size_human", "?")
        print(f"{status}  ({rows} rows, {size})")
        for err in entry.get("errors", []):
            print(f"    ERROR:   {err}")
        for warn in entry.get("warnings", []):
            print(f"    WARNING: {warn}")

    out_path.write_text(json.dumps(report, indent=2))
    overall = "PASS" if report["overall_pass"] else "FAIL"
    print(f"\nOverall: {overall}")
    print(f"Report:  {out_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate ingested Parquet files for a trace version."
    )
    parser.add_argument(
        "--trace",
        choices=["v2026", "v2023"],
        required=True,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path for the JSON report (default: outputs/logs/validation_{trace}.json).",
    )
    args = parser.parse_args(argv)
    run(args.trace, args.out)


if __name__ == "__main__":
    main()
