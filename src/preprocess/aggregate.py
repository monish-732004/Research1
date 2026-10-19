"""Stage D — Hourly fragmentation aggregation.

Computes per-(rel_hour, server_tier) GPU fragmentation metrics:

  gpu_used_est      = gpu_requested × gpu_util
  frag_gpu_hours    = gpu_requested − gpu_used_est

Server tier is derived from the server's total GPU count (if a
pod→server mapping exists in pod_hourly_canonical.server_uid):
  small  : gpu_count_total ≤ SERVER_TIER_SMALL_MAX   (default ≤ 4)
  medium : gpu_count_total ≤ SERVER_TIER_MEDIUM_MAX  (default ≤ 8)
  large  : gpu_count_total >  SERVER_TIER_MEDIUM_MAX

Output
------
  data/processed/hourly_frag.parquet
    rel_hour, server_tier,
    total_gpus_requested, total_gpus_used, total_frag_gpu_hours, pod_count

Usage
-----
    python -m src.preprocess.aggregate --trace v2026
    python -m src.preprocess.aggregate --trace v2026 --force
"""

from __future__ import annotations

import argparse
from typing import Any

import duckdb

from src.config import (
    INTERIM_DIR, PROCESSED_DIR,
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS,
    SERVER_TIER_SMALL_MAX, SERVER_TIER_MEDIUM_MAX,
)


def run(trace: str, *, force: bool = False) -> dict[str, Any]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "hourly_frag.parquet"

    if out.exists() and not force:
        rows = duckdb.execute(
            f"SELECT count(*) FROM read_parquet('{out}')"
        ).fetchone()[0]
        print(f"  [aggregate] output exists ({rows:,} rows) — skipping")
        return {"rows_out": rows, "skipped": True}

    pod_src    = INTERIM_DIR / "pod_hourly_canonical.parquet"
    server_src = INTERIM_DIR / "server_hourly_canonical.parquet"

    if not pod_src.exists():
        raise FileNotFoundError(f"{pod_src} not found. Run canonicalize first.")

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")

    # Detect whether pod table carries a server_uid mapping
    pod_cols = {
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{pod_src}')"
        ).fetchall()
    }
    has_server_col  = "server_uid" in pod_cols
    has_server_file = server_src.exists()
    use_server_join = has_server_col and has_server_file

    if use_server_join:
        server_join = f"""
        LEFT JOIN read_parquet('{server_src}') s
          ON p.server_uid = s.server_uid
         AND p.day = s.day
         AND p.hour = s.hour
        """
        tier_expr = f"""
        CASE
            WHEN s.gpu_count_total IS NULL          THEN 'unknown'
            WHEN s.gpu_count_total <= {SERVER_TIER_SMALL_MAX}  THEN 'small'
            WHEN s.gpu_count_total <= {SERVER_TIER_MEDIUM_MAX} THEN 'medium'
            ELSE 'large'
        END
        """
    else:
        server_join = ""
        tier_expr   = "'unknown'"

    sql = f"""
    WITH pod_frag AS (
        SELECT
            p.rel_hour,
            p.gpu_requested,
            COALESCE(p.gpu_requested * p.gpu_util, 0.0)           AS gpu_used_est,
            COALESCE(p.gpu_requested * (1.0 - p.gpu_util), p.gpu_requested::FLOAT) AS frag_gpu_hours,
            {tier_expr} AS server_tier
        FROM read_parquet('{pod_src}') p
        {server_join}
        WHERE p.state = 'RUNNING'
          AND p.gpu_requested IS NOT NULL
          AND p.gpu_requested > 0
    )
    SELECT
        rel_hour,
        server_tier,
        sum(gpu_requested)     AS total_gpus_requested,
        sum(gpu_used_est)      AS total_gpus_used,
        sum(frag_gpu_hours)    AS total_frag_gpu_hours,
        count(*)               AS pod_count
    FROM pod_frag
    GROUP BY rel_hour, server_tier
    ORDER BY rel_hour, server_tier
    """

    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT PARQUET)")
    n_rows = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]

    total_frag = con.execute(
        f"SELECT sum(total_frag_gpu_hours) FROM read_parquet('{out}')"
    ).fetchone()[0] or 0.0

    join_note = "with server join" if use_server_join else "pod-level only (no server mapping)"
    print(
        f"  [aggregate] {n_rows:,} (rel_hour × tier) rows  "
        f"| total frag: {total_frag:,.0f} GPU-hours  [{join_note}]"
    )
    return {"rows_out": n_rows, "skipped": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute hourly GPU fragmentation.")
    parser.add_argument("--trace", choices=["v2026", "v2023"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    run(args.trace, force=args.force)


if __name__ == "__main__":
    main()
