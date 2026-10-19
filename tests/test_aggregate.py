"""Tests for Stage D — hourly fragmentation aggregation arithmetic."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import SERVER_TIER_SMALL_MAX, SERVER_TIER_MEDIUM_MAX


# ── Fragmentation SQL (mirrors aggregate.py, without file I/O) ────────────────

def _frag_sql(pod_src: Path) -> str:
    """Pod-level aggregation (no server join)."""
    return f"""
    WITH pod_frag AS (
        SELECT
            p.rel_hour,
            p.gpu_requested,
            COALESCE(p.gpu_requested * p.gpu_util, 0.0)                          AS gpu_used_est,
            COALESCE(p.gpu_requested * (1.0 - p.gpu_util), p.gpu_requested::FLOAT) AS frag_gpu_hours,
            'unknown' AS server_tier
        FROM read_parquet('{pod_src}') p
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
    ORDER BY rel_hour
    """


def _frag_sql_with_server(pod_src: Path, server_src: Path) -> str:
    return f"""
    WITH pod_frag AS (
        SELECT
            p.rel_hour,
            p.gpu_requested,
            COALESCE(p.gpu_requested * p.gpu_util, 0.0)                          AS gpu_used_est,
            COALESCE(p.gpu_requested * (1.0 - p.gpu_util), p.gpu_requested::FLOAT) AS frag_gpu_hours,
            CASE
                WHEN s.gpu_count_total IS NULL              THEN 'unknown'
                WHEN s.gpu_count_total <= {SERVER_TIER_SMALL_MAX}  THEN 'small'
                WHEN s.gpu_count_total <= {SERVER_TIER_MEDIUM_MAX} THEN 'medium'
                ELSE 'large'
            END AS server_tier
        FROM read_parquet('{pod_src}') p
        LEFT JOIN read_parquet('{server_src}') s
          ON p.server_uid = s.server_uid AND p.day = s.day AND p.hour = s.hour
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


def _pod_table(rows: list[dict]) -> pa.Table:
    n = len(rows)
    return pa.table({
        "pod_uid":       pa.array([r["pod_uid"]          for r in rows]),
        "day":           pa.array([r["rel_hour"] // 24   for r in rows], type=pa.int32()),
        "hour":          pa.array([r["rel_hour"] % 24    for r in rows], type=pa.int32()),
        "rel_hour":      pa.array([r["rel_hour"]         for r in rows], type=pa.int32()),
        "gpu_requested": pa.array([r["gpu_requested"]    for r in rows], type=pa.int32()),
        "gpu_util":      pa.array([r["gpu_util"]         for r in rows], type=pa.float32()),
        "priority":      pa.array([r.get("priority", 1) for r in rows], type=pa.int32()),
        "state":         pa.array([r.get("state", "RUNNING") for r in rows]),
        "server_uid":    pa.array([r.get("server_uid", None) for r in rows], type=pa.string()),
    })


def _server_table(rows: list[dict]) -> pa.Table:
    return pa.table({
        "server_uid":      pa.array([r["server_uid"]      for r in rows]),
        "day":             pa.array([r["day"]              for r in rows], type=pa.int32()),
        "hour":            pa.array([r["hour"]             for r in rows], type=pa.int32()),
        "rel_hour":        pa.array([r["day"]*24+r["hour"] for r in rows], type=pa.int32()),
        "gpu_count_total": pa.array([r["gpu_count_total"]  for r in rows], type=pa.int32()),
        "cpu_count_total": pa.array([r.get("cpu_count_total", 32) for r in rows], type=pa.int32()),
        "asw_id":          pa.array([r.get("asw_id", "az1") for r in rows]),
        "rack_id":         pa.array([r.get("rack_id", "r1") for r in rows]),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFragmentationArithmetic:
    def test_basic_two_pods_one_hour(self, tmp_path):
        """
        Pod A: gpu_requested=4, gpu_util=0.5  → frag = 4 * 0.5 = 2.0
        Pod B: gpu_requested=2, gpu_util=0.75 → frag = 2 * 0.25 = 0.5
        Total frag = 2.5, total requested = 6, total used ≈ 3.5
        """
        rows = [
            {"pod_uid": "A", "rel_hour": 0, "gpu_requested": 4, "gpu_util": 0.5},
            {"pod_uid": "B", "rel_hour": 0, "gpu_requested": 2, "gpu_util": 0.75},
        ]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_table(rows), path)

        con = duckdb.connect()
        result = con.execute(_frag_sql(path)).df()
        assert len(result) == 1
        row = result.iloc[0]
        assert row["total_gpus_requested"] == 6
        assert math.isclose(row["total_gpus_used"], 3.5, rel_tol=1e-5)
        assert math.isclose(row["total_frag_gpu_hours"], 2.5, rel_tol=1e-5)
        assert row["pod_count"] == 2

    def test_zero_util_means_full_fragmentation(self, tmp_path):
        rows = [{"pod_uid": "p", "rel_hour": 0, "gpu_requested": 8, "gpu_util": 0.0}]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_table(rows), path)

        con = duckdb.connect()
        result = con.execute(_frag_sql(path)).df()
        row = result.iloc[0]
        assert math.isclose(row["total_frag_gpu_hours"], 8.0, rel_tol=1e-5)
        assert math.isclose(row["total_gpus_used"], 0.0, abs_tol=1e-5)

    def test_full_util_means_zero_fragmentation(self, tmp_path):
        rows = [{"pod_uid": "p", "rel_hour": 0, "gpu_requested": 4, "gpu_util": 1.0}]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_table(rows), path)

        con = duckdb.connect()
        result = con.execute(_frag_sql(path)).df()
        row = result.iloc[0]
        assert math.isclose(row["total_frag_gpu_hours"], 0.0, abs_tol=1e-5)

    def test_non_running_pods_excluded(self, tmp_path):
        rows = [
            {"pod_uid": "r", "rel_hour": 0, "gpu_requested": 4, "gpu_util": 0.5, "state": "RUNNING"},
            {"pod_uid": "p", "rel_hour": 0, "gpu_requested": 8, "gpu_util": 0.0, "state": "PENDING"},
            {"pod_uid": "s", "rel_hour": 0, "gpu_requested": 4, "gpu_util": 1.0, "state": "SUCCEEDED"},
        ]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_table(rows), path)

        con = duckdb.connect()
        result = con.execute(_frag_sql(path)).df()
        assert result.iloc[0]["pod_count"] == 1     # only RUNNING counted
        assert result.iloc[0]["total_gpus_requested"] == 4

    def test_multiple_hours_aggregated_independently(self, tmp_path):
        rows = [
            {"pod_uid": "p1", "rel_hour": 0, "gpu_requested": 4, "gpu_util": 0.25},
            {"pod_uid": "p1", "rel_hour": 1, "gpu_requested": 4, "gpu_util": 0.75},
            {"pod_uid": "p1", "rel_hour": 2, "gpu_requested": 4, "gpu_util": 0.50},
        ]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_table(rows), path)

        con = duckdb.connect()
        result = con.execute(_frag_sql(path)).df().set_index("rel_hour")
        assert math.isclose(result.loc[0, "total_frag_gpu_hours"], 3.0, rel_tol=1e-5)
        assert math.isclose(result.loc[1, "total_frag_gpu_hours"], 1.0, rel_tol=1e-5)
        assert math.isclose(result.loc[2, "total_frag_gpu_hours"], 2.0, rel_tol=1e-5)


class TestServerTierAssignment:
    def test_tier_classification_via_sql(self, tmp_path):
        """Server tiers small/medium/large assigned based on gpu_count_total."""
        pod_rows = [
            {"pod_uid": f"p{i}", "rel_hour": 0, "gpu_requested": 2,
             "gpu_util": 0.5, "server_uid": f"s{i}"}
            for i in range(3)
        ]
        server_rows = [
            {"server_uid": "s0", "day": 0, "hour": 0, "gpu_count_total": 4},   # small
            {"server_uid": "s1", "day": 0, "hour": 0, "gpu_count_total": 8},   # medium
            {"server_uid": "s2", "day": 0, "hour": 0, "gpu_count_total": 16},  # large
        ]
        pod_path = tmp_path / "pod.parquet"
        srv_path = tmp_path / "srv.parquet"
        pq.write_table(_pod_table(pod_rows), pod_path)
        pq.write_table(_server_table(server_rows), srv_path)

        con = duckdb.connect()
        result = con.execute(_frag_sql_with_server(pod_path, srv_path)).df()
        tiers = set(result["server_tier"].tolist())
        assert tiers == {"small", "medium", "large"}

    def test_small_boundary_exact(self, tmp_path):
        """gpu_count_total == SERVER_TIER_SMALL_MAX should map to 'small'."""
        pod_rows = [{"pod_uid": "p", "rel_hour": 0, "gpu_requested": 2,
                     "gpu_util": 0.5, "server_uid": "s0"}]
        server_rows = [{"server_uid": "s0", "day": 0, "hour": 0,
                        "gpu_count_total": SERVER_TIER_SMALL_MAX}]

        pod_path = tmp_path / "pod.parquet"
        srv_path = tmp_path / "srv.parquet"
        pq.write_table(_pod_table(pod_rows), pod_path)
        pq.write_table(_server_table(server_rows), srv_path)

        con = duckdb.connect()
        result = con.execute(_frag_sql_with_server(pod_path, srv_path)).df()
        assert result.iloc[0]["server_tier"] == "small"

    def test_medium_boundary_exact(self, tmp_path):
        pod_rows = [{"pod_uid": "p", "rel_hour": 0, "gpu_requested": 2,
                     "gpu_util": 0.5, "server_uid": "s0"}]
        server_rows = [{"server_uid": "s0", "day": 0, "hour": 0,
                        "gpu_count_total": SERVER_TIER_MEDIUM_MAX}]

        pod_path = tmp_path / "pod.parquet"
        srv_path = tmp_path / "srv.parquet"
        pq.write_table(_pod_table(pod_rows), pod_path)
        pq.write_table(_server_table(server_rows), srv_path)

        con = duckdb.connect()
        result = con.execute(_frag_sql_with_server(pod_path, srv_path)).df()
        assert result.iloc[0]["server_tier"] == "medium"
