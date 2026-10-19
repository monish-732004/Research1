"""Tests for Stage A — schema canonicalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.preprocess.canonicalize import (
    _actual_columns, _discover_mapping, _build_pod_select,
    _POD_ALIASES, _SERVER_ALIASES,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _write_pod_parquet(
    columns: dict[str, pa.Array],
    path: Path,
) -> None:
    table = pa.table(columns)
    pq.write_table(table, path)


def _basic_pod_columns(n: int = 20) -> dict[str, pa.Array]:
    """Synthetic pod data with non-canonical column names (like a real trace)."""
    return {
        "pod_id":       pa.array([f"p{i:04d}" for i in range(n)]),
        "day":          pa.array([i // 24 for i in range(n)], type=pa.int32()),
        "hour":         pa.array([i % 24 for i in range(n)], type=pa.int32()),
        "gpu_num":      pa.array([4] * n, type=pa.int32()),
        "avg_gpu_util": pa.array([0.5] * n, type=pa.float32()),
        "priority":     pa.array([1] * n, type=pa.int32()),
        "status":       pa.array(["RUNNING"] * n),
    }


def _percent_util_columns(n: int = 10) -> dict[str, pa.Array]:
    """gpu_util expressed as 0–100 (should be normalised to 0–1)."""
    cols = _basic_pod_columns(n)
    cols["avg_gpu_util"] = pa.array([50.0] * n, type=pa.float32())
    return cols


# ── Tests: alias resolution ───────────────────────────────────────────────────

class TestAliasResolution:
    def test_canonical_names_resolve_from_aliases(self, tmp_path):
        path = tmp_path / "pod.parquet"
        _write_pod_parquet(_basic_pod_columns(), path)
        actual = _actual_columns(path)
        mapping = _discover_mapping(actual, _POD_ALIASES)

        assert mapping["pod_uid"] == "pod_id"
        assert mapping["day"] == "day"
        assert mapping["hour"] == "hour"
        assert mapping["gpu_requested"] == "gpu_num"
        assert mapping["gpu_util"] == "avg_gpu_util"
        assert mapping["priority"] == "priority"
        assert mapping["state"] == "status"

    def test_canonical_name_used_as_fallback(self, tmp_path):
        """If the actual column name IS the canonical name, it should still resolve."""
        path = tmp_path / "pod.parquet"
        n = 5
        cols = {
            "pod_uid":       pa.array([f"p{i}" for i in range(n)]),
            "day":           pa.array([0] * n, type=pa.int32()),
            "hour":          pa.array(list(range(n)), type=pa.int32()),
            "gpu_requested": pa.array([2] * n, type=pa.int32()),
            "gpu_util":      pa.array([0.3] * n, type=pa.float32()),
            "priority":      pa.array([0] * n, type=pa.int32()),
            "state":         pa.array(["PENDING"] * n),
        }
        _write_pod_parquet(cols, path)
        actual = _actual_columns(path)
        mapping = _discover_mapping(actual, _POD_ALIASES)
        assert mapping["pod_uid"] == "pod_uid"
        assert mapping["gpu_requested"] == "gpu_requested"

    def test_missing_optional_column_maps_to_none(self, tmp_path):
        path = tmp_path / "pod.parquet"
        _write_pod_parquet(_basic_pod_columns(), path)
        actual = _actual_columns(path)
        mapping = _discover_mapping(actual, _POD_ALIASES)
        # server_uid is not in basic columns → None
        assert mapping["server_uid"] is None


# ── Tests: SELECT query generation ───────────────────────────────────────────

class TestSelectGeneration:
    def test_output_has_canonical_columns(self, tmp_path):
        pod_src = tmp_path / "pod_raw.parquet"
        pod_out = tmp_path / "pod_canonical.parquet"
        _write_pod_parquet(_basic_pod_columns(), pod_src)

        actual = _actual_columns(pod_src)
        mapping = _discover_mapping(actual, _POD_ALIASES)

        con = duckdb.connect()
        sql = _build_pod_select(pod_src, mapping, con)
        con.execute(f"COPY ({sql}) TO '{pod_out}' (FORMAT PARQUET)")

        result = pq.read_table(pod_out)
        assert "pod_uid"       in result.column_names
        assert "day"           in result.column_names
        assert "hour"          in result.column_names
        assert "rel_hour"      in result.column_names
        assert "gpu_requested" in result.column_names
        assert "gpu_util"      in result.column_names
        assert "priority"      in result.column_names
        assert "state"         in result.column_names

    def test_rel_hour_is_day_times_24_plus_hour(self, tmp_path):
        pod_src = tmp_path / "pod_raw.parquet"
        pod_out = tmp_path / "pod_canonical.parquet"

        n = 5
        cols = {
            "pod_uid":       pa.array([f"p{i}" for i in range(n)]),
            "day":           pa.array([3, 3, 7, 7, 10], type=pa.int32()),
            "hour":          pa.array([0, 1, 0, 23, 5], type=pa.int32()),
            "gpu_num":       pa.array([4] * n, type=pa.int32()),
            "avg_gpu_util":  pa.array([0.5] * n, type=pa.float32()),
            "priority":      pa.array([0] * n, type=pa.int32()),
            "status":        pa.array(["RUNNING"] * n),
        }
        _write_pod_parquet(cols, pod_src)

        actual = _actual_columns(pod_src)
        mapping = _discover_mapping(actual, _POD_ALIASES)

        con = duckdb.connect()
        sql = _build_pod_select(pod_src, mapping, con)
        result_df = con.execute(sql).df()

        expected_rel_hours = [3*24+0, 3*24+1, 7*24+0, 7*24+23, 10*24+5]
        assert list(result_df["rel_hour"]) == expected_rel_hours

    def test_gpu_util_normalised_from_percent(self, tmp_path):
        """If source stores 0–100, the output should be 0.0–1.0."""
        pod_src = tmp_path / "pod_pct.parquet"
        pod_out = tmp_path / "pod_can.parquet"
        _write_pod_parquet(_percent_util_columns(), pod_src)

        actual = _actual_columns(pod_src)
        mapping = _discover_mapping(actual, _POD_ALIASES)

        con = duckdb.connect()
        sql = _build_pod_select(pod_src, mapping, con)
        result_df = con.execute(sql).df()

        assert all(0.0 <= v <= 1.0 for v in result_df["gpu_util"].dropna())

    def test_state_uppercased(self, tmp_path):
        """State values should be uppercased in the canonical table."""
        pod_src = tmp_path / "pod_lower.parquet"
        n = 3
        cols = {
            "pod_uid":  pa.array([f"p{i}" for i in range(n)]),
            "day":      pa.array([0] * n, type=pa.int32()),
            "hour":     pa.array(list(range(n)), type=pa.int32()),
            "gpu_num":  pa.array([2] * n, type=pa.int32()),
            "avg_gpu_util": pa.array([0.6] * n, type=pa.float32()),
            "priority": pa.array([0] * n, type=pa.int32()),
            "status":   pa.array(["running", "pending", "failed"]),
        }
        _write_pod_parquet(cols, pod_src)

        actual = _actual_columns(pod_src)
        mapping = _discover_mapping(actual, _POD_ALIASES)

        con = duckdb.connect()
        sql = _build_pod_select(pod_src, mapping, con)
        result_df = con.execute(sql).df()

        assert set(result_df["state"]) == {"RUNNING", "PENDING", "FAILED"}

    def test_missing_gpu_util_column_yields_null(self, tmp_path):
        """A source with no gpu_util column should produce NULLs, not crash."""
        pod_src = tmp_path / "pod_no_util.parquet"
        n = 4
        cols = {
            "pod_uid":  pa.array([f"p{i}" for i in range(n)]),
            "day":      pa.array([0] * n, type=pa.int32()),
            "hour":     pa.array(list(range(n)), type=pa.int32()),
            "gpu_num":  pa.array([8] * n, type=pa.int32()),
            "priority": pa.array([1] * n, type=pa.int32()),
            "status":   pa.array(["RUNNING"] * n),
            # no gpu_util column
        }
        _write_pod_parquet(cols, pod_src)

        actual = _actual_columns(pod_src)
        mapping = _discover_mapping(actual, _POD_ALIASES)
        assert mapping["gpu_util"] is None

        con = duckdb.connect()
        sql = _build_pod_select(pod_src, mapping, con)
        result_df = con.execute(sql).df()

        assert result_df["gpu_util"].isna().all()
