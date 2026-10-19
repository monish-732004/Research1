"""Tests for Stage C — left/right truncation flagging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import TRACE_START_HOUR, TRACE_END_HOUR


# ── SQL for pod lifecycle summary ─────────────────────────────────────────────
# Mirrors the logic in truncation.py without the file I/O.

def _flags_sql(pod_src: Path) -> str:
    return f"""
    WITH summary AS (
        SELECT
            pod_uid,
            min(rel_hour)          AS first_rel_hour,
            max(rel_hour)          AS last_rel_hour,
            arg_min(state, rel_hour) AS first_state,
            arg_max(state, rel_hour) AS last_state
        FROM read_parquet('{pod_src}')
        GROUP BY pod_uid
    )
    SELECT
        pod_uid,
        (first_rel_hour = {TRACE_START_HOUR} AND first_state = 'RUNNING') AS left_truncated,
        (last_rel_hour  = {TRACE_END_HOUR}   AND last_state  = 'RUNNING') AS right_truncated,
        first_rel_hour,
        last_rel_hour,
        last_state AS terminal_state
    FROM summary
    ORDER BY pod_uid
    """


def _write_pods(rows: list[dict], path: Path) -> None:
    pq.write_table(pa.table({
        "pod_uid":       pa.array([r["pod_uid"] for r in rows]),
        "day":           pa.array([r["rel_hour"] // 24 for r in rows], type=pa.int32()),
        "hour":          pa.array([r["rel_hour"] % 24 for r in rows], type=pa.int32()),
        "rel_hour":      pa.array([r["rel_hour"] for r in rows], type=pa.int32()),
        "gpu_requested": pa.array([r.get("gpu_requested", 4) for r in rows], type=pa.int32()),
        "gpu_util":      pa.array([r.get("gpu_util", 0.5) for r in rows], type=pa.float32()),
        "priority":      pa.array([r.get("priority", 1) for r in rows], type=pa.int32()),
        "state":         pa.array([r["state"] for r in rows]),
        "server_uid":    pa.array([None] * len(rows), type=pa.string()),
    }), path)


# ── Known-result synthetic traces ─────────────────────────────────────────────

def _build_test_pods() -> list[dict]:
    rows = []
    # LEFT-TRUNCATED pods (2): appear at rel_hour=0 with state=RUNNING
    for pid in ["left_A", "left_B"]:
        for h in range(0, 50):
            state = "RUNNING" if h < 48 else "SUCCEEDED"
            rows.append({"pod_uid": pid, "rel_hour": h, "state": state})

    # RIGHT-TRUNCATED pods (3): last hour is TRACE_END_HOUR with state=RUNNING
    for pid in ["right_A", "right_B", "right_C"]:
        start = TRACE_END_HOUR - 30
        for h in range(start, TRACE_END_HOUR + 1):
            rows.append({"pod_uid": pid, "rel_hour": h, "state": "RUNNING"})

    # NORMAL pods: neither truncated
    # Pod that starts at 0 but with PENDING (not left-truncated)
    for h in range(0, 20):
        state = "PENDING" if h < 2 else ("RUNNING" if h < 18 else "SUCCEEDED")
        rows.append({"pod_uid": "normal_pending_start", "rel_hour": h, "state": state})

    # Pod that ends at TRACE_END_HOUR with SUCCEEDED (not right-truncated)
    for h in range(TRACE_END_HOUR - 5, TRACE_END_HOUR + 1):
        state = "RUNNING" if h < TRACE_END_HOUR else "SUCCEEDED"
        rows.append({"pod_uid": "normal_succeeded_end", "rel_hour": h, "state": state})

    # Entirely interior pod
    for h in range(100, 200):
        state = "RUNNING" if h < 198 else "FAILED"
        rows.append({"pod_uid": "interior", "rel_hour": h, "state": state})

    return rows


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTruncationFlags:
    @pytest.fixture
    def flags_df(self, tmp_path):
        path = tmp_path / "pod_canonical.parquet"
        _write_pods(_build_test_pods(), path)
        con = duckdb.connect()
        return con.execute(_flags_sql(path)).df().set_index("pod_uid")

    def test_two_left_truncated_pods_identified(self, flags_df):
        n_left = flags_df["left_truncated"].sum()
        assert n_left == 2, f"Expected 2 left-truncated, got {n_left}"
        assert flags_df.loc["left_A", "left_truncated"]
        assert flags_df.loc["left_B", "left_truncated"]

    def test_three_right_truncated_pods_identified(self, flags_df):
        n_right = flags_df["right_truncated"].sum()
        assert n_right == 3, f"Expected 3 right-truncated, got {n_right}"
        for pid in ["right_A", "right_B", "right_C"]:
            assert flags_df.loc[pid, "right_truncated"]

    def test_pending_start_not_left_truncated(self, flags_df):
        """Pod starting at hour 0 with PENDING is NOT left-truncated."""
        assert not flags_df.loc["normal_pending_start", "left_truncated"]

    def test_succeeded_at_end_not_right_truncated(self, flags_df):
        """Pod ending at TRACE_END_HOUR with SUCCEEDED is NOT right-truncated."""
        assert not flags_df.loc["normal_succeeded_end", "right_truncated"]

    def test_interior_pod_not_truncated(self, flags_df):
        assert not flags_df.loc["interior", "left_truncated"]
        assert not flags_df.loc["interior", "right_truncated"]

    def test_first_last_rel_hour_correct(self, flags_df):
        assert flags_df.loc["left_A", "first_rel_hour"] == TRACE_START_HOUR
        for pid in ["right_A", "right_B", "right_C"]:
            assert flags_df.loc[pid, "last_rel_hour"] == TRACE_END_HOUR

    def test_terminal_state_recorded(self, flags_df):
        assert flags_df.loc["interior", "terminal_state"] == "FAILED"
        assert flags_df.loc["normal_succeeded_end", "terminal_state"] == "SUCCEEDED"


class TestSurvivalTable:
    def _survival_sql(self, flags_path: Path) -> str:
        return f"""
        SELECT pod_uid,
               (last_rel_hour - first_rel_hour + 1) AS duration_hours_observed,
               right_truncated AS is_censored
        FROM read_parquet('{flags_path}')
        """

    def test_right_truncated_is_censored(self, tmp_path):
        path = tmp_path / "pod.parquet"
        _write_pods(_build_test_pods(), path)
        # Build flags table first
        flags_path = tmp_path / "flags.parquet"
        con = duckdb.connect()
        con.execute(f"COPY ({_flags_sql(path)}) TO '{flags_path}' (FORMAT PARQUET)")

        surv = con.execute(self._survival_sql(flags_path)).df().set_index("pod_uid")
        for pid in ["right_A", "right_B", "right_C"]:
            assert surv.loc[pid, "is_censored"]

    def test_interior_pod_not_censored(self, tmp_path):
        path = tmp_path / "pod.parquet"
        _write_pods(_build_test_pods(), path)
        flags_path = tmp_path / "flags.parquet"
        con = duckdb.connect()
        con.execute(f"COPY ({_flags_sql(path)}) TO '{flags_path}' (FORMAT PARQUET)")

        surv = con.execute(self._survival_sql(flags_path)).df().set_index("pod_uid")
        assert not surv.loc["interior", "is_censored"]
        assert surv.loc["interior", "duration_hours_observed"] == 100
