"""Tests for Stage B — partition completeness and gap detection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# We test the SQL logic directly without calling run() (which needs the full
# interim directory layout).  The core logic is the mask query — we reproduce
# it against synthetic data.


def _pod_hourly(rows: list[dict]) -> pa.Table:
    return pa.table({
        "pod_uid":      pa.array([r["pod_uid"] for r in rows]),
        "day":          pa.array([r["day"] for r in rows], type=pa.int32()),
        "hour":         pa.array([r["hour"] for r in rows], type=pa.int32()),
        "rel_hour":     pa.array([r["rel_hour"] for r in rows], type=pa.int32()),
        "gpu_requested":pa.array([r.get("gpu_requested", 4) for r in rows], type=pa.int32()),
        "gpu_util":     pa.array([r.get("gpu_util", 0.5) for r in rows], type=pa.float32()),
        "priority":     pa.array([r.get("priority", 1) for r in rows], type=pa.int32()),
        "state":        pa.array([r.get("state", "RUNNING") for r in rows]),
        "server_uid":   pa.array([r.get("server_uid", None) for r in rows]),
    })


def _gap_mask_sql(pod_src: Path) -> str:
    return f"""
    WITH active AS (
        SELECT pod_uid, rel_hour
        FROM read_parquet('{pod_src}')
        WHERE state IN ('PENDING', 'RUNNING')
    ),
    pod_windows AS (
        SELECT pod_uid, min(rel_hour) AS first_hour, max(rel_hour) AS last_hour
        FROM active GROUP BY pod_uid
    ),
    expected AS (
        SELECT p.pod_uid, h.rel_hour
        FROM pod_windows p,
             LATERAL (
                 SELECT unnest(range(p.first_hour, p.last_hour + 1)) AS rel_hour
             ) h
    ),
    actual_distinct AS (
        SELECT DISTINCT pod_uid, rel_hour FROM active
    )
    SELECT
        e.pod_uid,
        e.rel_hour,
        (a.pod_uid IS NOT NULL) AS is_present,
        (a.pod_uid IS NULL)     AS is_gap_filled
    FROM expected e
    LEFT JOIN actual_distinct a USING (pod_uid, rel_hour)
    ORDER BY e.pod_uid, e.rel_hour
    """


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPartitionCompleteness:
    def test_complete_pod_no_gaps(self, tmp_path):
        """A pod present at every hour has no gaps."""
        rows = [{"pod_uid": "p0", "day": h // 24, "hour": h % 24, "rel_hour": h}
                for h in range(10)]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()

        assert len(result) == 10
        assert result["is_present"].all()
        assert not result["is_gap_filled"].any()

    def test_single_gap_detected(self, tmp_path):
        """Pod disappears at hour 5, reappears at hour 6 → gap at hour 5."""
        hours = list(range(0, 5)) + list(range(6, 10))   # gap at 5
        rows = [{"pod_uid": "p0", "day": h // 24, "hour": h % 24, "rel_hour": h}
                for h in hours]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()

        assert len(result) == 10   # 9 actual + 1 generated gap
        gap_rows = result[result["rel_hour"] == 5]
        assert len(gap_rows) == 1
        assert not gap_rows.iloc[0]["is_present"]
        assert gap_rows.iloc[0]["is_gap_filled"]

    def test_three_known_gaps(self, tmp_path):
        """Build a trace with 3 holes and confirm all three are detected."""
        base_hours = list(range(20))
        holes = {3, 7, 14}
        hours = [h for h in base_hours if h not in holes]
        rows = [{"pod_uid": "pod_with_gaps", "day": h // 24, "hour": h % 24, "rel_hour": h}
                for h in hours]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()

        gap_rows = result[~result["is_present"]]
        detected_gaps = set(gap_rows["rel_hour"].astype(int).tolist())
        assert detected_gaps == holes, f"Expected gaps at {holes}, got {detected_gaps}"

    def test_multiple_pods_independent(self, tmp_path):
        """Gaps in one pod do not affect another pod."""
        rows = (
            [{"pod_uid": "clean",  "day": h//24, "hour": h%24, "rel_hour": h}
             for h in range(5)]
            + [{"pod_uid": "gappy", "day": h//24, "hour": h%24, "rel_hour": h}
               for h in [0, 1, 3, 4]]   # gap at hour 2
        )
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()

        clean_gaps = result[(result["pod_uid"] == "clean") & (~result["is_present"])]
        gappy_gaps = result[(result["pod_uid"] == "gappy") & (~result["is_present"])]

        assert len(clean_gaps) == 0
        assert len(gappy_gaps) == 1
        assert gappy_gaps.iloc[0]["rel_hour"] == 2

    def test_pending_state_included_in_active_window(self, tmp_path):
        """PENDING pods are included in the activity window."""
        rows = [
            {"pod_uid": "p0", "day": 0, "hour": 0, "rel_hour": 0, "state": "PENDING"},
            {"pod_uid": "p0", "day": 0, "hour": 1, "rel_hour": 1, "state": "RUNNING"},
            {"pod_uid": "p0", "day": 0, "hour": 2, "rel_hour": 2, "state": "RUNNING"},
        ]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()
        assert len(result) == 3
        assert result["is_present"].all()

    def test_succeeded_pod_not_extended(self, tmp_path):
        """SUCCEEDED hours are not considered active; the window stops at last active hour."""
        rows = [
            {"pod_uid": "p0", "day": 0, "hour": 0, "rel_hour": 0, "state": "RUNNING"},
            {"pod_uid": "p0", "day": 0, "hour": 1, "rel_hour": 1, "state": "RUNNING"},
            {"pod_uid": "p0", "day": 0, "hour": 2, "rel_hour": 2, "state": "SUCCEEDED"},
        ]
        path = tmp_path / "pod.parquet"
        pq.write_table(_pod_hourly(rows), path)

        con = duckdb.connect()
        result = con.execute(_gap_mask_sql(path)).df()
        # Activity window: hours 0–1 only (SUCCEEDED excluded)
        assert set(result["rel_hour"].astype(int).tolist()) == {0, 1}
