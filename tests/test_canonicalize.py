"""Tests for Phase 2 canonicalization logic.

Tests use DuckDB with synthetic in-memory data — no Parquet files needed.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.preprocess.tdp import get_tdp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


# ── 1. hour cast: VARCHAR "00"-"23" → INTEGER ─────────────────────────────────

class TestHourCast:
    def test_hour_cast(self):
        con = _con()
        con.execute("""
            CREATE TABLE t AS
            SELECT * FROM (VALUES
                ('s1', 3, '07'),
                ('s2', 7, '23')
            ) AS v(server_id, day, hour)
        """)
        df = con.execute("""
            SELECT
                day,
                CAST(hour AS INTEGER) AS hour_int,
                day * 24 + CAST(hour AS INTEGER) AS rel_hour
            FROM t
        """).df()

        assert list(df["hour_int"])  == [7, 23]
        assert list(df["rel_hour"])  == [3*24+7, 7*24+23]


# ── 2. server tier classification ─────────────────────────────────────────────

class TestServerTier:
    @pytest.mark.parametrize("gpu_count,expected_tier", [
        (1,  "small"),
        (2,  "small"),
        (3,  "medium"),
        (8,  "medium"),
        (9,  "large"),
        (16, "large"),
    ])
    def test_server_tier(self, gpu_count, expected_tier):
        con = _con()
        df = con.execute(f"""
            SELECT
                CASE
                    WHEN {gpu_count} <= 2 THEN 'small'
                    WHEN {gpu_count} <= 8 THEN 'medium'
                    ELSE 'large'
                END AS server_tier
        """).df()
        assert df["server_tier"].iloc[0] == expected_tier


# ── 3. gpu_hours_requested = gpu_request × duration_hours ─────────────────────

class TestGpuHours:
    def test_gpu_hours_calculation(self):
        con = _con()
        con.execute("""
            CREATE TABLE jobs AS
            SELECT * FROM (VALUES
                ('j1', 0.24, 10.0),
                ('j2', 1.0,  5.0),
                ('j3', 0.5,  0.0)
            ) AS v(pod_id, gpu_request, duration_hours)
        """)
        df = con.execute("""
            SELECT pod_id, gpu_request * duration_hours AS gpu_hours_requested
            FROM jobs
        """).df()

        assert abs(df.loc[df["pod_id"]=="j1","gpu_hours_requested"].iloc[0] - 2.4) < 1e-9
        assert abs(df.loc[df["pod_id"]=="j2","gpu_hours_requested"].iloc[0] - 5.0) < 1e-9
        assert abs(df.loc[df["pod_id"]=="j3","gpu_hours_requested"].iloc[0] - 0.0) < 1e-9


# ── 4. TDP lookup (exact, prefix, fallback) ───────────────────────────────────

class TestTdpLookup:
    def test_exact_match(self):
        assert get_tdp("A10")  == 150
        assert get_tdp("H100") == 700
        assert get_tdp("L20")  == 275

    def test_prefix_match(self):
        assert get_tdp("A100-80GB")  == 400   # prefix "A100"
        assert get_tdp("H800-80G")   == 700   # prefix "H800"

    def test_fallback_for_unknown(self):
        assert get_tdp("UNKNOWN_XYZ") == 350
        assert get_tdp("")             == 350
        assert get_tdp("unknown")      == 350


# ── 5. is_deferrable flag ─────────────────────────────────────────────────────

class TestIsDeferrable:
    def test_online_inference_never_deferrable(self):
        """online_inference jobs must not be deferred regardless of priority."""
        con = _con()
        con.execute("""
            CREATE TABLE jobs AS
            SELECT * FROM (VALUES
                ('j1', 'online_inference', 'LP', TRUE),
                ('j2', 'online_inference', 'HP', TRUE),
                ('j3', 'offline_inference', 'LP', TRUE),
                ('j4', 'batch',            'LP', TRUE),
                ('j5', 'batch',            'HP', TRUE),
                ('j6', 'batch',            'LP', FALSE)
            ) AS v(pod_id, job_type_public, priority_class, ready_status)
        """)
        df = con.execute("""
            SELECT
                pod_id,
                (
                    job_type_public NOT IN ('online_inference')
                    AND priority_class = 'LP'
                    AND ready_status = TRUE
                ) AS is_deferrable
            FROM jobs
        """).df()

        by_id = df.set_index("pod_id")["is_deferrable"]
        assert by_id["j1"] == False   # online_inference LP → NOT deferrable
        assert by_id["j2"] == False   # online_inference HP → NOT deferrable
        assert by_id["j3"] == True    # offline_inference LP ready → deferrable
        assert by_id["j4"] == True    # batch LP ready → deferrable
        assert by_id["j5"] == False   # batch HP → NOT deferrable (wrong priority)
        assert by_id["j6"] == False   # batch LP not ready → NOT deferrable


# ── 6. allocation_efficiency ──────────────────────────────────────────────────

class TestAllocationEfficiency:
    def test_efficiency_calculation(self):
        """Server with 8 GPUs × 10 hours = 80 capacity; 60 GPU-hours requested → 0.75."""
        con = _con()
        con.execute("""
            CREATE TABLE capacity AS
            SELECT * FROM (VALUES ('srv1', 80.0)) AS v(server_id, total_capacity_gpu_hours)
        """)
        con.execute("""
            CREATE TABLE requested AS
            SELECT * FROM (VALUES ('srv1', 60.0)) AS v(server_id, total_requested_gpu_hours)
        """)
        df = con.execute("""
            SELECT
                c.server_id,
                r.total_requested_gpu_hours / c.total_capacity_gpu_hours
                    AS allocation_efficiency
            FROM capacity c
            JOIN requested r USING (server_id)
        """).df()

        eff = df["allocation_efficiency"].iloc[0]
        assert abs(eff - 0.75) < 1e-9

    def test_zero_capacity_yields_null(self):
        con = _con()
        df = con.execute("""
            SELECT
                CASE WHEN 0.0 = 0 THEN NULL ELSE 60.0 / 0.0 END
                    AS allocation_efficiency
        """).df()
        assert pd.isna(df["allocation_efficiency"].iloc[0])
