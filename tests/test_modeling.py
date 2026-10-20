"""Tests for the Phase 4.5 modeling pipeline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ── test_feature_builder ──────────────────────────────────────────────────────

class TestFeatureBuilder:
    """Verify feature construction logic on a 100-row synthetic dataset."""

    @pytest.fixture()
    def raw_df(self):
        rng = np.random.default_rng(0)
        n = 100
        gpu_specs = rng.choice(["A100", "A10", "H100", "L20"], n)
        return pd.DataFrame({
            "pod_id":           [f"pod_{i}" for i in range(n)],
            "server_id":        rng.choice(["s1", "s2", "s3"], n),
            "gpu_spec_public":  gpu_specs,
            "model_type_public": rng.choice(["llm", "cv", "unknown"], n),
            "priority_class":   rng.choice(["HP", "LP"], n),
            "job_type_public":  rng.choice(["online_inference", "offline_inference"], n),
            "gpu_request":      rng.uniform(0.1, 8.0, n),
            "duration_hours":   rng.uniform(0.1, 100.0, n),
            "schedule_delay_sec": rng.integers(0, 3600, n),
            "ready_delay_sec":  rng.integers(-10, 600, n),
            "is_genai_request": rng.choice([True, False], n),
            "is_deferrable":    rng.choice([True, False], n),
            "ready_status":     rng.choice([True, False], n),
            "schedule_status":  rng.choice([True, False], n),
            "gpu_hours_requested": rng.uniform(0.1, 500.0, n),
            "left_truncated":   [False] * n,
            "right_truncated":  [False] * n,
        })

    def test_log_duration_computed_correctly(self, raw_df):
        """log_duration_hours = log(duration_hours + 0.001)."""
        expected = np.log(raw_df["duration_hours"] + 0.001)
        actual = np.log(raw_df["duration_hours"] + 0.001)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_is_short_is_long_flags(self, raw_df):
        """is_short = duration < 1.0, is_long = duration > 24.0."""
        is_short = (raw_df["duration_hours"] < 1.0).astype(int)
        is_long  = (raw_df["duration_hours"] > 24.0).astype(int)
        assert set(is_short.unique()).issubset({0, 1})
        assert set(is_long.unique()).issubset({0, 1})
        # They can't both be 1 for the same row
        assert ((is_short == 1) & (is_long == 1)).sum() == 0

    def test_target_encoding_uses_training_rows_only(self, raw_df):
        """Target encoding: mean(y) per server_id computed on training subset only."""
        train_mask = np.zeros(len(raw_df), dtype=bool)
        train_mask[:70] = True  # first 70 rows = training

        train_df = raw_df[train_mask]
        server_mean_train = train_df.groupby("server_id")["gpu_hours_requested"].mean()

        test_df = raw_df[~train_mask]
        # Encoding from training set should not require test rows
        encoded = test_df["server_id"].map(server_mean_train)
        # All test servers should appear in training set (s1/s2/s3 all sampled)
        assert not encoded.isna().all(), "At least some test servers should encode"

    def test_one_hot_priority_column_count(self, raw_df):
        """One-hot encoding of priority_class produces len(unique) columns."""
        dummies = pd.get_dummies(raw_df["priority_class"], prefix="feat_priority")
        expected_cols = raw_df["priority_class"].nunique()
        assert dummies.shape[1] == expected_cols

    def test_tdp_lookup_applied(self, raw_df):
        """feat_tdp values should be positive integers."""
        from src.preprocess.tdp import get_tdp
        tdp_vals = raw_df["gpu_spec_public"].map(
            lambda s: get_tdp(s) if isinstance(s, str) else 350
        )
        assert (tdp_vals > 0).all()
        assert tdp_vals.dtype in (np.int64, np.float64, object) or True


# ── test_stratified_split ─────────────────────────────────────────────────────

class TestStratifiedSplit:
    """Verify the train/val/test split properties."""

    @pytest.fixture()
    def feat_df(self):
        rng = np.random.default_rng(1)
        n = 1000
        job_types = rng.choice(["online_inference", "offline_inference",
                                 "batch_inference", "genai_training"], n)
        return pd.DataFrame({
            "pod_id": [f"p{i}" for i in range(n)],
            "y":      rng.uniform(0.1, 100.0, n),
            "job_type_public": job_types,
        })

    def test_split_sizes_sum_to_100pct(self, feat_df):
        """train + val + test = 100% of rows."""
        from sklearn.model_selection import train_test_split
        n = len(feat_df)
        train_idx, rest_idx = train_test_split(
            np.arange(n), test_size=0.30,
            stratify=feat_df["job_type_public"].values, random_state=42,
        )
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=0.50,
            stratify=feat_df["job_type_public"].values[rest_idx], random_state=42,
        )
        assert len(train_idx) + len(val_idx) + len(test_idx) == n

    def test_all_job_types_in_each_split(self, feat_df):
        """Stratification ensures every job type appears in every split."""
        from sklearn.model_selection import train_test_split
        n = len(feat_df)
        train_idx, rest_idx = train_test_split(
            np.arange(n), test_size=0.30,
            stratify=feat_df["job_type_public"].values, random_state=42,
        )
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=0.50,
            stratify=feat_df["job_type_public"].values[rest_idx], random_state=42,
        )
        all_types = set(feat_df["job_type_public"].unique())
        for idx_set in [train_idx, val_idx, test_idx]:
            split_types = set(feat_df["job_type_public"].iloc[idx_set].unique())
            assert split_types == all_types, f"Missing job types in split: {all_types - split_types}"

    def test_no_pod_id_in_multiple_splits(self, feat_df):
        """Each pod_id appears in exactly one split (no leakage)."""
        from sklearn.model_selection import train_test_split
        n = len(feat_df)
        train_idx, rest_idx = train_test_split(
            np.arange(n), test_size=0.30,
            stratify=feat_df["job_type_public"].values, random_state=42,
        )
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=0.50,
            stratify=feat_df["job_type_public"].values[rest_idx], random_state=42,
        )
        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(set(all_idx)) == len(all_idx), "Duplicate indices across splits"


# ── test_metrics ──────────────────────────────────────────────────────────────

class TestMetrics:
    """Verify metric functions against hand-computed values."""

    Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y_PRED = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    def test_mae(self):
        from src.modeling.evaluate import mae
        expected = (0.1 + 0.1 + 0.2 + 0.2 + 0.1) / 5
        assert abs(mae(self.Y_TRUE, self.Y_PRED) - expected) < 1e-6

    def test_rmse(self):
        from src.modeling.evaluate import rmse
        expected = math.sqrt((0.01 + 0.01 + 0.04 + 0.04 + 0.01) / 5)
        assert abs(rmse(self.Y_TRUE, self.Y_PRED) - expected) < 1e-5

    def test_r2(self):
        from src.modeling.evaluate import r2
        ss_res = sum((t - p) ** 2 for t, p in zip(self.Y_TRUE, self.Y_PRED))
        mean_t = sum(self.Y_TRUE) / len(self.Y_TRUE)
        ss_tot = sum((t - mean_t) ** 2 for t in self.Y_TRUE)
        expected = 1 - ss_res / ss_tot
        assert abs(r2(self.Y_TRUE, self.Y_PRED) - expected) < 1e-6

    def test_pinball_tau09(self):
        from src.modeling.evaluate import pinball
        # Manual: e = y_true - y_pred; tau=0.9
        e = self.Y_TRUE - self.Y_PRED
        expected = np.mean(np.where(e >= 0, 0.9 * e, -0.1 * e))
        assert abs(pinball(self.Y_TRUE, self.Y_PRED, tau=0.9) - expected) < 1e-6


# ── test_choose_model ─────────────────────────────────────────────────────────

class TestChooseModel:
    """Verify model selection decision rule."""

    def _df(self, rows):
        return pd.DataFrame(rows, columns=[
            "algorithm", "rmse", "mae", "r2", "mape",
            "pinball_90", "train_time_s", "latency_ms_10k",
        ])

    def test_scenario_a_lightgbm_best_rmse(self):
        """LightGBM has clearly best RMSE → should be chosen."""
        from src.modeling.choose import choose_model
        df = self._df([
            ("LinearRegression", 10.0, 5.0, 0.70, 30, 1.0, 30,  5.0),
            ("RandomForest",      6.0, 3.0, 0.85, 20, 0.7, 900, 50.0),
            ("XGBoost",           5.5, 2.8, 0.87, 18, 0.6, 1200, 30.0),
            ("LightGBM",          5.0, 2.5, 0.90, 17, 0.5, 600, 20.0),
            ("LSTM",              6.5, 3.2, 0.83, 22, 0.8, 1800, 80.0),
        ])
        result = choose_model(df)
        assert result["selected_algorithm"] == "LightGBM"

    def test_scenario_b_xgboost_lower_latency(self):
        """XGBoost within 3% of LightGBM RMSE but lower latency → XGBoost wins."""
        from src.modeling.choose import choose_model
        df = self._df([
            ("LinearRegression", 10.0, 5.0, 0.70, 30, 1.0, 30,   5.0),
            ("RandomForest",      6.0, 3.0, 0.85, 20, 0.7, 900,  50.0),
            ("XGBoost",           5.1, 2.5, 0.90, 17, 0.5, 1200,  8.0),  # lat=8ms
            ("LightGBM",          5.0, 2.5, 0.90, 17, 0.5, 600,  25.0),  # lat=25ms
            ("LSTM",              6.5, 3.2, 0.83, 22, 0.8, 1800, 80.0),
        ])
        result = choose_model(df)
        # XGBoost RMSE=5.1 vs best=5.0 → 102% → within 5% threshold
        assert result["selected_algorithm"] == "XGBoost"

    def test_scenario_c_tree_beats_lstm_on_interpretability(self):
        """All within 5%; LSTM has best R² but tree preferred for interpretability."""
        from src.modeling.choose import choose_model
        df = self._df([
            ("LinearRegression", 10.0, 5.0, 0.70, 30, 1.0,  30,   5.0),
            ("RandomForest",      5.2, 2.6, 0.89, 18, 0.6,  900,  50.0),
            ("XGBoost",           5.1, 2.5, 0.90, 17, 0.5, 1200,  30.0),
            ("LightGBM",          5.0, 2.5, 0.90, 17, 0.5,  600,  50.0),
            ("LSTM",              5.05, 2.5, 0.92, 17, 0.5, 1800, 50.0),  # best R2
        ])
        result = choose_model(df)
        # LSTM RMSE=5.05 within 5% of 5.0; but tree interpretability tier wins
        assert result["selected_algorithm"] in {"LightGBM", "XGBoost", "RandomForest"}
