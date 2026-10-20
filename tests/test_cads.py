"""Tests for Phase 5 CADS simulation.

Deterministic 10-job, 48-hour synthetic trace with a known carbon profile:
  - Hours  0–8:  low carbon  (300 gCO₂/kWh)
  - Hours  9–17: high carbon (700 gCO₂/kWh)
  - Hours 18–47: low carbon  (300 gCO₂/kWh)

All jobs use the same region/anchor; the carbon array is injected directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.cads import _find_best_starts
from src.simulation.baseline import _job_co2_vectorised


# ── Synthetic carbon profile ──────────────────────────────────────────────────

N_HOURS = 4440
LOW_CI  = 300.0
HIGH_CI = 700.0


def _make_carbon(n: int = N_HOURS) -> np.ndarray:
    """Low carbon everywhere except hours 9–17 (mod 24)."""
    ci = np.full(n, LOW_CI)
    for h in range(n):
        if 9 <= (h % 24) <= 17:
            ci[h] = HIGH_CI
    return ci


CI    = _make_carbon()
CI_CS = np.concatenate([[0.0], np.cumsum(CI)])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jobs_df(**overrides) -> pd.DataFrame:
    """
    10-job synthetic DataFrame matching the schema expected by run_cads
    (via _load_jobs).  Uses integer gpu_request, float duration_hours.
    """
    data = {
        "pod_id":          [f"p{i}" for i in range(10)],
        "gpu_spec_public": ["A100"] * 10,
        "gpu_request":     [1, 2, 4, 1, 1, 2, 1, 1, 1, 1],
        "duration_hours":  [2.0, 3.0, 1.0, 4.0, 2.0,
                            6.0, 1.0, 2.0, 1.0, 3.0],
        # All start at hour 12 (high-carbon window)
        "schedule_delay_sec": [12*3600] * 10,
        "is_deferrable":   [True, False, True, True, False,
                            True, True,  True, False, True],
        "job_type_public": ["training", "inference", "training", "training",
                            "inference", "training", "training", "training",
                            "inference", "training"],
        "priority_class":  ["low", "high", "low", "low", "high",
                            "low",  "low",  "low", "high", "low"],
        "left_truncated":  [False] * 10,
        "right_truncated": [False] * 10,
    }
    data.update(overrides)
    df = pd.DataFrame(data)
    df["tdp_w"]  = 400
    df["tdp_kw"] = 0.4
    df["dur_h"]  = np.ceil(df["duration_hours"]).astype(np.int32).clip(1)
    df["start_h"] = (df["schedule_delay_sec"] / 3600.0).astype(np.int32)
    df["start_uncertain"] = df["schedule_delay_sec"] == 0
    return df


# ── Unit tests: _find_best_starts ─────────────────────────────────────────────

class TestFindBestStarts:

    def test_shifts_to_low_carbon_window(self):
        """A job starting at hour 12 (high) should shift to hour 4 (low) with max_defer=8."""
        starts = np.array([12], dtype=np.int32)
        durs   = np.array([2],  dtype=np.int32)
        best   = _find_best_starts(starts, durs, CI_CS, max_defer=8)
        # Hour 4 is in the low-carbon window [0,8]
        assert best[0] <= 8, f"Expected best start ≤8 (low carbon), got {best[0]}"
        # Should be better than the original start
        orig_ci = (CI_CS[12+2] - CI_CS[12]) / 2
        new_ci  = (CI_CS[best[0]+2] - CI_CS[best[0]]) / 2
        assert new_ci < orig_ci

    def test_max_defer_respected(self):
        """Job should not be shifted more than max_defer_hours."""
        starts = np.array([12], dtype=np.int32)
        durs   = np.array([2],  dtype=np.int32)
        for md in [1, 2, 4]:
            best = _find_best_starts(starts, durs, CI_CS, max_defer=md)
            assert best[0] <= 12 + md, (
                f"max_defer={md}: best_start={best[0]} > start+max_defer={12+md}"
            )

    def test_already_in_low_carbon_stays(self):
        """A job starting at hour 3 (low carbon) should not be moved."""
        starts = np.array([3], dtype=np.int32)
        durs   = np.array([2], dtype=np.int32)
        best   = _find_best_starts(starts, durs, CI_CS, max_defer=4)
        new_ci  = (CI_CS[best[0]+2] - CI_CS[best[0]]) / 2
        orig_ci = (CI_CS[3+2]       - CI_CS[3])       / 2
        assert new_ci <= orig_ci + 1e-6   # no worse

    def test_vectorised_batch(self):
        """Multiple jobs processed together gives the same result as individually."""
        starts = np.array([12, 14, 20, 0], dtype=np.int32)
        durs   = np.array([2,  3,  1,  4], dtype=np.int32)
        batch  = _find_best_starts(starts, durs, CI_CS, max_defer=8)
        for i in range(len(starts)):
            single = _find_best_starts(starts[i:i+1], durs[i:i+1], CI_CS, max_defer=8)
            assert batch[i] == single[0], (
                f"Job {i}: batch={batch[i]}, single={single[0]}"
            )


# ── Unit tests: CO₂ calculation ───────────────────────────────────────────────

class TestCO2Calculation:

    def test_co2_positive(self):
        starts = np.array([0, 12], dtype=np.int32)
        durs   = np.array([2,  2], dtype=np.int32)
        gpus   = np.array([1.0, 1.0], dtype=np.float32)
        tdp    = np.array([0.4, 0.4], dtype=np.float32)
        co2    = _job_co2_vectorised(starts, durs, gpus, tdp, CI_CS)
        assert (co2 > 0).all()

    def test_high_carbon_higher_cost(self):
        """Job at hour 12 (high CI) costs more than same job at hour 0 (low CI)."""
        gpus = np.array([1.0], dtype=np.float32)
        tdp  = np.array([0.4], dtype=np.float32)
        durs = np.array([2],   dtype=np.int32)
        co2_low  = _job_co2_vectorised(np.array([0],  dtype=np.int32), durs, gpus, tdp, CI_CS)
        co2_high = _job_co2_vectorised(np.array([12], dtype=np.int32), durs, gpus, tdp, CI_CS)
        assert co2_high[0] > co2_low[0]


# ── Integration tests: deferral behaviour ────────────────────────────────────

class TestCADSBehaviour:

    def _run(self, jobs, threshold=0.0, max_defer=8):
        """Run CADS directly using synthetic CI — bypass file I/O."""
        eligible_mask    = jobs["is_deferrable"].values.astype(bool)
        starts_all       = jobs["start_h"].values.astype(np.int32)
        durs_all         = jobs["dur_h"].values.astype(np.int32)
        ends_orig        = np.minimum(starts_all + durs_all, len(CI_CS) - 1)
        orig_mean_ci     = (CI_CS[ends_orig] - CI_CS[starts_all]) / np.maximum(durs_all, 1)
        above_thr        = orig_mean_ci > threshold
        defer_mask       = eligible_mask & above_thr
        new_starts       = starts_all.copy()
        if defer_mask.any():
            idx = np.where(defer_mask)[0]
            new_starts[idx] = _find_best_starts(starts_all[idx], durs_all[idx],
                                                CI_CS, max_defer)
        return starts_all, new_starts, defer_mask

    def test_non_deferrable_never_shifts(self):
        """Jobs with is_deferrable=False must never change start hour."""
        jobs = _make_jobs_df()
        orig, new, _ = self._run(jobs)
        non_def = ~jobs["is_deferrable"].values
        assert (orig[non_def] == new[non_def]).all(), \
            "Non-deferrable job start hours changed"

    def test_eligible_job_shifts_to_lower_carbon(self):
        """Deferrable job at high-CI hour 12 should move to lower-CI hour."""
        jobs = _make_jobs_df()
        orig, new, defer_mask = self._run(jobs, threshold=0.0, max_defer=8)
        for i in np.where(defer_mask)[0]:
            ci_orig = (CI_CS[orig[i]+int(jobs["dur_h"].iloc[i])] - CI_CS[orig[i]]) \
                      / jobs["dur_h"].iloc[i]
            ci_new  = (CI_CS[new[i]+int(jobs["dur_h"].iloc[i])]  - CI_CS[new[i]]) \
                      / jobs["dur_h"].iloc[i]
            assert ci_new <= ci_orig + 1e-6, \
                f"Job {i}: shifted to HIGHER CI ({ci_new:.1f} > {ci_orig:.1f})"

    def test_cads_total_co2_less_than_baseline(self):
        """Total CO₂ after CADS must be ≤ baseline CO₂."""
        jobs = _make_jobs_df()
        gpus   = jobs["gpu_request"].values.astype(np.float32)
        tdp_kw = jobs["tdp_kw"].values.astype(np.float32)
        orig, new, _ = self._run(jobs, threshold=0.0, max_defer=8)
        durs = jobs["dur_h"].values.astype(np.int32)
        baseline_co2 = _job_co2_vectorised(orig, durs, gpus, tdp_kw, CI_CS).sum()
        cads_co2     = _job_co2_vectorised(new,  durs, gpus, tdp_kw, CI_CS).sum()
        assert cads_co2 <= baseline_co2, \
            f"CADS CO₂ ({cads_co2:.2f}) > baseline ({baseline_co2:.2f})"

    def test_high_threshold_defers_nothing(self):
        """With threshold > max carbon intensity, no jobs are deferred."""
        jobs = _make_jobs_df()
        orig, new, defer_mask = self._run(jobs, threshold=10000.0, max_defer=8)
        assert not defer_mask.any(), "Jobs deferred despite threshold above max CI"
        assert (orig == new).all()

    def test_zero_max_defer_no_shift(self):
        """max_defer=0 means no shift is possible — all jobs stay put."""
        jobs = _make_jobs_df()
        orig, new, _ = self._run(jobs, threshold=0.0, max_defer=0)
        assert (orig == new).all(), "Jobs shifted despite max_defer=0"
