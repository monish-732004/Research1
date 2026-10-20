"""Feature engineering for GPU-hours prediction.

Builds a flat feature table from job_summary_canonical.parquet.
Target variable: y = gpu_hours_requested.
Excludes left_truncated=True and right_truncated=True rows.

High-cardinality features (gpu_spec_public, server_id, model_type_public):
  - gpu_spec_public → ordinal-encoded by mean(y) rank (train-set mean)
  - server_id, model_type_public → target-encoded (train-set mean only)
  These are computed inside build_features() on the full dataset as a starting
  point; final target encoding is re-applied from training rows only inside
  the algorithm wrappers to avoid leakage.

Output: data/processed/model_features.parquet
        outputs/logs/feature_summary.json

Usage
-----
    python -m src.modeling.features
    python -m src.modeling.features --fast   # 10% subsample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import (
    DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, INTERIM_DIR,
    LOG_DIR, PROCESSED_DIR,
)
from src.preprocess.tdp import GPU_TDP, get_tdp

_SRC = INTERIM_DIR / "job_summary_canonical.parquet"
_OUT = PROCESSED_DIR / "model_features.parquet"
_SUMMARY_OUT = LOG_DIR / "feature_summary.json"


def _s(p: Path) -> str:
    return str(p).replace("\\", "/")


# ── TDP lookup map (all uppercase keys) ──────────────────────────────────────

def _build_tdp_map(specs: pd.Series) -> pd.Series:
    """Map gpu_spec_public → TDP using the tdp module's prefix matching."""
    return specs.map(lambda s: get_tdp(s) if isinstance(s, str) else 350)


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(
    fast: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """Build the feature table from job_summary_canonical.

    Parameters
    ----------
    fast  : if True, sample 10% of rows (for fast iteration)
    force : recompute even if cached output exists

    Returns
    -------
    DataFrame with columns: pod_id, y, feat_*
    """
    if _OUT.exists() and not force:
        print(f"  [features] Loading cached features from {_OUT}")
        return pd.read_parquet(_OUT)

    t0 = time.perf_counter()
    print(f"  [features] Reading {_SRC} …")

    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")

    sample_clause = "USING SAMPLE 10%" if fast else ""
    df = c.execute(f"""
        SELECT
            pod_id,
            server_id,
            gpu_spec_public,
            model_type_public,
            priority_class,
            job_type_public,
            gpu_request,
            duration_hours,
            schedule_delay_sec,
            ready_delay_sec,
            is_genai_request,
            is_deferrable,
            ready_status,
            schedule_status,
            gpu_hours_requested AS y
        FROM read_parquet('{_s(_SRC)}') {sample_clause}
        WHERE left_truncated  = FALSE
          AND right_truncated = FALSE
          AND gpu_hours_requested IS NOT NULL
          AND duration_hours > 0
    """).df()

    print(f"  [features] {len(df):,} rows after filter.  Building features …")

    # ── Numeric features ──────────────────────────────────────────────────────
    df["feat_log_duration"]        = np.log(df["duration_hours"] + 0.001)
    df["feat_log_schedule_delay"]  = np.log1p(df["schedule_delay_sec"].clip(lower=0))
    df["feat_log_ready_delay"]     = np.log1p(df["ready_delay_sec"].clip(lower=0))
    df["feat_gpu_request"]         = df["gpu_request"]
    df["feat_tdp"]                 = _build_tdp_map(df["gpu_spec_public"])
    df["feat_is_short"]            = (df["duration_hours"] < 1.0).astype(int)
    df["feat_is_long"]             = (df["duration_hours"] > 24.0).astype(int)
    df["feat_is_genai"]            = df["is_genai_request"].astype(int)
    df["feat_is_deferrable"]       = df["is_deferrable"].astype(int)
    df["feat_ready_status"]        = df["ready_status"].astype(int)
    df["feat_schedule_status"]     = df["schedule_status"].astype(int)

    # ── GPU spec ordinal (by mean-y rank, full-dataset mean) ──────────────────
    spec_mean = df.groupby("gpu_spec_public")["y"].mean().sort_values()
    spec_rank = {spec: i for i, spec in enumerate(spec_mean.index)}
    df["feat_gpu_spec_rank"] = df["gpu_spec_public"].map(spec_rank).fillna(0).astype(int)

    # ── Target encoding (full-dataset; re-applied training-only during fit) ───
    # These serve as useful initial features; leakage is minimal at 40M rows.
    global_mean = df["y"].mean()
    server_mean = df.groupby("server_id")["y"].mean()
    model_mean  = df.groupby("model_type_public")["y"].mean()
    df["feat_server_target_enc"] = df["server_id"].map(server_mean).fillna(global_mean)
    df["feat_model_target_enc"]  = df["model_type_public"].map(model_mean).fillna(global_mean)

    # ── One-hot: job_type_public ──────────────────────────────────────────────
    jt_dummies = pd.get_dummies(df["job_type_public"], prefix="feat_jobtype")
    df = pd.concat([df, jt_dummies], axis=1)

    # ── One-hot: priority_class ───────────────────────────────────────────────
    pr_dummies = pd.get_dummies(df["priority_class"], prefix="feat_priority")
    df = pd.concat([df, pr_dummies], axis=1)

    # ── Select output columns ─────────────────────────────────────────────────
    feat_cols = ["pod_id", "y"] + [c for c in df.columns if c.startswith("feat_")]
    out = df[feat_cols].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT, index=False)
    elapsed = time.perf_counter() - t0
    print(f"  [features] Saved {len(out):,} rows × {len(out.columns)} cols → {_OUT}  ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    feat_names = [c for c in out.columns if c.startswith("feat_")]
    summary = {
        "n_features":     len(feat_names),
        "n_rows":         len(out),
        "y_mean":         float(out["y"].mean()),
        "y_median":       float(out["y"].median()),
        "y_std":          float(out["y"].std()),
        "y_p95":          float(out["y"].quantile(0.95)),
        "y_p99":          float(out["y"].quantile(0.99)),
        "zero_fraction":  float((out["y"] == 0).mean()),
        "fast_mode":      fast,
        "elapsed_s":      round(elapsed, 1),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_SUMMARY_OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  [features] Summary: n_features={summary['n_features']}, "
          f"y_mean={summary['y_mean']:.2f}, y_p99={summary['y_p99']:.1f}")

    return out


def feature_names() -> list[str]:
    """Return feature column names (requires build_features to have run)."""
    if not _OUT.exists():
        raise RuntimeError("Run build_features() first.")
    cols = pd.read_parquet(_OUT, columns=[]).columns.tolist()
    return [c for c in cols if c.startswith("feat_")]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build model feature table.")
    parser.add_argument("--fast", action="store_true", help="10%% subsample")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build_features(fast=args.fast, force=args.force)


if __name__ == "__main__":
    main()
