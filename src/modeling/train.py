"""Train all five algorithms on the feature dataset.

Usage
-----
    python -m src.modeling.train          # full 40M rows (~90 min)
    python -m src.modeling.train --fast   # 10% sample (~5 min)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.modeling.algorithms import all_algorithms
from src.modeling.features import build_features
from src.modeling.split import make_splits

_FEAT   = PROCESSED_DIR / "model_features.parquet"
_SPLITS = PROCESSED_DIR / "model_splits.parquet"


def _load_xy(split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats  = pd.read_parquet(_FEAT)
    splits = pd.read_parquet(_SPLITS)
    merged = feats.merge(splits[["pod_id", "split"]], on="pod_id", how="left")
    merged["split"] = merged["split"].fillna("train")
    sub = merged[merged["split"] == split_name]
    feat_cols = [c for c in sub.columns if c.startswith("feat_")]
    X = sub[feat_cols].values.astype(np.float32)
    y = sub["y"].values.astype(np.float32)
    return X, y, feat_cols


def train_all(fast: bool = False) -> tuple[list, dict[str, float]]:
    """Fit all algorithms. Returns (fitted_models, {name: train_time_s})."""
    # Ensure features and splits exist
    build_features(fast=fast)
    make_splits()

    print("  [train] Loading train/val splits …")
    X_train, y_train, feat_names = _load_xy("train")
    X_val,   y_val,   _          = _load_xy("val")
    print(f"  [train] train={len(X_train):,}  val={len(X_val):,}  features={X_train.shape[1]}")

    fitted     = []
    train_times: dict[str, float] = {}

    for algo in all_algorithms():
        print(f"\n  [train] Fitting {algo.name} …")
        t0 = time.perf_counter()
        algo.fit(X_train, y_train, X_val, y_val)
        elapsed = time.perf_counter() - t0
        train_times[algo.name] = elapsed
        fitted.append(algo)
        print(f"  [train] {algo.name} done in {elapsed:.1f}s")

    return fitted, train_times


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train all five algorithms.")
    parser.add_argument("--fast", action="store_true", help="10%% subsample")
    args = parser.parse_args(argv)
    train_all(fast=args.fast)


if __name__ == "__main__":
    main()
