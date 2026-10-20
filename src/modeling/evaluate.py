"""Evaluation metrics and feature importance for all six algorithms.

Computes on TEST split only (train/val are for fitting; test is held out).

Metrics per algorithm: MAE, RMSE, MAPE, R², Pinball@0.9,
                       training time, prediction latency.
CAPSR-only metric: Deferral Precision@K (fraction of top-K confidence jobs
                   that are actually above-mean gpu_hours for their stratum).
Also produces per-job-type MAE/RMSE and feature importance (tree models).

Outputs
-------
outputs/tables/model_comparison.csv
outputs/tables/model_comparison_per_jobtype.csv
outputs/tables/feature_importance.csv

Usage
-----
    python -m src.modeling.evaluate --fast
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LOG_DIR, PROCESSED_DIR, TABLE_DIR

_FEAT    = PROCESSED_DIR / "model_features.parquet"
_SPLITS  = PROCESSED_DIR / "model_splits.parquet"
_OUT_CMP = TABLE_DIR / "model_comparison.csv"
_OUT_JT  = TABLE_DIR / "model_comparison_per_jobtype.csv"
_OUT_FI  = TABLE_DIR / "feature_importance.csv"


# ── Metric functions ──────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_y: float = 0.1) -> float:
    mask = y_true >= min_y
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pinball(y_true: np.ndarray, y_pred: np.ndarray, tau: float = 0.9) -> float:
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def latency_ms_per_10k(model, X: np.ndarray, n_runs: int = 5) -> float:
    """Median wall-clock time to predict 10,000 rows, in milliseconds."""
    Xs = X[:10_000]
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(Xs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def deferral_precision_at_k(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_pct: float = 0.10,
) -> float | None:
    """Deferral Precision@K for CAPSR: fraction of top-K confidence jobs that
    are actually above the global mean gpu_hours.

    Returns None for models without predict_deferral_confidence().
    """
    if not hasattr(model, "predict_deferral_confidence"):
        return None
    confidence = model.predict_deferral_confidence(X_test).astype(np.float64)
    k          = max(1, int(len(y_test) * k_pct))
    top_k_idx  = np.argsort(confidence)[-k:]
    global_mean = float(y_test.mean())
    precision   = float((y_test[top_k_idx] > global_mean).mean())
    return round(precision, 4)


# ── Load data ─────────────────────────────────────────────────────────────────

def _load_split(split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features + labels for a named split.  Returns (X, y, feat_names)."""
    feats  = pd.read_parquet(_FEAT)
    splits = pd.read_parquet(_SPLITS)
    merged = feats.merge(splits[["pod_id", "split"]], on="pod_id", how="left")
    merged["split"] = merged["split"].fillna("train")
    sub = merged[merged["split"] == split_name]
    feat_cols = [c for c in sub.columns if c.startswith("feat_")]
    X = sub[feat_cols].values.astype(np.float32)
    y = sub["y"].values.astype(np.float32)
    return X, y, feat_cols


# ── Per-job-type breakdown ────────────────────────────────────────────────────

def _per_jobtype_metrics(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_names: list[str],
) -> pd.DataFrame:
    """MAE + RMSE on test set broken down by job_type_public."""
    from src.config import INTERIM_DIR
    import duckdb
    from src.config import DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS

    feats  = pd.read_parquet(_FEAT, columns=["pod_id"])
    splits = pd.read_parquet(_SPLITS)
    merged = feats.merge(splits[["pod_id", "split", "job_type_public"]], on="pod_id",
                         how="left")
    test_jt = merged[merged["split"] == "test"]["job_type_public"].values
    test_jt = test_jt[:len(X_test)]  # align with fast-mode truncation

    pred = model.predict(X_test)
    rows = []
    for jt in np.unique(test_jt):
        mask = test_jt == jt
        rows.append({
            "algorithm":      model.name,
            "job_type_public": jt,
            "mae":  round(mae(y_test[mask], pred[mask]), 4),
            "rmse": round(rmse(y_test[mask], pred[mask]), 4),
            "n":    int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ── Feature importance ────────────────────────────────────────────────────────

def _feature_importance(model, feat_names: list[str]) -> pd.DataFrame | None:
    """Extract feature importance for tree models. Returns None for others."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_names)
        imp = imp.sort_values(ascending=False).head(15)
        return pd.DataFrame({
            "algorithm": model.name,
            "feature":   imp.index,
            "importance": imp.values,
        })
    if hasattr(model, "feature_importances") and callable(model.feature_importances):
        try:
            imp = model.feature_importances(feat_names).head(15)
            return pd.DataFrame({
                "algorithm": model.name,
                "feature":   imp.index,
                "importance": imp.values,
            })
        except Exception:
            return None
    return None


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    fitted_models: list,
    train_times: dict[str, float],
    fast: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all fitted models on the test split.

    Parameters
    ----------
    fitted_models : list of fitted algorithm objects
    train_times   : {model.name: seconds}
    fast          : if True, limit test to 100K rows

    Returns
    -------
    (comparison_df, per_jobtype_df, feature_importance_df)
    """
    X_test, y_test, feat_names = _load_split("test")
    if fast:
        X_test, y_test = X_test[:100_000], y_test[:100_000]

    X_val, y_val, _ = _load_split("val")

    cmp_rows  = []
    jt_frames = []
    fi_frames = []

    for model in fitted_models:
        print(f"  [eval] {model.name} …")
        pred = model.predict(X_test)

        dp_at_k = deferral_precision_at_k(model, X_test, y_test, k_pct=0.10)
        row = {
            "algorithm":                  model.name,
            "mae":                        round(mae(y_test, pred), 4),
            "rmse":                       round(rmse(y_test, pred), 4),
            "mape":                       round(mape(y_test, pred), 2),
            "r2":                         round(r2(y_test, pred), 4),
            "pinball_90":                 round(pinball(y_test, pred), 4),
            "train_time_s":               round(train_times.get(model.name, float("nan")), 1),
            "latency_ms_10k":             round(latency_ms_per_10k(model, X_test), 2),
            "deferral_precision_at_10pct": dp_at_k,
        }
        cmp_rows.append(row)

        jt_frames.append(_per_jobtype_metrics(model, X_test, y_test, feat_names))
        fi = _feature_importance(model, feat_names)
        if fi is not None:
            fi_frames.append(fi)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(_OUT_CMP, index=False)

    jt_df = pd.concat(jt_frames, ignore_index=True) if jt_frames else pd.DataFrame()
    if not jt_df.empty:
        jt_df.to_csv(_OUT_JT, index=False)

    fi_df = pd.concat(fi_frames, ignore_index=True) if fi_frames else pd.DataFrame()
    if not fi_df.empty:
        fi_df.to_csv(_OUT_FI, index=False)

    print(f"\n=== Model Comparison (test set) ===")
    print(cmp_df.to_string(index=False))

    return cmp_df, jt_df, fi_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate all five algorithms.")
    parser.add_argument("--fast", action="store_true", help="100K test rows only")
    args = parser.parse_args(argv)

    if not _FEAT.exists() or not _SPLITS.exists():
        print("Run src.modeling.features and src.modeling.split first.")
        return

    from src.modeling.train import train_all
    fitted, train_times = train_all(fast=args.fast)
    run_evaluation(fitted, train_times, fast=args.fast)


if __name__ == "__main__":
    main()
