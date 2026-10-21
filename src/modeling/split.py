"""Stratified random train / val / test split for the modeling dataset.

Temporal split is not possible — job_execution_summary has no reliable
start timestamps. Stratified random split ensures job type distribution
is representative across splits. This is a known limitation — documented
in paper Section 6.1.

Split sizes: train=70%, val=15%, test=15%
Stratification key: job_type_public

Output: data/processed/model_splits.parquet
  Columns: pod_id, split ("train"|"val"|"test"), job_type_public

Usage
-----
    python -m src.modeling.split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR

_FEAT = PROCESSED_DIR / "model_features.parquet"
_OUT  = PROCESSED_DIR / "model_splits.parquet"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42


def make_splits(force: bool = False) -> pd.DataFrame:
    """Create and save train/val/test split index.

    Returns
    -------
    DataFrame[pod_id: str, split: str, job_type_public: str]
    """
    if _OUT.exists() and not force:
        return pd.read_parquet(_OUT)

    if not _FEAT.exists():
        raise RuntimeError("Run src.modeling.features first.")

    print("  [split] Reading feature table pod_ids and job types …")
    # job_type_public is stored in the interim canonical, but to avoid re-reading
    # 40M rows we re-read it from the already-filtered features parquet via DuckDB.
    import duckdb
    from src.config import DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS, INTERIM_DIR

    c = duckdb.connect()
    c.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    c.execute(f"SET threads={DUCKDB_THREADS}")

    feat_path = str(_FEAT).replace("\\", "/")
    interim   = str(INTERIM_DIR / "job_summary_canonical.parquet").replace("\\", "/")

    # Join on pod_id to get job_type for only the filtered feature rows.
    # Use ANY_VALUE to collapse duplicate pod_ids in the canonical to one row.
    merged = c.execute(f"""
        SELECT f.pod_id,
               COALESCE(ANY_VALUE(j.job_type_public), 'unknown') AS job_type_public
        FROM read_parquet('{feat_path}') f
        LEFT JOIN read_parquet('{interim}') j USING (pod_id)
        GROUP BY f.pod_id
    """).df()

    # Stratified split: train vs (val+test), then split remainder into val/test
    train_idx, rest_idx = train_test_split(
        np.arange(len(merged)),
        test_size=(1 - TRAIN_FRAC),
        stratify=merged["job_type_public"].values,
        random_state=SEED,
    )
    val_size_of_rest = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=(1 - val_size_of_rest),
        stratify=merged["job_type_public"].values[rest_idx],
        random_state=SEED,
    )

    split_col = np.full(len(merged), "train", dtype=object)
    split_col[val_idx]  = "val"
    split_col[test_idx] = "test"

    result = pd.DataFrame({
        "pod_id":          merged["pod_id"],
        "split":           split_col,
        "job_type_public": merged["job_type_public"],
    })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(_OUT, index=False)

    counts = result["split"].value_counts()
    print(f"  [split] train={counts.get('train', 0):,}  "
          f"val={counts.get('val', 0):,}  "
          f"test={counts.get('test', 0):,}  → {_OUT}")
    return result


def load_splits(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_train, X_val, X_test) DataFrames with split column dropped.

    feat_df must contain pod_id and feat_* columns plus y.
    """
    splits = make_splits()
    merged = feat_df.merge(splits[["pod_id", "split"]], on="pod_id", how="left")
    merged["split"] = merged["split"].fillna("train")

    feat_cols = [c for c in merged.columns if c.startswith("feat_")] + ["y"]
    train = merged[merged["split"] == "train"][feat_cols].reset_index(drop=True)
    val   = merged[merged["split"] == "val"][feat_cols].reset_index(drop=True)
    test  = merged[merged["split"] == "test"][feat_cols].reset_index(drop=True)
    return train, val, test


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    make_splits(force=args.force)


if __name__ == "__main__":
    main()
