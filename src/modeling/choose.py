"""Model selection from benchmark results.

Decision rule (applied in order):
  1. Shortlist: all models within 5% of the best RMSE
  2. From shortlist: pick lowest prediction latency
  3. If tied: pick highest R²
  4. If still tied: tree > LSTM > linear (interpretability)

Writes outputs/logs/model_selection_rationale.md
Saves chosen model to data/processed/production_model.joblib
  (or references lstm_best.pt for the LSTM)

Usage
-----
    python -m src.modeling.choose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import LOG_DIR, PROCESSED_DIR, TABLE_DIR

_CMP      = TABLE_DIR / "model_comparison.csv"
_RATIONALE = LOG_DIR / "model_selection_rationale.md"
_MODEL_OUT = PROCESSED_DIR / "production_model.joblib"

TIER_ORDER = {"LinearRegression": 3, "LSTM": 2,
              "RandomForest": 1, "XGBoost": 1, "LightGBM": 1}


def choose_model(
    cmp_df: pd.DataFrame,
    fitted_models: list | None = None,
) -> dict:
    """Apply the decision rule and return a dict with the selected model info.

    Parameters
    ----------
    cmp_df        : DataFrame from model_comparison.csv
    fitted_models : list of fitted algorithm objects (optional; needed for saving)
    """
    df = cmp_df.copy().sort_values("rmse").reset_index(drop=True)
    best_rmse = df["rmse"].min()
    threshold = best_rmse * 1.05

    shortlist = df[df["rmse"] <= threshold].copy()
    shortlist = shortlist.sort_values("latency_ms_10k").reset_index(drop=True)

    # Step 3: within-1ms latency ties → prefer higher R²
    best_lat = shortlist["latency_ms_10k"].min()
    lat_tied = shortlist[shortlist["latency_ms_10k"] <= best_lat + 1.0]
    if len(lat_tied) > 1:
        shortlist = lat_tied.sort_values("r2", ascending=False).reset_index(drop=True)
    else:
        shortlist = lat_tied.reset_index(drop=True)

    # Step 4: among remaining ties → prefer interpretable tier (tree > LSTM > linear)
    shortlist = shortlist.copy()
    shortlist["_tier"] = shortlist["algorithm"].map(lambda a: TIER_ORDER.get(a, 99))
    shortlist = shortlist.sort_values("_tier").reset_index(drop=True)

    winner = shortlist.iloc[0]
    name   = winner["algorithm"]
    pct_of_best = float(winner["rmse"] / best_rmse * 100)

    reasoning = (
        f"RMSE {winner['rmse']:.4f} is within the 5% threshold of best "
        f"({best_rmse:.4f}); lowest latency among the shortlist; "
        f"tree-based models preferred over LSTM for interpretability."
        if len(shortlist) > 1 else
        f"Clear winner with RMSE {winner['rmse']:.4f}."
    )

    result = {
        "selected_algorithm": name,
        "rmse":               float(winner["rmse"]),
        "pct_of_best":        round(pct_of_best, 1),
        "latency_ms_10k":     float(winner["latency_ms_10k"]),
        "r2":                 float(winner["r2"]),
        "reasoning":          reasoning,
    }

    # Write rationale
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_RATIONALE, "w", encoding="utf-8") as fh:
        fh.write(f"# Model Selection Rationale\n\n")
        fh.write(f"**Selected:** {name}\n\n")
        fh.write(f"| Metric | Value |\n|--------|-------|\n")
        fh.write(f"| RMSE | {winner['rmse']:.4f} ({pct_of_best:.1f}% of best) |\n")
        fh.write(f"| Latency | {winner['latency_ms_10k']:.2f} ms / 10K rows |\n")
        fh.write(f"| R² | {winner['r2']:.4f} |\n")
        fh.write(f"| MAE | {winner['mae']:.4f} |\n\n")
        fh.write(f"**Reasoning:** {reasoning}\n\n")
        fh.write("## Shortlist\n\n```\n")
        fh.write(shortlist[["algorithm", "rmse", "latency_ms_10k", "r2"]]
                 .to_string(index=False))
        fh.write("\n```\n\n## Full comparison\n\n```\n")
        fh.write(df[["algorithm", "rmse", "mae", "r2", "latency_ms_10k"]]
                 .to_string(index=False))
        fh.write("\n```\n")

    # Save model
    if fitted_models:
        for m in fitted_models:
            if m.name == name:
                if name == "LSTM":
                    print(f"  [choose] LSTM selected → model at {PROCESSED_DIR / 'lstm_best.pt'}")
                else:
                    joblib.dump(m, _MODEL_OUT)
                    print(f"  [choose] Saved {name} → {_MODEL_OUT}")
                break

    print(f"  [choose] Selected: {name}  RMSE={result['rmse']:.4f}  "
          f"lat={result['latency_ms_10k']:.2f}ms/10K  R²={result['r2']:.4f}")
    return result


def load_production_model():
    """Load the saved production model for inference."""
    if _MODEL_OUT.exists():
        return joblib.load(_MODEL_OUT)
    lstm_path = PROCESSED_DIR / "lstm_best.pt"
    if lstm_path.exists():
        from src.modeling.algorithms import LSTMRegressor
        m = LSTMRegressor(ckpt_path=lstm_path)
        # n_features must be set; infer from features parquet
        feat = pd.read_parquet(PROCESSED_DIR / "model_features.parquet",
                                columns=[c for c in pd.read_parquet(
                                    PROCESSED_DIR / "model_features.parquet",
                                    columns=[]
                                ).columns if c.startswith("feat_")])
        return m   # caller must fit before predict if not saved
    raise FileNotFoundError("No production model found. Run src.modeling.choose first.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Select best model from benchmark.")
    parser.parse_args(argv)

    if not _CMP.exists():
        print("Run src.modeling.evaluate first.")
        return

    cmp_df = pd.read_csv(_CMP)
    choose_model(cmp_df)


if __name__ == "__main__":
    main()
