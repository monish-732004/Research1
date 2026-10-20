"""Six regression algorithms for GPU-hours prediction.

Each class is sklearn-compatible (.fit / .predict / .name).
Training on the full 40M-row dataset takes ~90 min; use --fast for 10% subsample.

Algorithm 5 note: The LSTM treats each job as a sequence of length 1.
Without a true time series this reduces to a 2-layer MLP — performance
reflects deep learning on tabular data, not sequence modeling.

Algorithm 6 — CAPSR: Carbon-Aware Priority-Stratified Regressor.
  Novel domain-specific algorithm built from first principles (numpy/scipy only).
  Three contributions vs standard algorithms:
    1. Stratified sub-models per (job_type × priority_class) stratum
    2. Carbon-weighted loss — samples weighted by execution-window carbon intensity
    3. Deferral confidence score — sigmoid(z-score vs stratum mean)

Usage
-----
    python -m src.modeling.algorithms --fast    # quick smoke test
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

_LSTM_CKPT = PROCESSED_DIR / "lstm_best.pt"


# ── 1. Linear Regression baseline ────────────────────────────────────────────

class LinearReg:
    name = "LinearRegression"

    def __init__(self) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LinearReg":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ── 2. Random Forest ──────────────────────────────────────────────────────────

class RandomForest:
    name = "RandomForest"

    def __init__(self) -> None:
        from sklearn.ensemble import RandomForestRegressor
        self._model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "RandomForest":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


# ── 3. XGBoost ────────────────────────────────────────────────────────────────

class XGBoost:
    name = "XGBoost"

    def __init__(self) -> None:
        import xgboost as xgb
        self._model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            eval_metric="rmse",
            tree_method="hist",
            device="cpu",
            random_state=42,
            verbosity=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "XGBoost":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self._model.fit(X, y, eval_set=eval_set, verbose=200)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        scores = self._model.get_booster().get_score(importance_type="gain")
        # XGBoost uses f0, f1, ... keys when feature names not set
        mapped = {}
        for k, v in scores.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    mapped[feature_names[idx]] = v
            else:
                mapped[k] = v
        return pd.Series(mapped).sort_values(ascending=False)


# ── 4. LightGBM ───────────────────────────────────────────────────────────────

class LightGBM:
    name = "LightGBM"

    def __init__(self) -> None:
        import lightgbm as lgb
        self._model = lgb.LGBMRegressor(
            n_estimators=2000,
            num_leaves=127,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=42,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LightGBM":
        import lightgbm as lgb
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200),
        ]
        fit_kwargs: dict = {"callbacks": callbacks}
        if X_val is not None:
            fit_kwargs["eval_X"] = X_val
            fit_kwargs["eval_y"] = y_val
        self._model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def feature_importances(self, feature_names: list[str]) -> pd.Series:
        scores = self._model.booster_.feature_importance(importance_type="gain")
        return pd.Series(scores, index=feature_names).sort_values(ascending=False)


# ── 5. LSTM (PyTorch — degenerates to MLP with seq_len=1) ────────────────────

class LSTMRegressor:
    name = "LSTM"

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 4096,
        max_epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 10,
        ckpt_path: Path = _LSTM_CKPT,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.batch_size  = batch_size
        self.max_epochs  = max_epochs
        self.lr          = lr
        self.patience    = patience
        self.ckpt_path   = ckpt_path
        self._net        = None
        self._n_features = None

    def _build_net(self, n_features: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, 1, input_size) — seq_len=1 → reduces to MLP
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        return _Net(n_features, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LSTMRegressor":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._n_features = X.shape[1]
        net = self._build_net(self._n_features).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = ReduceLROnPlateau(opt, patience=5, factor=0.5)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        train_dl = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv_t = torch.tensor(X_val, dtype=torch.float32)
            yv_t = torch.tensor(y_val, dtype=torch.float32)
            val_dl = DataLoader(
                TensorDataset(Xv_t, yv_t),
                batch_size=self.batch_size * 2, shuffle=False,
            )

        best_val_rmse = float("inf")
        no_improve = 0

        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in train_dl:
                xb = xb.unsqueeze(1).to(device)   # (batch, 1, features)
                yb = yb.to(device)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()

            if has_val:
                net.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb = xb.unsqueeze(1).to(device)
                        yb = yb.to(device)
                        pred = net(xb)
                        val_losses.append(((pred - yb) ** 2).mean().item())
                val_rmse = float(np.sqrt(np.mean(val_losses)))
                sched.step(val_rmse)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    no_improve = 0
                    self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(), self.ckpt_path)
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print(f"  [LSTM] Early stopping at epoch {epoch+1} "
                              f"(val RMSE={val_rmse:.4f})")
                        break

                if (epoch + 1) % 10 == 0:
                    print(f"  [LSTM] Epoch {epoch+1:3d}  val RMSE={val_rmse:.4f}")

        if self.ckpt_path.exists():
            net.load_state_dict(torch.load(self.ckpt_path, map_location=device,
                                           weights_only=True))
        self._net = net
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        net = self._net
        net.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        dl  = DataLoader(TensorDataset(X_t), batch_size=self.batch_size * 2,
                         shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.unsqueeze(1).to(self._device)
                preds.append(net(xb).cpu().numpy())
        return np.concatenate(preds)


# ── 6. CAPSR: Carbon-Aware Priority-Stratified Regressor ─────────────────────

class CAPSRRegressor:
    """Carbon-Aware Priority-Stratified Regressor.

    Built from first principles using only numpy + scipy — no sklearn estimator
    base class.  sklearn-compatible interface (.fit / .predict / .name).

    Novel contributions vs standard algorithms:
    1. Stratified sub-models per (job_type_public × priority_class) stratum
       — each stratum learns its own distribution via a separate Ridge model.
    2. Carbon-weighted loss — samples weighted by carbon intensity of their
       execution window; high-carbon jobs contribute more to the fit.
    3. Deferral confidence score — predict_deferral_confidence() outputs a
       sigmoid(z-score) indicating how strongly each job is a defer candidate.
    """

    name = "CAPSR"

    def __init__(
        self,
        base_estimator: str = "ridge",   # only 'ridge' implemented; 'tree' reserved
        carbon_weight_col: str | None = None,
        min_stratum_size: int = 100,
        fallback_to_global: bool = True,
        random_state: int = 42,
        ridge_alpha: float = 1.0,
        feat_names: list[str] | None = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.carbon_weight_col = carbon_weight_col
        self.min_stratum_size  = min_stratum_size
        self.fallback_to_global = fallback_to_global
        self.random_state      = random_state
        self.ridge_alpha       = ridge_alpha
        self._feat_names_arg   = feat_names

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_feat_names(self) -> list[str]:
        """Load feature column names from the processed parquet on disk."""
        from src.config import PROCESSED_DIR
        p = PROCESSED_DIR / "model_features.parquet"
        if p.exists():
            import pandas as _pd
            cols = _pd.read_parquet(p, columns=[]).columns.tolist()
            return [c for c in cols if c.startswith("feat_")]
        return []

    def _build_stratum_indices(self, feat_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return index arrays for feat_jobtype_* and feat_priority_* columns."""
        jt = np.array([i for i, n in enumerate(feat_names)
                       if n.startswith("feat_jobtype_")], dtype=np.int32)
        pr = np.array([i for i, n in enumerate(feat_names)
                       if n.startswith("feat_priority_")], dtype=np.int32)
        return jt, pr

    def _get_strata(self, X: np.ndarray) -> np.ndarray:
        """Return (N,) int array encoding (job_type_idx, priority_idx) per row.

        Encodes as jt_idx * 1000 + pr_idx; supports up to 1000 job types and
        1000 priority classes.
        """
        if len(self.jt_indices_) > 0:
            jt = np.argmax(X[:, self.jt_indices_], axis=1).astype(np.int32)
        else:
            jt = np.zeros(len(X), dtype=np.int32)
        if len(self.pr_indices_) > 0:
            pr = np.argmax(X[:, self.pr_indices_], axis=1).astype(np.int32)
        else:
            pr = np.zeros(len(X), dtype=np.int32)
        return jt * 1000 + pr

    @staticmethod
    def _fit_ridge(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
        sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Closed-form weighted Ridge: (X'WX + α I)⁻¹ X'Wy.

        Uses scipy.linalg.solve on the (F×F) normal-equation system.
        """
        from scipy.linalg import solve
        Xd = X.astype(np.float64)
        yd = y.astype(np.float64)
        if sample_weight is not None:
            W = sample_weight.astype(np.float64)
            XtW = Xd.T * W[np.newaxis, :]
        else:
            XtW = Xd.T
        A   = XtW @ Xd + alpha * np.eye(Xd.shape[1])
        b   = XtW @ yd
        return solve(A, b, assume_a="pos")

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        carbon_weights: np.ndarray | None = None,
    ) -> "CAPSRRegressor":
        """Fit stratified Ridge sub-models + global fallback.

        Parameters
        ----------
        carbon_weights : optional (N,) array of carbon intensity values.
            Sample weight for job i = 1 + carbon_weights[i] / mean(carbon_weights).
            If None, uniform weights are used (standard Ridge).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Resolve feature names
        self.feat_names_   = self._feat_names_arg or self._load_feat_names()
        self.jt_indices_, self.pr_indices_ = self._build_stratum_indices(self.feat_names_)

        # Build sample weights
        if carbon_weights is not None:
            ci      = np.asarray(carbon_weights, dtype=np.float64)
            mean_ci = ci.mean() + 1e-8
            weights = (1.0 + ci / mean_ci).astype(np.float64)
        else:
            weights = None

        # Global model and stats (used as fallback for small strata)
        self.global_coef_  = self._fit_ridge(X, y, self.ridge_alpha, weights)
        self.global_stats_ = {
            "mean": float(np.mean(y)),
            "std":  float(np.std(y) + 1e-8),
        }
        self.max_y_ = float(np.max(y))

        # Stratum sub-models
        strata = self._get_strata(X)
        self.stratum_models_: dict[int, np.ndarray] = {}
        self.stratum_stats_:  dict[int, dict]       = {}

        trained = 0
        for s in np.unique(strata):
            mask = strata == s
            y_s  = y[mask]
            self.stratum_stats_[int(s)] = {
                "mean": float(np.mean(y_s)),
                "std":  float(np.std(y_s) + 1e-8),
            }
            if mask.sum() >= self.min_stratum_size:
                w_s = weights[mask] if weights is not None else None
                self.stratum_models_[int(s)] = self._fit_ridge(
                    X[mask], y_s, self.ridge_alpha, w_s
                )
                trained += 1

        n_total = len(np.unique(strata))
        print(f"  [CAPSR] {trained}/{n_total} strata trained "
              f"(≥{self.min_stratum_size} samples); global fallback for rest.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route each sample to its stratum model; clip to [0, max_train_y]."""
        X      = np.asarray(X, dtype=np.float32)
        strata = self._get_strata(X)
        preds  = np.empty(len(X), dtype=np.float64)

        from collections import defaultdict
        idx_by_stratum: dict[int, list] = defaultdict(list)
        for i, s in enumerate(strata.tolist()):
            idx_by_stratum[s].append(i)

        for s, idxs in idx_by_stratum.items():
            coef = self.stratum_models_.get(s, self.global_coef_)
            ia   = np.array(idxs, dtype=np.int64)
            preds[ia] = X[ia].astype(np.float64) @ coef

        return np.clip(preds, 0.0, self.max_y_).astype(np.float32)

    def predict_deferral_confidence(self, X: np.ndarray) -> np.ndarray:
        """Return confidence ∈ [0, 1] that each job is an above-mean defer candidate.

        confidence = sigmoid(z-score vs stratum mean/std)

        High confidence → job predicted to use above-average GPU-hours for its
        type + priority, making it a strong carbon-deferral candidate.
        """
        X      = np.asarray(X, dtype=np.float32)
        pred   = self.predict(X).astype(np.float64)
        strata = self._get_strata(X)

        means = np.array([
            self.stratum_stats_.get(int(s), self.global_stats_)["mean"]
            for s in strata
        ])
        stds = np.array([
            self.stratum_stats_.get(int(s), self.global_stats_)["std"]
            for s in strata
        ])
        z = (pred - means) / stds
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


# ── Instantiate all six ───────────────────────────────────────────────────────

def all_algorithms(fast: bool = False) -> list:
    """Return all five algorithms.

    fast=True uses smaller hyper-params (fewer trees/epochs) for quick iteration.
    Results are directionally correct but not tuned for production use.
    """
    if fast:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        import lightgbm as lgb

        rf = RandomForest.__new__(RandomForest)
        rf._model = RandomForestRegressor(
            n_estimators=50, max_depth=10, min_samples_leaf=50,
            n_jobs=-1, random_state=42,
        )

        xgb_m = XGBoost.__new__(XGBoost)
        xgb_m._model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, eval_metric="rmse",
            tree_method="hist", device="cpu", random_state=42, verbosity=0,
        )

        lgb_m = LightGBM.__new__(LightGBM)
        lgb_m._model = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=63, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=100, random_state=42, verbose=-1,
        )

        lstm_m = LSTMRegressor(
            hidden_size=64, num_layers=1, dropout=0.0,
            batch_size=8192, max_epochs=10, patience=5,
        )

        capsr_fast = CAPSRRegressor(min_stratum_size=50, ridge_alpha=1.0)
        return [LinearReg(), rf, xgb_m, lgb_m, lstm_m, capsr_fast]

    return [LinearReg(), RandomForest(), XGBoost(), LightGBM(), LSTMRegressor(),
            CAPSRRegressor()]


# ── Smoke test ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test all algorithms on tiny data.")
    parser.parse_args(argv)

    rng = np.random.default_rng(0)
    N, F = 2000, 12
    X  = rng.standard_normal((N, F)).astype(np.float32)
    y  = (X[:, 0] * 2 + X[:, 1] + rng.standard_normal(N) * 0.5).astype(np.float32)
    Xv = rng.standard_normal((400, F)).astype(np.float32)
    yv = (Xv[:, 0] * 2 + Xv[:, 1] + rng.standard_normal(400) * 0.5).astype(np.float32)

    # CAPSR smoke test uses synthetic feat_names with jobtype/priority columns
    capsr_feat_names = [f"feat_jobtype_{j}" for j in range(3)] + \
                       [f"feat_priority_{p}" for p in range(2)] + \
                       [f"feat_x{i}" for i in range(F - 5)]
    # Rebuild X with one-hot structure in first 5 cols
    X_capsr  = X.copy();  X_capsr[:, :3]  = 0; X_capsr[np.arange(N), rng.integers(0,3,N)] = 1
    X_capsr[:, 3:5] = 0;  X_capsr[np.arange(N), 3 + rng.integers(0,2,N)] = 1
    Xv_capsr = Xv.copy(); Xv_capsr[:, :3] = 0; Xv_capsr[np.arange(400), rng.integers(0,3,400)] = 1
    Xv_capsr[:, 3:5] = 0; Xv_capsr[np.arange(400), 3 + rng.integers(0,2,400)] = 1

    for algo in all_algorithms():
        t0 = time.perf_counter()
        if isinstance(algo, CAPSRRegressor):
            algo.feat_names_ = None  # force lazy-load path to be bypassed
            algo._feat_names_arg = capsr_feat_names
            algo.fit(X_capsr, y, Xv_capsr, yv)
            pred = algo.predict(Xv_capsr[:100])
            conf = algo.predict_deferral_confidence(Xv_capsr[:100])
            rmse_val = float(np.sqrt(np.mean((pred - yv[:100]) ** 2)))
            print(f"  {algo.name:20s}  RMSE={rmse_val:.4f}  conf_mean={conf.mean():.3f}  "
                  f"t={time.perf_counter()-t0:.1f}s")
        else:
            algo.fit(X, y, Xv, yv)
            pred = algo.predict(Xv[:100])
            rmse_val = float(np.sqrt(np.mean((pred - yv[:100]) ** 2)))
            print(f"  {algo.name:20s}  RMSE={rmse_val:.4f}  t={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
