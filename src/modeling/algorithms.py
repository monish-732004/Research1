"""Five regression algorithms for GPU-hours prediction.

Each class is sklearn-compatible (.fit / .predict / .name).
Training on the full 40M-row dataset takes ~90 min; use --fast for 10% subsample.

Algorithm 5 note: The LSTM treats each job as a sequence of length 1.
Without a true time series this reduces to a 2-layer MLP — performance
reflects deep learning on tabular data, not sequence modeling.

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


# ── Instantiate all five ──────────────────────────────────────────────────────

def all_algorithms() -> list:
    return [LinearReg(), RandomForest(), XGBoost(), LightGBM(), LSTMRegressor()]


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

    for algo in all_algorithms():
        t0 = time.perf_counter()
        algo.fit(X, y, Xv, yv)
        pred = algo.predict(Xv[:100])
        rmse = float(np.sqrt(np.mean((pred - yv[:100]) ** 2)))
        print(f"  {algo.name:20s}  RMSE={rmse:.4f}  t={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
