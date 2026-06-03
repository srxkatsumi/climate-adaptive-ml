"""
Model 2 — Adaptive ensemble (the scientific contribution).
RF + GradientBoosting + SGD with temporal decay and weight rebalancing.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


class AdaptiveEnsemble:
    """
    Weighted ensemble with temporal decay and performance-based weight adaptation.
    Weights update after each validation cycle using recent MAE performance.
    """

    def __init__(self, decay_factor: float = 0.95, min_weight: float = 0.05):
        self.decay_factor = decay_factor
        self.min_weight = min_weight

        self.rf = RandomForestRegressor(
            n_estimators=200, max_features="sqrt", min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        self.gb = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            random_state=42
        )
        self.sgd = SGDRegressor(
            loss="huber", max_iter=2000, tol=1e-3, random_state=42
        )
        self._scaler = StandardScaler()

        self._weights = np.array([1/3, 1/3, 1/3])
        self._weight_history: list[dict] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveEnsemble":
        X_scaled = self._scaler.fit_transform(X)
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        self.sgd.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        preds = np.column_stack([
            self.rf.predict(X),
            self.gb.predict(X),
            self.sgd.predict(X_scaled),
        ])
        return preds @ self._weights

    def update_weights(self, X_val: pd.DataFrame, y_val: pd.Series, run_date: str) -> dict:
        """Recompute weights based on recent validation MAE (lower MAE = higher weight)."""
        X_scaled = self._scaler.transform(X_val)
        maes = np.array([
            np.mean(np.abs(self.rf.predict(X_val) - y_val)),
            np.mean(np.abs(self.gb.predict(X_val) - y_val)),
            np.mean(np.abs(self.sgd.predict(X_scaled) - y_val)),
        ])

        inverse = 1.0 / (maes + 1e-9)
        raw_weights = inverse / inverse.sum()
        new_weights = np.maximum(raw_weights, self.min_weight)
        new_weights = new_weights / new_weights.sum()

        self._weights = (
            self.decay_factor * self._weights
            + (1 - self.decay_factor) * new_weights
        )
        self._weights = self._weights / self._weights.sum()

        record = {
            "date": run_date,
            "w_rf": round(float(self._weights[0]), 4),
            "w_gb": round(float(self._weights[1]), 4),
            "w_sgd": round(float(self._weights[2]), 4),
            "mae_rf": round(float(maes[0]), 4),
            "mae_gb": round(float(maes[1]), 4),
            "mae_sgd": round(float(maes[2]), 4),
        }
        self._weight_history.append(record)
        return record

    def confidence_score(self, X: pd.DataFrame, y_actual: pd.Series) -> float:
        """
        Spearman ρ between predicted rank and actual rank.
        Measures forecast ordering quality — used as confidence indicator.
        """
        y_pred = self.predict(X)
        rho, _ = spearmanr(y_pred, y_actual)
        return float(rho)

    @property
    def weights(self) -> dict:
        return {"rf": self._weights[0], "gb": self._weights[1], "sgd": self._weights[2]}
