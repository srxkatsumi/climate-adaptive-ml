"""
Classical gradient-boosted trees — adapted for temperature regression.
Ported from smart-wallet-ml/models/classical.py (classifier → regressor).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


def _get_feature_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype in (np.float64, np.float32, int)]


def train_xgb(X: pd.DataFrame, y: pd.Series) -> dict:
    if not _HAS_XGB:
        return {"type": "xgb", "unavailable": True}
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_sc, y.values)
    return {"type": "xgb", "model": model, "scaler": scaler}


def predict_xgb(state: dict, X: pd.DataFrame) -> np.ndarray:
    if state.get("unavailable"):
        return np.full(len(X), np.nan)
    return state["model"].predict(state["scaler"].transform(X))


def train_lgbm(X: pd.DataFrame, y: pd.Series) -> dict:
    if not _HAS_LGBM:
        return {"type": "lgbm", "unavailable": True}
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)
    model = LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_sc, y.values)
    return {"type": "lgbm", "model": model, "scaler": scaler}


def predict_lgbm(state: dict, X: pd.DataFrame) -> np.ndarray:
    if state.get("unavailable"):
        return np.full(len(X), np.nan)
    return state["model"].predict(state["scaler"].transform(X))
