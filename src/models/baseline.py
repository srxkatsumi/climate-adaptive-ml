"""
Model 1 — Static Random Forest baseline.
Fixed weights, no temporal adaptation. Represents simple ML state of the art.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


def train_static_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_static_rf(
    model: RandomForestRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    return model.predict(X)
