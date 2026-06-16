"""
ARIMA time-series baseline for temperature forecasting.
Ported from smart-wallet-ml/models/timeseries.py.
Fits ARIMA(5,1,1) on the historical target series and forecasts D+1.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


def train_arima(y: pd.Series) -> dict:
    if not _HAS_STATSMODELS:
        return {"type": "arima", "unavailable": True}
    try:
        result = ARIMA(y.values.astype(float), order=(5, 1, 1)).fit()
        return {"type": "arima", "model": result}
    except Exception as e:
        return {"type": "arima", "unavailable": True, "reason": str(e)}


def predict_arima_next(state: dict, n_steps: int = 1) -> np.ndarray:
    if state.get("unavailable"):
        return np.full(n_steps, np.nan)
    try:
        return np.array(state["model"].forecast(steps=n_steps))
    except Exception:
        return np.full(n_steps, np.nan)
