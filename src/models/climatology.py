"""
Model 3 — Climatology persistence baseline.
Forecast = historical mean for that day-of-year.
Minimum benchmark: any model must beat this to have scientific value.
"""

import numpy as np
import pandas as pd


class ClimatologyBaseline:
    """Predicts the historical daily mean for each day-of-year."""

    def __init__(self):
        self._clim: pd.Series | None = None

    def fit(self, y: pd.Series) -> "ClimatologyBaseline":
        self._clim = y.groupby(y.index.day_of_year).mean()
        return self

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if self._clim is None:
            raise RuntimeError("Must call fit() before predict()")
        doy = dates.day_of_year
        return np.array([self._clim.get(d, self._clim.mean()) for d in doy])

    def predict_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(self.predict(dates), index=dates, name="climatology_forecast")
