"""
Feature engineering for climate forecasting.
Transforms raw Open-Meteo data into ML-ready features.
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, lag_days: int = 14) -> pd.DataFrame:
    """Add lag features, rolling statistics, and climatological features."""
    df = df.copy()

    target_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
    ]

    for var in target_vars:
        if var not in df.columns:
            continue
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"{var}_lag{lag}"] = df[var].shift(lag)
        for window in [3, 7, 14]:
            df[f"{var}_roll{window}_mean"] = df[var].shift(1).rolling(window).mean()
            df[f"{var}_roll{window}_std"] = df[var].shift(1).rolling(window).std()

    df = _add_calendar_features(df)
    df = _add_extreme_flags(df)

    return df.dropna()


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_year"] = df.index.day_of_year
    df["month"] = df.index.month
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _add_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "temperature_2m_max" in df.columns:
        q90 = df["temperature_2m_max"].quantile(0.90)
        df["heat_flag"] = (df["temperature_2m_max"] > q90).astype(int)

    if "precipitation_sum" in df.columns:
        df["heavy_rain_flag"] = (df["precipitation_sum"] > 50).astype(int)
        df["dry_flag"] = (df["precipitation_sum"] < 1).astype(int)

    return df


def compute_climatology(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Daily climatology (30-year mean by day-of-year) for the persistence baseline."""
    clim = (
        df.groupby(df.index.day_of_year)[variable]
        .mean()
        .rename("climatology_mean")
    )
    return clim
