"""
Open-Meteo API fetcher — historical and forecast data.
Docs: https://open-meteo.com/en/docs
"""

import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

TEMPERATURE_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
]

PRECIPITATION_VARS = [
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
]

WIND_VARS = [
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
]

ALL_DAILY_VARS = TEMPERATURE_VARS + PRECIPITATION_VARS + WIND_VARS


def fetch_forecast(city_config: dict, horizon_days: int = 15) -> pd.DataFrame:
    """Download forecast for the next `horizon_days` days from Open-Meteo."""
    params = {
        "latitude": city_config["latitude"],
        "longitude": city_config["longitude"],
        "daily": ",".join(ALL_DAILY_VARS),
        "forecast_days": horizon_days,
        "timezone": city_config["timezone"],
    }
    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    return _parse_daily_response(response.json())


def fetch_historical(
    city_config: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Download historical observations from Open-Meteo ERA5 archive."""
    params = {
        "latitude": city_config["latitude"],
        "longitude": city_config["longitude"],
        "daily": ",".join(ALL_DAILY_VARS),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": city_config["timezone"],
    }
    response = requests.get(OPEN_METEO_HISTORICAL_URL, params=params, timeout=60)
    response.raise_for_status()
    return _parse_daily_response(response.json())


def fetch_last_n_days(city_config: dict, n_days: int = 365) -> pd.DataFrame:
    """Download the last `n_days` of historical data."""
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=n_days)
    return fetch_historical(city_config, start_date, end_date)


def _parse_daily_response(response_json: dict) -> pd.DataFrame:
    daily = response_json.get("daily", {})
    if not daily:
        raise ValueError("Empty daily data in Open-Meteo response")
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.set_index("date")
    return df
