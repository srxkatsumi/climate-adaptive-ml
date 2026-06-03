"""
Daily pipeline — runs every day at 06:00 UTC via GitHub Actions.
1. Fetch fresh forecast from Open-Meteo
2. Validate yesterday's predictions against observed values
3. Update adaptive ensemble weights
4. Append new forecasts to predictions log
"""

import logging
import yaml
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from src.data.fetcher import fetch_forecast, fetch_last_n_days
from src.data.processor import build_features, compute_climatology
from src.models.adaptive import AdaptiveEnsemble
from src.models.baseline import train_static_rf, predict_static_rf
from src.models.climatology import ClimatologyBaseline
from src.evaluation.metrics import full_scorecard

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


def load_cities() -> dict:
    with open(CONFIG_DIR / "cities.yaml") as f:
        return yaml.safe_load(f)["cities"]


def run_city(city_key: str, city_cfg: dict) -> None:
    log.info(f"Processing {city_cfg['name']} ({city_cfg['climate_type']})")

    predictions_path = DATA_DIR / "predictions" / f"{city_key}_predictions.csv"

    log.info("  Fetching historical data (last 365 days)...")
    hist_df = fetch_last_n_days(city_cfg, n_days=365)

    log.info("  Building features...")
    features_df = build_features(hist_df)

    target = "temperature_2m_mean"
    feature_cols = [c for c in features_df.columns if target not in c or "lag" in c or "roll" in c]
    feature_cols = [c for c in feature_cols if c not in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]]

    X = features_df[feature_cols]
    y = features_df[target]

    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    log.info("  Training models...")
    ensemble = AdaptiveEnsemble()
    ensemble.fit(X_train, y_train)

    rf_static = train_static_rf(X_train, y_train)

    clim_model = ClimatologyBaseline()
    clim_model.fit(y)

    log.info("  Updating adaptive weights...")
    weight_record = ensemble.update_weights(X_val, y_val, str(date.today()))
    log.info(f"  Weights: RF={weight_record['w_rf']:.3f} GB={weight_record['w_gb']:.3f} SGD={weight_record['w_sgd']:.3f}")

    log.info("  Fetching new forecast...")
    forecast_df = fetch_forecast(city_cfg, horizon_days=15)

    forecast_features = build_features(
        pd.concat([hist_df, forecast_df]).tail(50)
    ).tail(len(forecast_df))

    if not forecast_features.empty:
        X_fc = forecast_features[[c for c in feature_cols if c in forecast_features.columns]]
        adaptive_preds = ensemble.predict(X_fc) if not X_fc.empty else []
        rf_preds = predict_static_rf(rf_static, X_fc) if not X_fc.empty else []
        clim_preds = clim_model.predict(forecast_df.index)

        rows = []
        for i, dt in enumerate(forecast_df.index):
            rows.append({
                "city": city_key,
                "run_date": str(date.today()),
                "target_date": str(dt.date()),
                "horizon_days": (dt.date() - date.today()).days,
                "variable": target,
                "forecast_adaptive": round(float(adaptive_preds[i]), 2) if len(adaptive_preds) > i else None,
                "forecast_rf_static": round(float(rf_preds[i]), 2) if len(rf_preds) > i else None,
                "forecast_climatology": round(float(clim_preds[i]), 2),
                "w_rf": weight_record["w_rf"],
                "w_gb": weight_record["w_gb"],
                "w_sgd": weight_record["w_sgd"],
                "observed": None,
            })

        new_rows = pd.DataFrame(rows)

        if predictions_path.exists():
            existing = pd.read_csv(predictions_path)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_rows

        combined.to_csv(predictions_path, index=False)
        log.info(f"  Saved {len(new_rows)} forecast rows → {predictions_path.name}")

    log.info(f"  Done: {city_cfg['name']}")


def fill_observations() -> None:
    """
    Back-fill observed values for past target_dates that have arrived.
    Runs before forecasting so validation is always up-to-date.
    """
    cities = load_cities()
    today = date.today()

    for city_key, city_cfg in cities.items():
        predictions_path = DATA_DIR / "predictions" / f"{city_key}_predictions.csv"
        if not predictions_path.exists():
            continue

        df = pd.read_csv(predictions_path)
        missing_obs = df[
            (df["observed"].isna()) &
            (pd.to_datetime(df["target_date"]).dt.date < today)
        ]

        if missing_obs.empty:
            continue

        past_dates = pd.to_datetime(missing_obs["target_date"]).dt.date
        start = min(past_dates)
        end = min(max(past_dates), today - timedelta(days=1))

        try:
            hist = fetch_last_n_days(city_cfg, n_days=(today - start).days + 2)
            hist_indexed = hist.reset_index()
            hist_indexed["date_only"] = hist_indexed["date"].dt.date

            for idx, row in df.iterrows():
                if pd.notna(df.at[idx, "observed"]):
                    continue
                td = pd.to_datetime(row["target_date"]).date()
                if td >= today:
                    continue
                match = hist_indexed[hist_indexed["date_only"] == td]
                if not match.empty:
                    df.at[idx, "observed"] = round(float(match["temperature_2m_mean"].values[0]), 2)

            df.to_csv(predictions_path, index=False)
            log.info(f"  {city_key}: filled observations up to {end}")

        except Exception as e:
            log.warning(f"  Could not fill observations for {city_key}: {e}")


def main() -> None:
    log.info("=== Climate Adaptive ML — Daily Pipeline ===")
    log.info(f"Run date: {date.today()}")

    log.info("Step 1: Filling past observations...")
    fill_observations()

    cities = load_cities()
    for city_key, city_cfg in cities.items():
        try:
            run_city(city_key, city_cfg)
        except Exception as e:
            log.error(f"Failed for {city_key}: {e}")

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
