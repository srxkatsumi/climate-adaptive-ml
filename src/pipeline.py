"""
Daily pipeline — runs every day at 06:00 UTC via GitHub Actions.
1. Fetch fresh forecast from Open-Meteo
2. Train all models (RF, XGBoost, LightGBM, LSTM, ARIMA, Adaptive Ensemble, Climatology)
3. Get D+1 min/max predictions per city per model
4. Compute accuracy from validation split (80/20)
5. Update README table
"""

import logging
import yaml
import numpy as np
import pandas as pd
from datetime import date, datetime, timezone
from pathlib import Path

from src.data.fetcher import fetch_forecast, fetch_last_n_days
from src.data.processor import build_features
from src.models.adaptive import AdaptiveEnsemble
from src.models.baseline import train_static_rf, predict_static_rf
from src.models.climatology import ClimatologyBaseline
from src.models.classical import train_xgb, predict_xgb, train_lgbm, predict_lgbm
from src.models.neural import train_lstm, predict_lstm
from src.models.timeseries_models import train_arima, predict_arima_next

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"

CITY_SHORT = {
    "barcelona": "BCN",
    "sao_paulo": "SP",
    "manaus":    "MNS",
}

MODEL_DISPLAY = {
    "climatology":       "Climatology",
    "random_forest":     "Random Forest",
    "xgboost":           "XGBoost",
    "lightgbm":          "LightGBM",
    "lstm":              "LSTM",
    "arima":             "ARIMA",
    "adaptive_ensemble": "Adaptive Ensemble",
}


def load_cities() -> dict:
    with open(CONFIG_DIR / "cities.yaml") as f:
        return yaml.safe_load(f)["cities"]


def _feat_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _accuracy_pct(y_pred: np.ndarray, y_true: np.ndarray, tol: float = 2.0) -> float:
    mask = ~(np.isnan(y_pred.astype(float)) | np.isnan(y_true.astype(float)))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask].astype(float) - y_true[mask].astype(float)) <= tol) * 100)


def run_city_models(city_key: str, city_cfg: dict) -> dict:
    """
    Train all models for one city, return per-model D+1 forecast and accuracy.
    Returns: {model_name: {"min": float, "max": float, "accuracy": float}}
    """
    log.info(f"  [{city_key}] Fetching historical data...")
    hist_df = fetch_last_n_days(city_cfg, n_days=365)
    features_df = build_features(hist_df)

    log.info(f"  [{city_key}] Fetching D+1 forecast...")
    forecast_raw = fetch_forecast(city_cfg, horizon_days=2)
    forecast_features = build_features(
        pd.concat([hist_df, forecast_raw]).tail(50)
    ).tail(len(forecast_raw))

    split = int(len(features_df) * 0.8)
    feat_cols = _feat_cols(features_df)

    # results[model][target_type] = value
    raw: dict[str, dict] = {m: {"acc_parts": []} for m in MODEL_DISPLAY}

    for target, col in [("min", "temperature_2m_min"), ("max", "temperature_2m_max")]:
        if col not in features_df.columns:
            continue

        X = features_df[feat_cols]
        y = features_df[col]
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # D+1 row aligned to feat_cols
        fc_cols = [c for c in feat_cols if c in forecast_features.columns]
        X_fc = forecast_features[fc_cols].reindex(columns=feat_cols, fill_value=0)
        d1 = X_fc.iloc[[1]] if len(X_fc) > 1 else X_fc.iloc[[0]]

        def _add(model_name: str, pred_d1: float, val_pred: np.ndarray) -> None:
            raw[model_name][target] = round(float(pred_d1), 1)
            acc = _accuracy_pct(val_pred, y_val.values)
            raw[model_name]["acc_parts"].append(acc)

        # ── Climatology ────────────────────────────────────────────────────
        clim = ClimatologyBaseline()
        clim.fit(y)
        _add("climatology",
             float(clim.predict(d1.index)[0]),
             np.array(clim.predict(X_val.index)))

        # ── Random Forest ──────────────────────────────────────────────────
        rf = train_static_rf(X_train, y_train)
        _add("random_forest",
             float(predict_static_rf(rf, d1)[0]),
             predict_static_rf(rf, X_val))

        # ── XGBoost ────────────────────────────────────────────────────────
        xgb = train_xgb(X_train, y_train)
        _add("xgboost",
             float(predict_xgb(xgb, d1)[0]),
             predict_xgb(xgb, X_val))

        # ── LightGBM ───────────────────────────────────────────────────────
        lgbm = train_lgbm(X_train, y_train)
        _add("lightgbm",
             float(predict_lgbm(lgbm, d1)[0]),
             predict_lgbm(lgbm, X_val))

        # ── LSTM ───────────────────────────────────────────────────────────
        lstm = train_lstm(X_train, y_train)
        lstm_val = predict_lstm(lstm, X_val)
        lstm_d1 = predict_lstm(lstm, d1)[0]
        if np.isnan(float(lstm_d1)):
            lstm_d1 = raw["random_forest"].get(target, float("nan"))
        _add("lstm", float(lstm_d1), lstm_val)

        # ── ARIMA (N-step forecast on val, faster than rolling) ────────────
        arima = train_arima(y_train)
        arima_val = predict_arima_next(arima, n_steps=len(y_val))
        arima_d1 = predict_arima_next(arima, n_steps=len(y_val) + 1)[-1]
        _add("arima", float(arima_d1), arima_val)

        # ── Adaptive Ensemble ──────────────────────────────────────────────
        ens = AdaptiveEnsemble()
        ens.fit(X_train, y_train)
        ens.update_weights(X_val, y_val, str(date.today()))
        _add("adaptive_ensemble",
             float(ens.predict(d1)[0]),
             ens.predict(X_val))

    # collapse acc_parts → single accuracy per model
    results = {}
    for model_name, data in raw.items():
        parts = [p for p in data.pop("acc_parts", []) if not np.isnan(p)]
        results[model_name] = {
            **{k: v for k, v in data.items()},
            "accuracy": round(float(np.mean(parts)), 1) if parts else float("nan"),
        }
    return results


def build_summary_df(all_city_results: dict) -> pd.DataFrame:
    """One row per model, columns per city min/max plus mean accuracy."""
    city_keys = list(all_city_results.keys())
    rows = []
    for model_name in MODEL_DISPLAY:
        row: dict = {"model": model_name}
        accs = []
        for city in city_keys:
            data = all_city_results.get(city, {}).get(model_name, {})
            s = CITY_SHORT[city]
            row[f"{s}_min"] = data.get("min", float("nan"))
            row[f"{s}_max"] = data.get("max", float("nan"))
            acc = data.get("accuracy", float("nan"))
            if not np.isnan(acc):
                accs.append(acc)
        row["accuracy"] = round(float(np.mean(accs)), 1) if accs else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def save_daily_summary(df: pd.DataFrame) -> None:
    path = DATA_DIR / "predictions" / "daily_summary.csv"
    today_str = str(date.today())
    df = df.copy()
    df["run_date"] = today_str
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["run_date"] != today_str]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved → {path.name}")


def update_readme_table(df: pd.DataFrame) -> None:
    readme_path = ROOT / "README.md"
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%d/%m/%Y")
    timestamp_str = now_utc.strftime("%d/%m/%Y %H:%M UTC")

    header = (
        "| Model | BCN Min | BCN Max | SP Min | SP Max | MNS Min | MNS Max "
        "| Updated | Accuracy D+1 |"
    )
    sep = (
        "|-------|---------|---------|--------|--------|---------|---------|"
        "---------|-------------|"
    )
    lines = [header, sep]

    def fmt(v):
        try:
            return f"{float(v):.1f}" if not np.isnan(float(v)) else "—"
        except Exception:
            return "—"

    for _, r in df.iterrows():
        acc = f"{r['accuracy']:.1f}%" if not np.isnan(float(r["accuracy"])) else "—"
        lines.append(
            f"| {MODEL_DISPLAY[r['model']]} "
            f"| {fmt(r['BCN_min'])} | {fmt(r['BCN_max'])} "
            f"| {fmt(r['SP_min'])} | {fmt(r['SP_max'])} "
            f"| {fmt(r['MNS_min'])} | {fmt(r['MNS_max'])} "
            f"| {today_str} | {acc} |"
        )

    content = readme_path.read_text()

    # timestamp
    ts, te = "<!-- UPDATE_TIME_START -->", "<!-- UPDATE_TIME_END -->"
    content = f"{content.split(ts)[0]}{ts}\n**Last update: {timestamp_str}**\n{te}{content.split(te)[1]}"

    # table
    fs, fe = "<!-- FORECAST_TABLE_START -->", "<!-- FORECAST_TABLE_END -->"
    table_block = "\n".join(lines)
    readme_path.write_text(
        f"{content.split(fs)[0]}{fs}\n{table_block}\n{fe}{content.split(fe)[1]}"
    )
    log.info("  README updated")


def fill_observations() -> None:
    cities = load_cities()
    today = date.today()
    for city_key, city_cfg in cities.items():
        pred_path = DATA_DIR / "predictions" / f"{city_key}_predictions.csv"
        if not pred_path.exists():
            continue
        df = pd.read_csv(pred_path)
        missing = df[
            df["observed"].isna() &
            (pd.to_datetime(df["target_date"]).dt.date < today)
        ]
        if missing.empty:
            continue
        past = pd.to_datetime(missing["target_date"]).dt.date
        try:
            hist = fetch_last_n_days(city_cfg, n_days=(today - min(past)).days + 2)
            hr = hist.reset_index()
            hr["date_only"] = hr["date"].dt.date
            for idx, row in df.iterrows():
                if pd.notna(df.at[idx, "observed"]):
                    continue
                td = pd.to_datetime(row["target_date"]).date()
                if td >= today:
                    continue
                m = hr[hr["date_only"] == td]
                if not m.empty:
                    df.at[idx, "observed"] = round(float(m["temperature_2m_mean"].values[0]), 2)
            df.to_csv(pred_path, index=False)
        except Exception as e:
            log.warning(f"  fill_observations {city_key}: {e}")


def main() -> None:
    log.info("=== Climate Adaptive ML — Daily Pipeline ===")
    log.info(f"Run date: {date.today()}")

    log.info("Step 1: Back-filling observations...")
    fill_observations()

    cities = load_cities()
    all_city_results: dict = {}

    log.info("Step 2: Training all models per city...")
    for city_key, city_cfg in cities.items():
        log.info(f"Processing {city_cfg['name']}...")
        try:
            all_city_results[city_key] = run_city_models(city_key, city_cfg)
        except Exception as e:
            log.error(f"Failed {city_key}: {e}", exc_info=True)
            all_city_results[city_key] = {}

    log.info("Step 3: Building summary table...")
    summary_df = build_summary_df(all_city_results)

    log.info("Step 4: Saving daily_summary.csv...")
    save_daily_summary(summary_df)

    log.info("Step 5: Updating README...")
    update_readme_table(summary_df)

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
