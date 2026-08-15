# Climate Adaptive ML

Daily ensemble forecasting system for temperature, precipitation, extreme events, and wind across three distinct climate regimes.

---

## Daily update

<!-- UPDATE_TIME_START -->
**Last update: 15/08/2026 06:39 UTC**
<!-- UPDATE_TIME_END -->

Forecasts are updated automatically every day at **06:00 UTC**.

Latest predictions: [data/predictions/](https://github.com/srxkatsumi/climate-adaptive-ml/tree/main/data/predictions)

<!-- FORECAST_TABLE_START -->
| Model | BCN Min | BCN Max | SP Min | SP Max | MNS Min | MNS Max | Updated | Accuracy D+1 |
|-------|---------|---------|--------|--------|---------|---------|---------|-------------|
| Climatology | 13.4 | 21.2 | 16.2 | 25.6 | 24.8 | 30.4 | 15/08/2026 | 100.0% |
| Random Forest | 19.6 | 26.8 | 16.2 | 28.0 | 26.0 | 33.5 | 15/08/2026 | 57.3% |
| XGBoost | 19.5 | 26.5 | 16.2 | 31.7 | 25.9 | 34.1 | 15/08/2026 | 58.2% |
| LightGBM | 20.0 | 27.8 | 16.6 | 31.6 | 26.2 | 33.9 | 15/08/2026 | 58.5% |
| LSTM | 19.6 | 26.8 | 16.2 | 28.0 | 26.0 | 33.5 | 15/08/2026 | 10.9% |
| ARIMA | 18.8 | 26.2 | 14.1 | 22.1 | 24.3 | 29.1 | 15/08/2026 | 38.8% |
| Adaptive Ensemble | 20.2 | 28.0 | 15.4 | 26.3 | 25.4 | 31.8 | 15/08/2026 | 49.5% |
<!-- FORECAST_TABLE_END -->

---

## Objective

Test whether an **adaptive ensemble with temporal decay** outperforms static ML baselines and climatology persistence across multiple climate zones.

**Null hypothesis:** adaptive ensemble = static ensemble  
**Alternative:** adaptive ensemble > static ensemble (statistically significant)

---

## Climate zones studied

| City | Country | Köppen | Regime |
|------|---------|--------|--------|
| Barcelona | Spain | Csa | Mediterranean |
| São Paulo | Brazil | Cfa | Tropical humid |
| Manaus | Brazil | Af | Equatorial |

---

## Three models compared

| # | Model | Role |
|---|-------|------|
| 1 | Climatology persistence | Minimum benchmark — any model must beat this |
| 2 | Static Random Forest | Simple ML state of the art — fixed weights |
| 3 | **Adaptive Ensemble** (RF + GB + SGD) | Temporal decay + weight adaptation |

---

## Variables forecast

| Variable | Horizons | Metrics |
|----------|----------|---------|
| Temperature (max / min / mean) | D+1 to D+15 | MAE, RMSE, ACC |
| Precipitation | D+1 to D+15 | MAE, RMSE, BSS, HSS |
| Extreme events (heat wave, drought, heavy rain) | Binary | HSS, BSS, POD, FAR |
| Wind speed | D+1 to D+7 | MAE, RMSE |

Statistical significance tested with the **Diebold-Mariano test**.

---

## Repository structure

```
clima/
├── config/
│   ├── cities.yaml          # city coordinates and climate metadata
│   └── variables.yaml       # variable definitions and thresholds
├── data/
│   ├── predictions/         # daily forecast log (committed)
│   ├── raw/                 # API responses (gitignored)
│   └── processed/           # feature files (gitignored)
├── src/
│   ├── data/
│   │   ├── fetcher.py       # Open-Meteo API calls
│   │   └── processor.py     # feature engineering
│   ├── models/
│   │   ├── climatology.py   # Model 1 — persistence baseline
│   │   ├── baseline.py      # Model 2 — static RF
│   │   └── adaptive.py      # Model 3 — adaptive ensemble
│   ├── evaluation/
│   │   ├── metrics.py       # MAE, RMSE, ACC, BSS, HSS
│   │   └── statistical.py   # Diebold-Mariano test
│   └── pipeline.py          # main daily runner
├── notebooks/
│   └── exploration/         # EDA and analysis notebooks
├── .github/
│   └── workflows/
│       └── daily_update.yml # daily 06:00 UTC automation
└── requirements.txt
```

---

## Running locally

```bash
pip install -r requirements.txt
python -m src.pipeline
```

---

## Data source

[Open-Meteo](https://open-meteo.com) — free, no API key required.  
Historical archive: ERA5 reanalysis via Open-Meteo Historical API.

---

## Timeline

| Phase | Period | Goal |
|-------|--------|------|
| 1 — Data pipeline | 2026 Q3 | Daily predictions running for all 3 cities |
| 2 — Validation accumulation | 2026 Q4–2027 Q2 | Minimum 6 months clean validations |
| 3 — Analysis | 2027 Q3–Q4 | Statistical comparison across climate regimes |
