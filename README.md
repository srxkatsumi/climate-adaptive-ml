# Climate Adaptive ML

Daily ensemble forecasting system for temperature, precipitation, extreme events, and wind across three distinct climate regimes.

---

## Daily update

<!-- UPDATE_TIME_START -->
**Last update: 24/06/2026 09:31 UTC**
<!-- UPDATE_TIME_END -->

Forecasts are updated automatically every day at **06:00 UTC**.

Latest predictions: [data/predictions/](https://github.com/srxkatsumi/climate-adaptive-ml/tree/main/data/predictions)

<!-- FORECAST_TABLE_START -->
| Model | BCN Min | BCN Max | SP Min | SP Max | MNS Min | MNS Max | Updated | Accuracy D+1 |
|-------|---------|---------|--------|--------|---------|---------|---------|-------------|
| Climatology | 12.9 | 20.8 | 16.0 | 25.5 | 24.7 | 30.3 | 24/06/2026 | 100.0% |
| Random Forest | 23.0 | 31.5 | 12.0 | 21.9 | 24.7 | 30.1 | 24/06/2026 | 76.3% |
| XGBoost | 23.5 | 33.0 | 11.3 | 19.8 | 24.9 | 29.5 | 24/06/2026 | 76.1% |
| LightGBM | 23.3 | 32.5 | 10.7 | 18.7 | 24.7 | 29.7 | 24/06/2026 | 77.0% |
| LSTM | 23.0 | 31.5 | 12.0 | 21.9 | 24.7 | 30.1 | 24/06/2026 | 20.5% |
| ARIMA | 10.3 | 18.6 | 17.9 | 27.1 | 24.6 | 29.8 | 24/06/2026 | 49.5% |
| Adaptive Ensemble | 21.6 | 30.3 | 11.0 | 19.6 | 24.4 | 28.7 | 24/06/2026 | 67.6% |
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
