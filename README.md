# Climate Adaptive ML

Daily ensemble forecasting system for temperature, precipitation, extreme events, and wind across three distinct climate regimes.

---

## Daily update

<!-- UPDATE_TIME_START -->
**Last update: 21/06/2026 09:53 UTC**
<!-- UPDATE_TIME_END -->

Forecasts are updated automatically every day at **06:00 UTC**.

Latest predictions: [data/predictions/](https://github.com/srxkatsumi/climate-adaptive-ml/tree/main/data/predictions)

<!-- FORECAST_TABLE_START -->
| Model | BCN Min | BCN Max | SP Min | SP Max | MNS Min | MNS Max | Updated | Accuracy D+1 |
|-------|---------|---------|--------|--------|---------|---------|---------|-------------|
| Climatology | 12.9 | 20.8 | 16.0 | 25.4 | 24.7 | 30.3 | 21/06/2026 | 100.0% |
| Random Forest | 22.5 | 30.7 | 11.4 | 22.1 | 24.7 | 29.9 | 21/06/2026 | 75.4% |
| XGBoost | 23.1 | 32.2 | 10.9 | 20.9 | 24.6 | 29.6 | 21/06/2026 | 78.9% |
| LightGBM | 23.8 | 32.4 | 11.1 | 21.8 | 24.2 | 30.1 | 21/06/2026 | 78.9% |
| LSTM | 22.5 | 30.7 | 11.4 | 22.1 | 24.7 | 29.9 | 21/06/2026 | 18.6% |
| ARIMA | 9.3 | 20.8 | 18.1 | 26.8 | 24.6 | 29.9 | 21/06/2026 | 51.4% |
| Adaptive Ensemble | 21.3 | 30.0 | 10.8 | 20.8 | 24.2 | 28.6 | 21/06/2026 | 65.9% |
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
