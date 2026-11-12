# Ozgen-GBQR: FluSight influenza hospital admission forecasts

Code for the `Ozgen-GBQR` model in the CDC
[FluSight 2026-27 forecast hub](https://github.com/cdcepi/FluSight-forecast-hub). It produces
weekly quantile forecasts of `wk inc flu hosp` (NHSN confirmed influenza hospital admissions)
for horizons -1 to 3 and all hub locations.

## Method (v1.0, 2026-11-12)

- **Scale.** z = (admissions per 100k) ** 0.25.
- **Global gradient-boosted quantile regression.** For the last reported week t and steps
  s = 1..4, one `HistGradientBoostingRegressor` per quantile level (quantile loss) predicts
  z[t+s] - z[t]. Features: level, 1-, 2- and 4-week differences, 4-week mean, slope and spread,
  52-week maximum, week of the season, log population and s. All locations are pooled. Each
  model is fit on the data vintage available at the forecast date only.
- **Two feature sets, averaged.** The model above (weight 0.7) and the same model without the
  season-week and 52-week-maximum features (weight 0.3), averaged quantile by quantile.
- **Backfill.** The last two reported weeks are multiplied by per-location factors: the median
  ratio of later to first-reported values over the previous 12 rounds' vintages.
- **Calibration.** Each quantile's distance from the median is widened by 1.3 on the log scale.
- Horizon -1 is the backfill-adjusted last report with a narrow band.

## Backtest

89 past FluSight rounds (2023-10-14 to 2026-10-31). Each forecast uses the hub's data vintage
dated on or before that round's Wednesday deadline, cut at the week ending one week before the
reference date. Scores are against the latest vintage. Relative WIS is mean WIS divided by the
mean WIS of `FluSight-baseline` on the same forecasts (horizons 0-3).

| season | rounds | Ozgen-GBQR | FluSight-ensemble |
|---|---|---|---|
| 2023-24 | 30 | 0.79 | 0.73 |
| 2024-25 | 27 | 0.72 | 0.67 |
| 2025-26 | 28 | 0.59 | 0.66 |
| 2026-27 (to 10-31) | 4 | 0.48 | 0.45 |
| all | 89 | 0.69 | 0.68 |

Coverage of the central 50% and 95% intervals: 0.49 and 0.92.

The feature sets, the weights, the backfill correction and the widening factor were chosen on
these backtests, so the table is not an out-of-sample estimate. The 2023-24 rounds had less
than two seasons of training history.

## Use

```sh
git clone --depth 1 https://github.com/cdcepi/FluSight-forecast-hub.git hub
python src/backtest.py hub results gbqr_full_bf 2023-10-14 10    # one component
python src/forecast.py hub 2026-11-14 2026-11-14-Ozgen-GBQR.csv  # weekly submission
python src/validate.py hub model-output/Ozgen-GBQR/2026-11-14-Ozgen-GBQR.csv
```

`src/fetch_nhsn.py` builds the newest vintage directly from data.cdc.gov (dataset mpgq-jmmr)
when the hub's target data is not yet updated.

Requirements: Python 3.12, numpy, pandas, scikit-learn, joblib.

## License

Code: MIT. Forecasts submitted to the hub: CC-BY-4.0.
