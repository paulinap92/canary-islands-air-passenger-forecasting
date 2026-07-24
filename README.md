# ✈️ Canary Islands Air Passenger Forecasting Dashboard

Interactive Streamlit dashboard for analysis and forecasting of Canary Islands air-passenger traffic using official WebTenerife data.

## What the project does

- ingests newly published monthly passenger data,
- rebuilds chronological lag and rolling-window features,
- generates production forecasts with persisted XGBoost and LSTM models,
- preserves an append-only history of forecasts,
- trains separate model candidates on a controlled schedule,
- presents historical data and future predictions in Streamlit.

## Production data and forecast workflow

The weekly GitHub Actions workflow can also be started manually. It:

1. checks WebTenerife for the next monthly XLSX,
2. updates historical CSV files when a new month exists,
3. rebuilds XGBoost features,
4. loads the current production XGBoost and LSTM artifacts,
5. generates forecasts without calling `fit()`,
6. appends the new predictions to `forecast_history.csv`,
7. commits changed production outputs.

Forecasting always covers at least 12 future months. The current workflow also extends the output through December 2027; once the rolling 12-month horizon reaches further than that date, the longer rolling horizon wins.

```bash
python monthly_update.py
```

The monthly workflow never retrains or replaces a model.

## Production artifacts

```text
models/xgb_best.pkl
models/lstm_best.h5
models/scaler_y.pkl
```

- `model_final_xgb.py` loads the persisted XGBoost model and performs inference only.
- `model_final_lstm.py` loads the persisted LSTM model and scaler and performs inference only.
- `forecast_horizon.py` provides the shared rolling forecast-horizon logic.

## Forecast history

`forecast_history.csv` records each genuine forward-looking production prediction with:

- generation timestamp,
- forecast origin,
- target month,
- horizon in months,
- model name,
- model-artifact version hash,
- predicted passenger value.

Repeated runs with the same origin, target, model, and version are deduplicated. Historical forecasts are not reconstructed retroactively.

## Model development and retraining

The root-level `my_models_trials.ipynb` documents the original model comparison, architecture experiments, model selection, and first persisted artifacts. It is not executed during routine production updates.

Quarterly retraining runs on 5 January, April, July, and October and can also be triggered manually. For both XGBoost and LSTM it:

1. evaluates a candidate trained on all available history,
2. evaluates a candidate using only data from January 2022 onward,
3. compares both variants on the same chronological 12-month holdout,
4. selects the lower-RMSE data scope,
5. retrains the selected candidate on all available data within that scope,
6. stores the candidate and its comparison report as a workflow artifact.

```bash
python training/retrain_models.py
python training/train_lstm.py
```

LSTM parameters can be overridden:

```bash
python training/train_lstm.py --units 32 --epochs 150 --batch-size 8
```

Candidate outputs remain separate:

```text
models/xgb_candidate.pkl
models/xgb_candidate_metrics.json
models/lstm_candidate.keras
models/scaler_y_candidate.pkl
models/lstm_candidate_metrics.json
```

No candidate automatically replaces a production model. Promotion requires human review of the holdout metrics and forecast behaviour.

## Automation and validation

- `CI` checks critical Python errors, compiles the source, and validates dashboard datasets.
- `Forecast CI` runs both persisted production models and validates the generated horizon, nulls, negative predictions, and forecast-history output.
- `Build branch forecast preview` creates reviewable forecast CSVs on the forecasting branch.
- `Update data and forecasts` performs the scheduled production data refresh after merge.
- `Train model candidates` performs scheduled candidate evaluation and training.

## Data flow

```text
WebTenerife XLSX
      ↓
download_agent.py
      ↓
result.csv + result_total.csv
      ↓
build_features()
      ↓
lag and rolling-feature datasets
      ↓
persisted XGBoost and LSTM inference
      ↓
forecast CSVs + forecast_history.csv
      ↓
Streamlit dashboard
```

## Project structure

```text
├── main.py
├── monthly_update.py
├── download_agent.py
├── forecast_horizon.py
├── forecast_history.py
├── model_final_xgb.py
├── model_final_lstm.py
├── my_models_trials.ipynb
├── training/
│   ├── retrain_models.py
│   └── train_lstm.py
├── models/
├── data/
├── forecast/
├── charts/
├── kpi/
└── ui/
```

## Run the dashboard

```bash
pipenv install
streamlit run main.py
```

## Data source

WebTenerife — Air Traffic Statistics.

## License

Educational and analytical use.
