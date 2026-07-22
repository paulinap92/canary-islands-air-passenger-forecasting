# ✈️ Canary Islands Air Passenger Forecasting Dashboard

Interactive Streamlit dashboard for analysis and forecasting of air passenger traffic in the Canary Islands using official WebTenerife data.

## Project overview

The project combines:

- monthly ingestion of official passenger data,
- historical analysis and visualisation,
- feature engineering for monthly time series,
- XGBoost and LSTM forecasting,
- persisted production models,
- a Streamlit dashboard.

## Model development lifecycle

### 1. Initial model selection and first training

`notebooks/my_models_trials.ipynb` documents the original experimental phase of the project. It was used to:

- compare classical regression models,
- run time-series validation and holdout evaluation,
- test XGBoost, LSTM, GRU, and Transformer variants,
- select the final XGBoost and LSTM approaches,
- create the first persisted production artifacts.

This notebook is an **initial model-selection and training record**. It is not intended to run during every monthly data update.

### 2. Monthly update without retraining

Run:

```bash
python monthly_update.py
```

This workflow:

1. downloads and processes the next available monthly XLSX file,
2. updates the historical CSV datasets,
3. rebuilds lag and rolling-window features,
4. loads the existing persisted XGBoost and LSTM models,
5. generates a fresh 12-month forecast.

It does **not** retrain either model.

### 3. Explicit retraining

Run retraining separately:

```bash
python training/retrain_models.py
```

The retraining workflow creates a new model candidate and reports validation metrics. It does not silently replace the production model. The candidate should be reviewed before promotion.

The current extracted retraining script covers XGBoost. The original LSTM training and architecture comparison remain documented in `notebooks/my_models_trials.ipynb` and should be extracted into a dedicated training module before LSTM retraining is automated.

## Production inference

- `model_final_xgb.py` loads `models/xgb_best.pkl` and performs inference only.
- `model_final_lstm.py` loads `models/lstm_best.h5` and `models/scaler_y.pkl` and performs inference only.

The forecasting horizon is relative to the latest historical month rather than fixed to a calendar date.

## Model artifacts

```text
models/
├── xgb_best.pkl
├── lstm_best.h5
└── scaler_y.pkl
```

Retraining creates candidate artifacts first, for example:

```text
models/xgb_candidate.pkl
```

## Data pipeline

```text
new monthly XLSX
        ↓
download_agent.py
        ↓
result.csv + result_total.csv
        ↓
build_features()
        ↓
lag/rolling feature datasets
        ↓
model inference
        ↓
forecast CSV files
        ↓
Streamlit dashboard
```

## Project structure

```text
├── main.py
├── monthly_update.py
├── download_agent.py
├── model_final_xgb.py
├── model_final_lstm.py
├── training/
│   └── retrain_models.py
├── notebooks/
│   ├── README.md
│   ├── my_models_trials.ipynb
│   ├── prepare_data_for_model.ipynb
│   └── data_processing.ipynb
├── models/
├── data/
├── charts/
├── forecast/
├── kpi/
└── ui/
```

## Data source

WebTenerife — Air Traffic Statistics.

## Run the dashboard

```bash
pipenv install
streamlit run main.py
```

## License

Educational and analytical use.
