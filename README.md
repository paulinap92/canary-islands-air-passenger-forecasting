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

Retraining is deliberately separate from the monthly update.

#### XGBoost candidate

```bash
python training/retrain_models.py
```

This trains and evaluates a new XGBoost candidate and saves:

```text
models/xgb_candidate.pkl
```

#### LSTM candidate

```bash
python training/train_lstm.py
```

Optional parameters:

```bash
python training/train_lstm.py --units 32 --epochs 300 --batch-size 8
```

This keeps the selected LSTM architecture from the initial notebook, performs a chronological 12-month holdout evaluation, and saves candidate artifacts:

```text
models/lstm_candidate.keras
models/scaler_y_candidate.pkl
models/lstm_candidate_metrics.json
```

Neither retraining script silently replaces the production model. Candidate metrics must be reviewed before promotion.

## Production inference

- `model_final_xgb.py` loads `models/xgb_best.pkl` and performs inference only.
- `model_final_lstm.py` loads `models/lstm_best.h5` and `models/scaler_y.pkl` and performs inference only.

Both scripts generate a horizon relative to the latest historical month rather than using a fixed calendar end date.

## Model artifacts

Production artifacts:

```text
models/
├── xgb_best.pkl
├── lstm_best.h5
└── scaler_y.pkl
```

Candidate artifacts are kept separate until reviewed:

```text
models/
├── xgb_candidate.pkl
├── lstm_candidate.keras
├── scaler_y_candidate.pkl
└── lstm_candidate_metrics.json
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
production model inference
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
│   ├── retrain_models.py
│   └── train_lstm.py
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
