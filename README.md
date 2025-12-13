# ✈️ Canary Islands Air Passenger Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-EC0000)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly)
![Folium](https://img.shields.io/badge/Folium-77B829)

Interactive Streamlit dashboard for **analysis and forecasting of air passenger traffic** in the Canary Islands, based on **real, official tourism data**.

---

---

## 📌 Project Overview

This project analyses historical air passenger traffic for the Canary Islands and forecasts future demand using machine learning and deep learning models.

Key goals:
- understand long-term trends and seasonality,
- build stable forecasting models,
- automatically update data and results,
- present insights in an interactive dashboard.

A broad experimental phase was conducted with multiple models and neural networks.  
After evaluation, **two stable models** were selected as final.

---

## 🧠 Models

- **XGBoost**
  - lag features (historical passenger values),
  - explicit seasonality encoding,
  - robust performance on tabular time-series data.

- **LSTM (Keras)**
  - long-term temporal dependencies,
  - complementary deep-learning approach.

Evaluation metrics:
- MAE
- RMSE
- MAPE

---

## 🧪 Model Training & Experiments

The final forecasting models used in the dashboard were obtained after an extensive experimental and training phase.

During development, multiple approaches, feature sets and neural network architectures were tested.  
The following files document the **training, experimentation and model selection process**:

### Training & experiments

- **`prepare_data_for_model.ipynb`**  
  Data preparation and feature engineering notebook.  
  Includes aggregation of passenger data, creation of lag features, seasonality encoding and datasets used for model training.

- **`my_models_trials.ipynb`**  
  Experimental notebook with multiple model trials and configurations, used to compare approaches before selecting the final models.

- **`data_processing.ipynb`**  
  Initial data exploration, cleaning and validation of raw XLSX files.

### Final training scripts

- **`model_final_xgb.py`**  
  Training script for the final **XGBoost model** (lags + seasonality) with model persistence.

- **`model_final_lstm.py`**  
  Training script for the final **LSTM (Keras) model** (sequence input) with model persistence.

### Trained model artifacts

- **`models/xgb_best.pkl`** – final trained XGBoost model  
- **`models/lstm_best.h5`** – final trained LSTM model  
- **`models/scaler_y.pkl`** – target scaler used during training and inference  

Only the **final, stable models** are loaded by the Streamlit application.  
Intermediate experiments and notebooks are kept for transparency and reproducibility.

---

## 📊 Data Source

All data used in this project is **real and official**.

**Source:**  
WebTenerife – Air Traffic Statistics  
https://www.webtenerife.com/investigacion/situacion-turistica/trafico-aereo/

---

## 🔄 Data Download Agent

The project includes a **download agent** that:
- retrieves newly published air passenger data,
- validates and preprocesses it,
- updates datasets used by the dashboard.

After running the agent, the dashboard automatically reflects updated data.

---

## 📂 Project Structure (from repository)

```text
canarias_dashboard/
├── main.py
├── config.py
├── download_agent.py
├── model_final_lstm.py
├── model_final_xgb.py
├── requirements.txt
├── Pipfile
├── Pipfile.lock
├── forecast_total_canarias_lstm.csv
├── forecast_total_canarias_xgb.csv
├── result.csv
├── result_total.csv
├── result_total_with_lags.csv
├── result_total_with_lags_coded.csv
│
├── .streamlit/
│   └── config.toml
│
├── charts/
│   ├── heatmap.py
│   ├── origins.py
│   └── trends.py
│
├── forecast/
│   └── forecast_plot.py
│
├── kpi/
│   └── kpi_calculator.py
│
├── ui/
│   ├── images.py
│   ├── map.py
│   └── tabs.py
│
├── models/
│   ├── lstm_best.h5
│   ├── scaler_y.pkl
│   └── xgb_best.pkl
│
├── data/
│   ├── *.xlsx
│   └── loader.py
│
├── backup_results/
│   └── *.csv
│
└── notebooks/
    ├── data_processing.ipynb
    ├── my_models_trials.ipynb
    └── prepare_data_for_model.ipynb
```


---

## ▶️ How to Run (Pipenv)

### 1️⃣ Install dependencies
```bash
pipenv install
```

### 2️⃣ Run Streamlit dashboard
```bash
 streamlit run main.py
```

## 📄 License

Educational and analytical use.
