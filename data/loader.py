"""Data loading utilities (cached in Streamlit)."""

import pandas as pd
import streamlit as st

@st.cache_data
def load_main_dataset():
    """Load the main passengers dataset from result.csv."""
    df = pd.read_csv("result.csv", parse_dates=["Fecha"], encoding="utf-8-sig")
    df = df.rename(columns=str.strip)
    return df

@st.cache_data
def load_forecasts():
    """Load XGB and LSTM forecast CSVs if available."""
    try:
        df_xgb = pd.read_csv("forecast_total_canarias_xgb.csv", parse_dates=["Fecha"], encoding="utf-8-sig")
        df_lstm = pd.read_csv("forecast_total_canarias_lstm.csv", parse_dates=["Fecha"], encoding="utf-8-sig")
        return df_xgb, df_lstm
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar las predicciones: {e}")
        return None, None
