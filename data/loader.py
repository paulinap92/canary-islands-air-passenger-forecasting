"""Data loading utilities (cached in Streamlit)."""

from pathlib import Path

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
        df_xgb = pd.read_csv(
            "forecast_total_canarias_xgb.csv",
            parse_dates=["Fecha"],
            encoding="utf-8-sig",
        )
        df_lstm = pd.read_csv(
            "forecast_total_canarias_lstm.csv",
            parse_dates=["Fecha"],
            encoding="utf-8-sig",
        )
        return df_xgb, df_lstm
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar las predicciones: {e}")
        return None, None


@st.cache_data
def load_forecast_history():
    """Load the optional legacy forecast snapshot."""
    path = Path("forecast_history.csv")
    if not path.exists():
        return pd.DataFrame()

    try:
        history = pd.read_csv(
            path,
            parse_dates=["data_through", "target_month"],
            encoding="utf-8-sig",
        )
        return history
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar la historia de predicciones: {e}")
        return pd.DataFrame()
