"""Data loading utilities cached by Streamlit."""

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_main_dataset() -> pd.DataFrame:
    """Load the main passengers dataset from result.csv."""
    df = pd.read_csv("result.csv", parse_dates=["Fecha"], encoding="utf-8-sig")
    df = df.rename(columns=str.strip)
    return df


@st.cache_data
def load_forecasts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
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
    except Exception as error:
        st.warning(f"⚠️ No se pudieron cargar las predicciones: {error}")
        return None, None


@st.cache_data
def load_forecast_history() -> pd.DataFrame:
    """Load immutable forecast snapshots when the history file exists."""
    history_path = Path("forecast_history.csv")
    columns = [
        "run_id",
        "generated_at_utc",
        "data_through",
        "model",
        "target_month",
        "horizon_months",
        "predicted_passengers",
        "source_commit",
    ]
    if not history_path.exists():
        return pd.DataFrame(columns=columns)

    try:
        history = pd.read_csv(history_path, encoding="utf-8-sig")
        missing = sorted(set(columns).difference(history.columns))
        if missing:
            raise KeyError(f"Faltan columnas: {missing}")

        history = history[columns].copy()
        history["generated_at_utc"] = pd.to_datetime(
            history["generated_at_utc"], utc=True, errors="coerce"
        )
        history["data_through"] = pd.to_datetime(
            history["data_through"], errors="coerce"
        )
        history["target_month"] = pd.to_datetime(
            history["target_month"], errors="coerce"
        )
        history["horizon_months"] = pd.to_numeric(
            history["horizon_months"], errors="coerce"
        )
        history["predicted_passengers"] = pd.to_numeric(
            history["predicted_passengers"], errors="coerce"
        )
        history = history.dropna(
            subset=[
                "generated_at_utc",
                "data_through",
                "target_month",
                "horizon_months",
                "predicted_passengers",
            ]
        )
        return history.sort_values(
            ["generated_at_utc", "model", "target_month"]
        ).reset_index(drop=True)
    except Exception as error:
        st.warning(f"⚠️ No se pudo cargar el histórico de predicciones: {error}")
        return pd.DataFrame(columns=columns)
