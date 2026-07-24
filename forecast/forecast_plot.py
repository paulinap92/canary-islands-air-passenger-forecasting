"""Forecast tab with one actual series and future model forecasts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATE_COL = "Fecha"
TARGET_COL = "Pasajeros"
PHASE_COL = "Phase"


def _prepare_forecast_frame(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    required = {DATE_COL, TARGET_COL, PHASE_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{model_name}: missing columns {missing}")

    result = df.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")
    result[TARGET_COL] = pd.to_numeric(result[TARGET_COL], errors="coerce")
    return result.dropna(subset=[DATE_COL, TARGET_COL]).sort_values(DATE_COL)


def plot_forecast_tab(
    df_full: pd.DataFrame,
    df_xgb: pd.DataFrame,
    df_lstm: pd.DataFrame,
) -> None:
    if df_xgb is None or df_lstm is None:
        st.warning("⚠️ No se pudieron cargar las predicciones.")
        return

    df_hist = df_full[
        df_full["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"
    ].copy()
    df_hist[DATE_COL] = pd.to_datetime(df_hist[DATE_COL], errors="coerce")
    df_hist[TARGET_COL] = pd.to_numeric(df_hist[TARGET_COL], errors="coerce")
    df_hist = (
        df_hist.dropna(subset=[DATE_COL, TARGET_COL])
        .groupby(DATE_COL, as_index=False)[TARGET_COL]
        .sum()
        .sort_values(DATE_COL)
    )
    if df_hist.empty:
        st.warning("No hay datos históricos de 'TOTAL PASAJEROS'.")
        return

    try:
        xgb = _prepare_forecast_frame(df_xgb, "XGBoost")
        lstm = _prepare_forecast_frame(df_lstm, "LSTM")
    except (KeyError, ValueError) as exc:
        st.warning(f"⚠️ Predicciones inválidas: {exc}")
        return

    last_real_date = df_hist[DATE_COL].max()
    xgb_future = xgb[
        (xgb[PHASE_COL] == "Forecast") & (xgb[DATE_COL] > last_real_date)
    ].copy()
    lstm_future = lstm[
        (lstm[PHASE_COL] == "Forecast") & (lstm[DATE_COL] > last_real_date)
    ].copy()

    available_end_dates = [
        frame[DATE_COL].max()
        for frame in (xgb_future, lstm_future)
        if not frame.empty
    ]
    if not available_end_dates:
        st.warning("No hay predicciones futuras disponibles.")
        return
    forecast_end = max(available_end_dates)
    st.subheader(
        "🔮 Pronóstico — datos reales y predicciones hasta "
        f"{forecast_end.strftime('%m/%Y')}"
    )

    model_choice = st.radio(
        "Modelo", ["XGB", "LSTM", "Ambos"], horizontal=True
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_hist[DATE_COL],
            y=df_hist[TARGET_COL],
            name="Datos reales",
            mode="lines",
            line=dict(width=3),
        )
    )

    if model_choice in {"XGB", "Ambos"}:
        fig.add_trace(
            go.Scatter(
                x=xgb_future[DATE_COL],
                y=xgb_future[TARGET_COL],
                name="XGB — pronóstico",
                mode="lines+markers",
                line=dict(width=3, dash="dash"),
            )
        )
    if model_choice in {"LSTM", "Ambos"}:
        fig.add_trace(
            go.Scatter(
                x=lstm_future[DATE_COL],
                y=lstm_future[TARGET_COL],
                name="LSTM — pronóstico",
                mode="lines+markers",
                line=dict(width=3, dash="dot"),
            )
        )

    fig.add_vline(
        x=last_real_date,
        line_dash="dash",
        annotation_text="Inicio del pronóstico",
        annotation_position="top left",
    )
    fig.update_layout(
        height=500,
        template="simple_white",
        xaxis_title="Fecha",
        yaxis_title="Pasajeros",
        margin=dict(l=20, r=20, t=50, b=20),
        legend_title_text="Serie",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Las líneas de XGB y LSTM muestran únicamente predicciones futuras. "
        "La historia corresponde a datos reales, no a valores ajustados por los modelos."
    )

    with st.expander("📋 Ver predicciones futuras"):
        tables = []
        if model_choice in {"XGB", "Ambos"}:
            xgb_display = xgb_future[[DATE_COL, TARGET_COL]].copy()
            xgb_display["Modelo"] = "XGB"
            tables.append(xgb_display)
        if model_choice in {"LSTM", "Ambos"}:
            lstm_display = lstm_future[[DATE_COL, TARGET_COL]].copy()
            lstm_display["Modelo"] = "LSTM"
            tables.append(lstm_display)

        display = pd.concat(tables, ignore_index=True)
        display[DATE_COL] = display[DATE_COL].dt.to_period("M").astype(str)
        display = display.rename(
            columns={DATE_COL: "Mes", TARGET_COL: "Pasajeros previstos"}
        )
        st.dataframe(
            display[["Mes", "Modelo", "Pasajeros previstos"]],
            use_container_width=True,
        )
