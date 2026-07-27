"""Forecast tab: show historical + XGB + LSTM predictions."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def plot_forecast_tab(
    df_full: pd.DataFrame,
    df_xgb: pd.DataFrame,
    df_lstm: pd.DataFrame,
    df_forecast_history: pd.DataFrame,
):
    """Render forecast tab: historical Total Canarias + XGB + LSTM."""
    st.subheader("🔮 Predicción — Histórico + XGB + LSTM (Total Canarias)")

    if df_xgb is None or df_lstm is None:
        st.warning("⚠️ No se pudieron cargar las predicciones.")
        return

    # Historical data for TOTAL PASAJEROS (all islands)
    df_hist = df_full[
        df_full["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"
    ].copy()

    if df_hist.empty:
        st.warning("No hay datos históricos de 'TOTAL PASAJEROS'.")
        return

    last_real_date = df_hist["Fecha"].max()

    xgb_real = df_xgb[df_xgb["Fecha"] <= last_real_date]
    xgb_pred = df_xgb[df_xgb["Fecha"] > last_real_date]

    lstm_real = df_lstm[df_lstm["Fecha"] <= last_real_date]
    lstm_pred = df_lstm[df_lstm["Fecha"] > last_real_date]

    model_choice = st.radio("Modelo", ["XGB", "LSTM", "Ambos"], horizontal=True)
    show_history = st.checkbox("Mostrar historia anterior", value=False)

    fig = go.Figure()

    # XGB traces
    if model_choice in ["XGB", "Ambos"]:
        fig.add_trace(go.Scatter(
            x=xgb_real["Fecha"],
            y=xgb_real["Pasajeros"],
            name="XGB (ajuste)",
            line=dict(color="orange", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=xgb_pred["Fecha"],
            y=xgb_pred["Pasajeros"],
            name="XGB (predicción)",
            line=dict(color="orange", width=3, dash="dash"),
        ))

    # LSTM traces
    if model_choice in ["LSTM", "Ambos"]:
        fig.add_trace(go.Scatter(
            x=lstm_real["Fecha"],
            y=lstm_real["Pasajeros"],
            name="LSTM (ajuste)",
            line=dict(color="green", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=lstm_pred["Fecha"],
            y=lstm_pred["Pasajeros"],
            name="LSTM (predicción)",
            line=dict(color="green", width=3, dash="dot"),
        ))

    # Optional legacy snapshot overlay. This does not modify current forecasts.
    if show_history and df_forecast_history is not None and not df_forecast_history.empty:
        history = df_forecast_history.copy()
        history["target_month"] = pd.to_datetime(history["target_month"], errors="coerce")
        history["predicted_passengers"] = pd.to_numeric(
            history["predicted_passengers"], errors="coerce"
        )
        history = history.dropna(subset=["target_month", "predicted_passengers"])

        if model_choice in ["XGB", "Ambos"]:
            xgb_history = history[history["model"].str.upper() == "XGB"].sort_values(
                "target_month"
            )
            if not xgb_history.empty:
                fig.add_trace(go.Scatter(
                    x=xgb_history["target_month"],
                    y=xgb_history["predicted_passengers"],
                    name="XGB (historia 2025-09)",
                    line=dict(color="orange", width=2, dash="dot"),
                    opacity=0.45,
                ))

        if model_choice in ["LSTM", "Ambos"]:
            lstm_history = history[history["model"].str.upper() == "LSTM"].sort_values(
                "target_month"
            )
            if not lstm_history.empty:
                fig.add_trace(go.Scatter(
                    x=lstm_history["target_month"],
                    y=lstm_history["predicted_passengers"],
                    name="LSTM (historia 2025-09)",
                    line=dict(color="green", width=2, dash="dash"),
                    opacity=0.45,
                ))

    fig.update_layout(
        height=500,
        template="simple_white",
        xaxis_title="Fecha",
        yaxis_title="Pasajeros",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Ver datos"):
        xgb_display = df_xgb[["Fecha", "Pasajeros", "Phase"]].copy()
        xgb_display["Fecha"] = xgb_display["Fecha"].dt.to_period("M").astype(str)
        xgb_display = xgb_display.sort_values("Fecha")

        lstm_display = df_lstm[["Fecha", "Pasajeros", "Phase"]].copy()
        lstm_display["Fecha"] = lstm_display["Fecha"].dt.to_period("M").astype(str)
        lstm_display = lstm_display.sort_values("Fecha")

        if model_choice == "XGB":
            st.markdown("### 🔸 Datos — XGB")
            st.dataframe(xgb_display, use_container_width=True)
        elif model_choice == "LSTM":
            st.markdown("### 🟢 Datos — LSTM")
            st.dataframe(lstm_display, use_container_width=True)
        else:
            st.markdown("### 🔸 Datos — XGB")
            st.dataframe(xgb_display, use_container_width=True)
            st.markdown("### 🟢 Datos — LSTM")
            st.dataframe(lstm_display, use_container_width=True)
