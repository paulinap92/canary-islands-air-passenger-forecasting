"""Trends charts (total passengers + % share by flight type)."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def plot_total_passengers(dfv: pd.DataFrame):
    """Plot monthly evolution of TOTAL PASAJEROS for the selected island and date range."""
    df_total = dfv[dfv["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"]

    if df_total.empty:
        st.warning("No se encontraron filas con 'TOTAL PASAJEROS'.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_total["Fecha"],
        y=df_total["Pasajeros"],
        mode="lines+markers",
        name="Total Pasajeros",
        line=dict(color="#004E98", width=4),
        marker=dict(size=8, color="#00A6FB"),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Total: %{y:,}<extra></extra>",
    ))

    fig.update_layout(
        height=430,
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Fecha",
        yaxis_title="Pasajeros",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_flight_type_shares(dfv: pd.DataFrame):
    """Plot % share of main flight types over time (interinsular, peninsular, extranjeros)."""
    st.subheader("📊 % de participación por tipo de vuelo")

    tipos = ["aerop. Interinsulares", "aerop. peninsulares", "Total aerop. Extranjeros"]
    df_types = dfv[dfv["AEROPUERTO_DE_PROCEDENCIA"].isin(tipos)]

    if df_types.empty:
        st.warning("No hay datos para estos tipos principales.")
        return

    df_month_total = df_types.groupby("Fecha")["Pasajeros"].sum().rename("Total")

    df_pct = (
        df_types.groupby(["Fecha", "AEROPUERTO_DE_PROCEDENCIA"])["Pasajeros"]
        .sum()
        .unstack(fill_value=0)
    )

    df_pct = df_pct.div(df_month_total, axis=0) * 100

    fig = go.Figure()
    colors = {
        "aerop. Interinsulares": "#1f77b4",
        "aerop. peninsulares": "#ff7f0e",
        "Total aerop. Extranjeros": "#2ca02c",
    }

    for col in df_pct.columns:
        fig.add_trace(go.Scatter(
            x=df_pct.index,
            y=df_pct[col],
            mode="lines+markers",
            name=col.replace("aerop.", "").strip(),
            line=dict(width=3, color=colors.get(col, "#555555")),
            hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        height=430,
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Fecha",
        yaxis_title="Porcentaje (%)",
        yaxis=dict(ticksuffix="%"),
    )
    st.plotly_chart(fig, use_container_width=True)
