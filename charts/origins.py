"""Origins donut chart for top origin airports/countries."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

EXCLUDE_ORIGINS = [
    "TOTAL PASAJEROS",
    "AEROP. INTERINSULARES",
    "AEROP. PENINSULARES",
    "TOTAL AEROP. EXTRANJEROS",
    "AEROP. PENINSULARES + AEROP. EXTRANJEROS",
]

def plot_origins_donut(dfv: pd.DataFrame):
    """Display a donut chart of top origin airports / countries."""
    st.subheader("🌍 Top aeropuertos de procedencia (donut)")

    df_clean = dfv[
        ~dfv["AEROPUERTO_DE_PROCEDENCIA"].str.upper().isin(EXCLUDE_ORIGINS)
    ]

    if df_clean.empty:
        st.warning("No hay datos de procedencia para este rango.")
        return

    country_sum = df_clean.groupby("AEROPUERTO_DE_PROCEDENCIA")["Pasajeros"].sum()
    df_top = country_sum.sort_values(ascending=False).head(10)
    otros_value = country_sum.sum() - df_top.sum()

    df_plot = df_top.copy()
    df_plot["Otros países"] = max(0, otros_value)

    fig = go.Figure(data=[go.Pie(
        labels=df_plot.index,
        values=df_plot.values,
        hole=0.55,
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(size=18),
        hovertemplate="<b>%{label}</b><br>%{value:,} pasajeros<br>%{percent}<extra></extra>",
        automargin=True,
        marker=dict(
            colors=[
                "#005BBB", "#00A6FB", "#8CCFFF", "#FDB833", "#2E4057",
                "#0077B6", "#0096C7", "#48CAE4", "#ADE8F4",
                "#CAF0F8", "#6C757D",
            ]
        ),
    )])

    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=150, r=150, t=100, b=120),
        showlegend=False,
    )
    st.plotly_chart(fig)
