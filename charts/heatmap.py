"""Heatmap of seasonality by month and year (using full dataset)."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def plot_seasonality_heatmap(df_full: pd.DataFrame, selected_island: str):
    """Plot a heatmap of monthly passengers by year for the island (TOTAL PASAJEROS only).

    Uses the FULL dataset (not filtered by the slider) and excludes pandemic years 2020–2021.
    """
    st.subheader("🔥 Heatmapa — Estacionalidad por mes y año (sin pandemia)")

    # Filter island
    df_island_full = df_full[
        df_full["Isla"].str.contains(selected_island, case=False, na=False)
    ].copy()

    # Only TOTAL PASAJEROS
    df_heat = df_island_full[
        df_island_full["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"
    ].copy()

    if df_heat.empty:
        st.warning("No hay datos para generar la heatmapa.")
        return

    # Ensure Fecha is datetime
    df_heat["Fecha"] = pd.to_datetime(df_heat["Fecha"], errors="coerce")

    # Recompute Año & Mes cleanly
    df_heat["Año"] = df_heat["Fecha"].dt.year.astype(int)
    df_heat["Mes"] = df_heat["Fecha"].dt.month.astype(int)

    # Remove pandemic years
    df_heat = df_heat[df_heat["Año"] >= 2022]

    if df_heat.empty:
        st.warning("No hay datos después de 2022 para generar la heatmapa.")
        return

    st.caption("📝 Los años 2020–2021 se excluyen debido al impacto anómalo de la pandemia en el tráfico aéreo.")

    month_labels = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    # Pivot: rows = Año, cols = Mes
    heat_table = df_heat.pivot(index="Año", columns="Mes", values="Pasajeros")
    heat_table = heat_table.reindex(columns=range(1, 13))

    heat_values = heat_table.fillna(0)
    text_values = heat_table.fillna("").astype(str)

    # 🔥 KLUCZ: oś Y jako stringi (kategorie), nie liczby
    y_labels = [str(int(y)) for y in heat_values.index]

    fig = go.Figure(data=go.Heatmap(
        z=heat_values.values,
        x=month_labels,
        y=y_labels,  # <- teksty zamiast liczb
        colorscale="Viridis",
        colorbar=dict(title="Pasajeros"),
        text=text_values.values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        hovertemplate="<b>Año %{y} %{x}</b><br>Pasajeros: %{z:,}<extra></extra>",
    ))

    fig.update_layout(
        height=580,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Mes",
        yaxis_title="Año",
        template="simple_white",
        yaxis=dict(type="category")  # <- wymuszamy kategorie
    )

    st.plotly_chart(fig, use_container_width=True)
