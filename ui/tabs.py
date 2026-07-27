"""Tabs layout and content rendering."""

import pandas as pd
import streamlit as st

from charts.heatmap import plot_seasonality_heatmap
from charts.origins import plot_origins_donut
from charts.trends import plot_flight_type_shares, plot_total_passengers
from forecast.forecast_plot import plot_forecast_tab


def display_tabs(
    dfv: pd.DataFrame,
    df_full: pd.DataFrame,
    df_xgb: pd.DataFrame | None,
    df_lstm: pd.DataFrame | None,
    df_forecast_history: pd.DataFrame,
    selected_island: str,
) -> None:
    """Create and render all dashboard tabs."""
    df_island = df_full[df_full["Isla"] == selected_island].copy()
    df_island["Fecha"] = pd.to_datetime(df_island["Fecha"], errors="coerce")

    min_d = df_island["Fecha"].min()
    max_d = df_island["Fecha"].max()
    default_start = max(min_d, max_d - pd.DateOffset(months=12))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Datos",
            "📈 Gráfico",
            "🌍 Origen",
            "✈️ Variación estacional",
            "🔮 Pronóstico",
        ]
    )

    with tab1:
        st.subheader(f"📅 Rango de fechas — {selected_island}")
        rango = st.slider(
            "Selecciona el rango de fechas",
            min_value=min_d.to_pydatetime(),
            max_value=max_d.to_pydatetime(),
            value=(default_start.to_pydatetime(), max_d.to_pydatetime()),
            format="YYYY-MM",
            key=f"slider_tab1_{selected_island}",
        )
        mask = (df_island["Fecha"] >= rango[0]) & (df_island["Fecha"] <= rango[1])
        dfv_filtered = df_island.loc[mask].copy()
        st.subheader("Totales por año y mes — filtrado por el rango seleccionado")
        df_total = dfv_filtered[
            dfv_filtered["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"
        ]
        if df_total.empty:
            st.warning("No hay datos 'TOTAL PASAJEROS' en este rango.")
        else:
            yearly = (
                df_total.groupby("Año", as_index=False)["Pasajeros"]
                .sum()
                .sort_values("Año")
            )
            monthly = df_total[["Año", "MesNum", "Pasajeros"]].sort_values(
                ["Año", "MesNum"]
            )
            col1, col2 = st.columns(2)
            col1.markdown("### 🟦 Totales por año")
            col1.dataframe(yearly, use_container_width=True)
            col2.markdown("### 🟦 Totales por mes")
            col2.dataframe(monthly, use_container_width=True)

    with tab2:
        st.subheader(f"📅 Rango de fechas — {selected_island}")
        rango = st.slider(
            "Selecciona el rango de fechas",
            min_value=min_d.to_pydatetime(),
            max_value=max_d.to_pydatetime(),
            value=(default_start.to_pydatetime(), max_d.to_pydatetime()),
            format="YYYY-MM",
            key=f"slider_tab2_{selected_island}",
        )
        mask = (df_island["Fecha"] >= rango[0]) & (df_island["Fecha"] <= rango[1])
        dfv_filtered = df_island.loc[mask].copy()
        st.subheader("📈 Evolución mensual — Total Pasajeros")
        plot_total_passengers(dfv_filtered)
        plot_flight_type_shares(dfv_filtered)

    with tab3:
        st.subheader(f"📅 Rango de fechas — {selected_island}")
        rango = st.slider(
            "Selecciona el rango de fechas",
            min_value=min_d.to_pydatetime(),
            max_value=max_d.to_pydatetime(),
            value=(default_start.to_pydatetime(), max_d.to_pydatetime()),
            format="YYYY-MM",
            key=f"slider_tab3_{selected_island}",
        )
        mask = (df_island["Fecha"] >= rango[0]) & (df_island["Fecha"] <= rango[1])
        dfv_filtered = df_island.loc[mask].copy()
        plot_origins_donut(dfv_filtered)

    with tab4:
        plot_seasonality_heatmap(df_full, selected_island)

    with tab5:
        plot_forecast_tab(df_full, df_xgb, df_lstm, df_forecast_history)
