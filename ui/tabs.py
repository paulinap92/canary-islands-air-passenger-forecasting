"""Tabs layout and content rendering."""

import streamlit as st
import pandas as pd
from charts.trends import plot_total_passengers, plot_flight_type_shares
from charts.origins import plot_origins_donut
from charts.heatmap import plot_seasonality_heatmap
from forecast.forecast_plot import plot_forecast_tab

def display_tabs(dfv: pd.DataFrame, df_full: pd.DataFrame, df_xgb, df_lstm, selected_island: str):
    """Create and render all tabs: Datos, Gráfico, Origen, Tipos de vuelo, Pronóstico."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Datos", "📈 Gráfico", "🌍 Origen", "✈️ Variación estacional", "🔮 Pronóstico",
    ])

    # --- TAB 1: Datos (year/month tables, filtered by slider) ---
    with tab1:
        st.subheader("Totales por año y mes — filtrado por el rango seleccionado")

        df_total_island_slider = dfv[
            dfv["AEROPUERTO_DE_PROCEDENCIA"].str.upper() == "TOTAL PASAJEROS"
        ].copy()

        if df_total_island_slider.empty:
            st.warning("No hay datos 'TOTAL PASAJEROS' para esta isla en el rango seleccionado.")
        else:
            yearly = (
                df_total_island_slider
                .groupby("Año", as_index=False)["Pasajeros"]
                .sum()
                .sort_values("Año")
            )
            monthly = (
                df_total_island_slider[["Año", "Mes", "Pasajeros"]]
                .sort_values(["Año", "Mes"])
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🟦 Totales por año (según rango)")
                st.dataframe(yearly, use_container_width=True)
            with col2:
                st.markdown("### 🟦 Totales por mes (según rango)")
                st.dataframe(monthly, use_container_width=True)

    # --- TAB 2: Gráfico (total + tipos) ---
    with tab2:
        st.subheader("📈 Evolución mensual — Total Pasajeros")
        plot_total_passengers(dfv)
        plot_flight_type_shares(dfv)

    # --- TAB 3: Origen donut ---
    with tab3:
        plot_origins_donut(dfv)

    # --- TAB 4: Heatmap full dataset ---
    with tab4:
        plot_seasonality_heatmap(df_full, selected_island)

    # --- TAB 5: Pronóstico total Canarias ---
    with tab5:
        plot_forecast_tab(df_full, df_xgb, df_lstm)
