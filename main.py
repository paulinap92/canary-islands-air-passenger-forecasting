"""Main Streamlit app for Canarias passengers dashboard."""

import pandas as pd
import streamlit as st

from data.loader import load_forecast_history, load_forecasts, load_main_dataset
from kpi.kpi_calculator import calculate_kpi_full
from ui.images import show_image_license, show_island_image
from ui.map import draw_island_map
from ui.tabs import display_tabs

st.set_page_config(page_title="✈️ Pasajeros Canarias Dashboard", layout="wide")
st.title("🗺️ Análisis de pasajeros aéreos en las Islas Canarias")

dfp = load_main_dataset()
df_xgb, df_lstm = load_forecasts()
df_forecast_history = load_forecast_history()

col_map, col_img = st.columns([2, 1], vertical_alignment="top")

with col_map:
    selected_island = draw_island_map()

with col_img:
    if selected_island:
        show_island_image(selected_island)
    else:
        st.info("Haz click en una isla del mapa para ver su imagen.")

show_image_license()

if not selected_island:
    st.stop()

st.markdown("---")
st.markdown(f"## 📊 KPI — {selected_island}")

kpi = calculate_kpi_full(dfp, selected_island)

c1, c2, c3 = st.columns(3)
c4, c5 = st.columns(2)

c1.metric(
    f"📅 Último mes ({kpi['last_month_label']})",
    f"{kpi['last_month_total']:,.0f}".replace(",", " "),
)

if kpi["prev_month_total"] is not None:
    c2.metric(
        "📅 Mismo mes año anterior",
        f"{kpi['prev_month_total']:,.0f}".replace(",", " "),
    )
else:
    c2.metric("📅 Mismo mes año anterior", "–")

if kpi["yoy_month_pct"] is not None:
    c3.metric("📈 YoY mensual (%)", f"{kpi['yoy_month_pct']:.1f}%")
else:
    c3.metric("📈 YoY mensual (%)", "–")

if kpi["yoy_year_pct"] is not None and kpi["full_year_prev"] and kpi["full_year_n"]:
    c4.metric(
        f"📅 YoY anual ({kpi['full_year_prev']}→{kpi['full_year_n']})",
        f"{kpi['yoy_year_pct']:.1f}%",
    )
else:
    c4.metric("📅 YoY anual", "–")

c5.metric(
    "🏆 Mejor mes",
    f"{kpi['best_value']:,.0f}".replace(",", " "),
    kpi["best_label"],
)

dfv = dfp[dfp["Isla"].str.contains(selected_island, case=False, na=False)].copy()

display_tabs(
    dfv,
    dfp,
    df_xgb,
    df_lstm,
    df_forecast_history,
    selected_island,
)
