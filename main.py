"""Main Streamlit app for Canarias passengers dashboard.

- UI texts are in Spanish
- Comments are in English
- Logic follows the big monolithic script we developed, but structured in modules.
"""

import streamlit as st
import pandas as pd

from data.loader import load_main_dataset, load_forecasts
from ui.map import draw_island_map
from ui.images import show_island_image, show_image_license
from kpi.kpi_calculator import calculate_kpi_full
from ui.tabs import display_tabs

# ---------- Global CSS for image style ----------
st.markdown(
    """<style>
    .map-image img {
        height: 520px !important;
        width: auto !important;
        object-fit: contain !important;
        border-radius: 12px;
        background: #f2f2f2;
        padding: 8px;
    }
    </style>""", unsafe_allow_html=True
)

# ---------- Page config ----------
st.set_page_config(page_title="✈️ Pasajeros Canarias Dashboard", layout="wide")
st.title("🗺️ Análisis de pasajeros aéreos en las Islas Canarias")

# ---------- Load data ----------
dfp = load_main_dataset()
df_xgb, df_lstm = load_forecasts()

# ---------- Map + image layout ----------
col_map, col_img = st.columns([2, 1])

with col_map:
    selected_island = draw_island_map()

with col_img:
    if selected_island:
        show_island_image(selected_island)
    else:
        st.info("Haz click en una isla del mapa para ver su imagen.")

show_image_license()

if not selected_island:
    st.info("Haz click en una isla o selecciona 'Total Canarias'.")
    st.stop()

# ---------- Filter data for selected island (before slider) ----------
dfv = dfp[dfp["Isla"].str.contains(selected_island, case=False, na=False)].copy()
if dfv.empty:
    st.warning(f"No hay datos para {selected_island}.")
    st.stop()

# ---------- Date slider (affects only dfv) ----------
st.subheader(f"📅 Rango de fechas — {selected_island}")

dfv["Fecha"] = pd.to_datetime(dfv["Fecha"], errors="coerce")
min_d, max_d = dfv["Fecha"].min(), dfv["Fecha"].max()
default_start = max(min_d, max_d - pd.DateOffset(months=12))

rango = st.slider(
    "Selecciona el rango de fechas",
    min_value=min_d.to_pydatetime(),
    max_value=max_d.to_pydatetime(),
    value=(default_start.to_pydatetime(), max_d.to_pydatetime()),
    format="YYYY-MM",
    key=f"slider_{selected_island}",
)

mask = (dfv["Fecha"] >= pd.Timestamp(rango[0])) & (dfv["Fecha"] <= pd.Timestamp(rango[1]))
dfv = dfv.loc[mask].copy()

# ---------- KPI (independent from slider, uses full dfp) ----------
kpi = calculate_kpi_full(dfp, selected_island)

st.markdown(f"## 📊 KPI — {selected_island}")
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

# ---------- Tabs (Datos, Gráfico, Origen, Tipos de vuelo, Pronóstico) ----------
display_tabs(dfv, dfp, df_xgb, df_lstm, selected_island)
