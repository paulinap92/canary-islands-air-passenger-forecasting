"""KPI calculation logic (independent from the date slider)."""

import pandas as pd

def calculate_kpi_full(df_full, selected_island):
    """Calculate YoY KPIs for the selected island.

    Logic:
    - Uses ALL rows for that island (not only TOTAL PASAJEROS),
      exactly as in the original script (we do not change the logic).
    - Last month vs same month previous year
    - Full previous year vs the year before
    - Best month ever for the island
    """

    df = df_full[df_full["Isla"] == selected_island].copy()

    # Ensure datetime column
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values("Fecha")

    # ----- 1) Last month YoY -----
    last_date = df["Fecha"].max()
    year = last_date.year
    month = last_date.month

    df_last = df[(df["Fecha"].dt.year == year) & (df["Fecha"].dt.month == month)]
    last_month_total = df_last["Pasajeros"]

    df_prev = df[(df["Fecha"].dt.year == year - 1) & (df["Fecha"].dt.month == month)]
    prev_month_total = df_prev["Pasajeros"] if not df_prev.empty else None

    if prev_month_total and prev_month_total > 0:
        yoy_month_pct = ((last_month_total - prev_month_total) / prev_month_total) * 100
        yoy_month_diff = last_month_total - prev_month_total
    else:
        yoy_month_pct = None
        yoy_month_diff = None

    # ----- 2) Full year YoY (previous full year vs year before) -----
    years = sorted(df["Año"].unique())

    if len(years) >= 3:
        full_year_n = years[-2]      # e.g. 2023
        full_year_prev = years[-3]   # e.g. 2022
    else:
        full_year_n = None
        full_year_prev = None

    if full_year_n and full_year_prev:
        total_year_n = df[df["Año"] == full_year_n]["Pasajeros"].sum()
        total_year_prev = df[df["Año"] == full_year_prev]["Pasajeros"].sum()

        if total_year_prev > 0:
            yoy_year_pct = ((total_year_n - total_year_prev) / total_year_prev) * 100
            yoy_year_diff = total_year_n - total_year_prev
        else:
            yoy_year_pct = None
            yoy_year_diff = None
    else:
        yoy_year_pct = None
        yoy_year_diff = None

    # ----- 3) Best month ever for this island -----
    best_row = df.loc[df["Pasajeros"].idxmax()]
    best_value = best_row["Pasajeros"]
    best_label = best_row["Mes_Año"]

    return {
        "last_month_label": last_date.strftime("%B %Y"),
        "last_month_total": last_month_total,
        "prev_month_total": prev_month_total,
        "yoy_month_pct": yoy_month_pct,
        "yoy_month_diff": yoy_month_diff,
        "full_year_n": full_year_n,
        "full_year_prev": full_year_prev,
        "yoy_year_pct": yoy_year_pct,
        "yoy_year_diff": yoy_year_diff,
        "best_value": best_value,
        "best_label": best_label,
    }
