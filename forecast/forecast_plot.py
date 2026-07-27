"""Forecast tab with current forecasts, historical snapshots and error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_COLORS = {"XGB": "orange", "LSTM": "green"}


def _load_actual_total(df_full: pd.DataFrame) -> pd.DataFrame:
    """Return one actual Total Canarias passenger value per month."""
    actual = df_full[
        df_full["AEROPUERTO_DE_PROCEDENCIA"].astype(str).str.upper()
        == "TOTAL PASAJEROS"
    ][["Fecha", "Pasajeros"]].copy()
    actual["Fecha"] = pd.to_datetime(actual["Fecha"], errors="coerce")
    actual["Pasajeros"] = pd.to_numeric(actual["Pasajeros"], errors="coerce")
    return (
        actual.dropna(subset=["Fecha", "Pasajeros"])
        .groupby("Fecha", as_index=False)["Pasajeros"]
        .sum()
        .sort_values("Fecha")
    )


def _selected_models(model_choice: str) -> list[str]:
    if model_choice == "Ambos":
        return ["XGB", "LSTM"]
    return [model_choice]


def _add_current_forecast(
    figure: go.Figure,
    forecast: pd.DataFrame,
    model: str,
    last_real_date: pd.Timestamp,
) -> None:
    """Add the current model forecast as a strong line."""
    if forecast is None or forecast.empty:
        return

    current = forecast.copy()
    current["Fecha"] = pd.to_datetime(current["Fecha"], errors="coerce")
    current["Pasajeros"] = pd.to_numeric(current["Pasajeros"], errors="coerce")
    current = current.dropna(subset=["Fecha", "Pasajeros"]).sort_values("Fecha")
    predicted = current[current["Fecha"] > last_real_date]
    if predicted.empty:
        return

    figure.add_trace(
        go.Scatter(
            x=predicted["Fecha"],
            y=predicted["Pasajeros"],
            name=f"{model} — actual",
            mode="lines+markers",
            line={
                "color": MODEL_COLORS[model],
                "width": 4,
                "dash": "dash" if model == "XGB" else "dot",
            },
            marker={"size": 7},
            hovertemplate=(
                f"{model} actual<br>%{{x|%Y-%m}}<br>"
                "%{y:,.0f} pasajeros<extra></extra>"
            ),
        )
    )


def _add_history_overlay(
    figure: go.Figure,
    history: pd.DataFrame,
    models: list[str],
    selected_runs: list[str],
) -> None:
    """Add selected historical snapshots as transparent lines."""
    if history.empty or not selected_runs:
        return

    filtered = history[
        history["model"].isin(models) & history["run_id"].isin(selected_runs)
    ].copy()
    if filtered.empty:
        return

    run_order = (
        filtered[["run_id", "generated_at_utc", "data_through"]]
        .drop_duplicates()
        .sort_values("generated_at_utc")
    )
    opacity_by_run = {
        run_id: 0.18 + 0.5 * (index + 1) / max(len(run_order), 1)
        for index, run_id in enumerate(run_order["run_id"])
    }

    for (run_id, model), group in filtered.groupby(["run_id", "model"], sort=False):
        group = group.sort_values("target_month")
        data_through = pd.Timestamp(group["data_through"].iloc[0]).strftime("%Y-%m")
        figure.add_trace(
            go.Scatter(
                x=group["target_month"],
                y=group["predicted_passengers"],
                name=f"{model} — historia {data_through}",
                mode="lines",
                line={
                    "color": MODEL_COLORS[model],
                    "width": 1.5,
                    "dash": "solid",
                },
                opacity=opacity_by_run[run_id],
                legendgroup=f"history-{model}",
                hovertemplate=(
                    f"{model} histórico<br>Datos hasta {data_through}<br>"
                    "%{x|%Y-%m}<br>%{y:,.0f} pasajeros<extra></extra>"
                ),
            )
        )


def _history_with_actuals(
    history: pd.DataFrame,
    actual: pd.DataFrame,
    models: list[str],
    selected_runs: list[str],
) -> pd.DataFrame:
    """Join historical predictions with actual values now available."""
    if history.empty or not selected_runs:
        return pd.DataFrame()

    filtered = history[
        history["model"].isin(models) & history["run_id"].isin(selected_runs)
    ].copy()
    evaluated = filtered.merge(
        actual.rename(columns={"Fecha": "target_month", "Pasajeros": "actual_passengers"}),
        on="target_month",
        how="inner",
    )
    if evaluated.empty:
        return evaluated

    evaluated["error"] = (
        evaluated["predicted_passengers"] - evaluated["actual_passengers"]
    )
    evaluated["abs_error"] = evaluated["error"].abs()
    evaluated["ape_pct"] = np.where(
        evaluated["actual_passengers"] != 0,
        evaluated["abs_error"] / evaluated["actual_passengers"] * 100.0,
        np.nan,
    )
    return evaluated


def _render_history_metrics(evaluated: pd.DataFrame) -> None:
    """Render aggregate and per-model historical forecast indicators."""
    st.markdown("### 📏 Indicadores históricos")
    if evaluated.empty:
        st.info(
            "Todavía no hay meses con valor real disponible para los snapshots "
            "seleccionados. Las métricas aparecerán cuando lleguen nuevos datos."
        )
        return

    mae = float(evaluated["abs_error"].mean())
    rmse = float(np.sqrt(np.mean(np.square(evaluated["error"]))))
    mape = float(evaluated["ape_pct"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicciones evaluadas", f"{len(evaluated):,}".replace(",", " "))
    c2.metric("MAE", f"{mae:,.0f}".replace(",", " "))
    c3.metric("RMSE", f"{rmse:,.0f}".replace(",", " "))
    c4.metric("MAPE", f"{mape:.2f}%")

    rows: list[dict[str, object]] = []
    for model, group in evaluated.groupby("model"):
        rows.append(
            {
                "Modelo": model,
                "Predicciones": len(group),
                "MAE": group["abs_error"].mean(),
                "RMSE": np.sqrt(np.mean(np.square(group["error"]))),
                "MAPE (%)": group["ape_pct"].mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values("MAE")
    st.dataframe(
        summary.style.format(
            {
                "MAE": "{:,.0f}",
                "RMSE": "{:,.0f}",
                "MAPE (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    by_horizon = (
        evaluated.groupby(["model", "horizon_months"], as_index=False)
        .agg(
            Predicciones=("abs_error", "size"),
            MAE=("abs_error", "mean"),
            RMSE=("error", lambda values: np.sqrt(np.mean(np.square(values)))),
            MAPE_pct=("ape_pct", "mean"),
        )
        .sort_values(["model", "horizon_months"])
    )
    with st.expander("📊 Indicadores por horizonte"):
        st.dataframe(
            by_horizon.style.format(
                {
                    "MAE": "{:,.0f}",
                    "RMSE": "{:,.0f}",
                    "MAPE_pct": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def plot_forecast_tab(
    df_full: pd.DataFrame,
    df_xgb: pd.DataFrame | None,
    df_lstm: pd.DataFrame | None,
    forecast_history: pd.DataFrame,
) -> None:
    """Render actual data, current forecasts and optional history overlay."""
    st.subheader("🔮 Predicción — Histórico real + XGB + LSTM")

    if df_xgb is None or df_lstm is None:
        st.warning("⚠️ No se pudieron cargar las predicciones actuales.")
        return

    actual = _load_actual_total(df_full)
    if actual.empty:
        st.warning("No hay datos históricos de 'TOTAL PASAJEROS'.")
        return

    last_real_date = actual["Fecha"].max()
    model_choice = st.radio("Modelo", ["XGB", "LSTM", "Ambos"], horizontal=True)
    models = _selected_models(model_choice)

    show_history = st.toggle(
        "History — superponer predicciones anteriores",
        value=False,
        help=(
            "Muestra snapshots guardados en ejecuciones anteriores junto con la "
            "serie real y la predicción actual."
        ),
    )

    selected_runs: list[str] = []
    history = forecast_history.copy()
    if not history.empty:
        history["model"] = history["model"].astype(str).str.upper()
        run_options = (
            history[history["model"].isin(models)][
                ["run_id", "generated_at_utc", "data_through"]
            ]
            .drop_duplicates()
            .sort_values("generated_at_utc", ascending=False)
        )
        run_labels = {
            row.run_id: (
                f"Dane do {pd.Timestamp(row.data_through):%Y-%m} · "
                f"run {str(row.run_id)[-8:]}"
            )
            for row in run_options.itertuples(index=False)
        }
        default_runs = run_options["run_id"].head(4).tolist()
        if show_history:
            selected_runs = st.multiselect(
                "Snapshoty historii",
                options=run_options["run_id"].tolist(),
                default=default_runs,
                format_func=lambda run_id: run_labels.get(run_id, run_id),
            )
    elif show_history:
        st.info("Historia prognoz zostanie utworzona po następnym jobie modeli.")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual["Fecha"],
            y=actual["Pasajeros"],
            name="Real",
            mode="lines+markers",
            line={"color": "#1f77b4", "width": 3},
            marker={"size": 5},
            hovertemplate="Real<br>%{x|%Y-%m}<br>%{y:,.0f} pasajeros<extra></extra>",
        )
    )

    if show_history:
        _add_history_overlay(figure, history, models, selected_runs)

    if "XGB" in models:
        _add_current_forecast(figure, df_xgb, "XGB", last_real_date)
    if "LSTM" in models:
        _add_current_forecast(figure, df_lstm, "LSTM", last_real_date)

    figure.add_vline(
        x=last_real_date,
        line_dash="dot",
        line_color="gray",
        annotation_text="Último dato real",
        annotation_position="top left",
    )
    figure.update_layout(
        height=570,
        template="simple_white",
        xaxis_title="Fecha",
        yaxis_title="Pasajeros",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    st.plotly_chart(figure, use_container_width=True)

    if show_history:
        evaluated = _history_with_actuals(history, actual, models, selected_runs)
        _render_history_metrics(evaluated)

    with st.expander("📋 Ver datos actuales"):
        tables = []
        if "XGB" in models:
            xgb_display = df_xgb[["Fecha", "Pasajeros", "Phase"]].copy()
            xgb_display["Modelo"] = "XGB"
            tables.append(xgb_display)
        if "LSTM" in models:
            lstm_display = df_lstm[["Fecha", "Pasajeros", "Phase"]].copy()
            lstm_display["Modelo"] = "LSTM"
            tables.append(lstm_display)
        display = pd.concat(tables, ignore_index=True)
        display["Fecha"] = pd.to_datetime(display["Fecha"]).dt.to_period("M").astype(str)
        st.dataframe(
            display[["Modelo", "Fecha", "Pasajeros", "Phase"]].sort_values(
                ["Modelo", "Fecha"]
            ),
            use_container_width=True,
            hide_index=True,
        )
