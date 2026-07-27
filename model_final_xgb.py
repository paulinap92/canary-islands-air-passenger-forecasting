"""Generate the Total Canarias forecast with the persisted XGBoost model.

This file performs inference only. It reads the current monthly totals, creates
the same features as ``model_train_xgb.py``, loads ``models/xgb_best.pkl`` and
forecasts the next months iteratively without retraining the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path(os.getenv("XGB_FORECAST_DATA_PATH", "result_total.csv"))
MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", "models/xgb_best.pkl"))
OUTPUT_PATH = Path(
    os.getenv("XGB_FORECAST_OUTPUT_PATH", "forecast_total_canarias_xgb.csv")
)
FORECAST_MONTHS = int(os.getenv("FORECAST_MONTHS", "12"))

FEATURES = [
    "month_sin",
    "month_cos",
    "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3",
    "roll6",
]


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the complete monthly history for Total Canarias."""
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")

    raw = pd.read_csv(data_path, encoding="utf-8-sig")
    required = {"Isla", DATE_COL, TARGET_COL}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"Faltan columnas en {data_path}: {missing}")

    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw[TARGET_COL] = pd.to_numeric(raw[TARGET_COL], errors="coerce")
    history = (
        raw[raw["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    if history.empty:
        raise ValueError(f"No hay datos para {ISLAND_NAME}.")

    if history[DATE_COL].duplicated().any():
        duplicated = history.loc[
            history[DATE_COL].duplicated(keep=False), DATE_COL
        ]
        raise ValueError(
            f"Hay meses duplicados: {duplicated.dt.strftime('%Y-%m').tolist()}"
        )

    periods = pd.PeriodIndex(history[DATE_COL], freq="M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    missing_periods = expected.difference(periods)
    if len(missing_periods):
        raise ValueError(
            "La serie mensual no es continua. Meses ausentes: "
            f"{[str(period) for period in missing_periods]}"
        )

    if len(history) < 12:
        raise ValueError("Se requieren al menos 12 meses históricos.")

    return history


def build_feature_row(
    values: list[float],
    next_date: pd.Timestamp,
    first_history_year: int,
) -> dict[str, float]:
    """Build one inference row using the same feature definitions as training."""
    if len(values) < 12:
        raise ValueError("No hay suficientes valores para crear 12 lags.")

    row: dict[str, float] = {
        "month_sin": float(np.sin(2 * np.pi * next_date.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * next_date.month / 12.0)),
        "year_norm": float(next_date.year - first_history_year),
    }

    for lag in range(1, 13):
        row[f"lag_{lag}"] = float(values[-lag])

    row["roll3"] = float(np.mean(values[-3:]))
    row["roll6"] = float(np.mean(values[-6:]))
    return row


def generate_forecast(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    forecast_months: int = FORECAST_MONTHS,
) -> pd.DataFrame:
    """Load the persisted model and forecast future months iteratively."""
    if forecast_months < 1:
        raise ValueError("FORECAST_MONTHS debe ser al menos 1.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo guardado: {model_path}. "
            "Ejecuta model_train_xgb.py antes del forecast."
        )

    history = load_history(data_path)
    model = joblib.load(model_path)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(FEATURES):
        raise ValueError(
            f"El modelo espera {expected_features} variables, "
            f"pero el forecast proporciona {len(FEATURES)}."
        )

    last_real_date = history[DATE_COL].max()
    first_history_year = int(history[DATE_COL].dt.year.min())
    values = history[TARGET_COL].astype(float).tolist()
    future_dates = pd.date_range(
        start=last_real_date + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )

    forecast_rows: list[dict[str, object]] = []
    for next_date in future_dates:
        feature_row = build_feature_row(values, next_date, first_history_year)
        x_pred = np.asarray(
            [[feature_row[feature] for feature in FEATURES]],
            dtype=float,
        )
        prediction = max(float(model.predict(x_pred)[0]), 0.0)
        values.append(prediction)
        forecast_rows.append(
            {
                "Isla": ISLAND_NAME,
                DATE_COL: next_date,
                TARGET_COL: prediction,
                **feature_row,
                "Phase": "Forecast",
            }
        )

    history_output = history[["Isla", DATE_COL, TARGET_COL]].copy()
    history_output["Phase"] = "History"
    forecast_output = pd.DataFrame(forecast_rows)

    return pd.concat(
        [history_output, forecast_output],
        ignore_index=True,
        sort=False,
    )


def main() -> None:
    forecast = generate_forecast()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    forecast_rows = forecast[forecast["Phase"] == "Forecast"]
    last_history_date = forecast.loc[
        forecast["Phase"] == "History", DATE_COL
    ].max()
    forecast_end = forecast_rows[DATE_COL].max()

    print(f"✅ Modelo cargado desde {MODEL_PATH}")
    print(
        f"📈 Forecast: {last_history_date.date()} → {forecast_end.date()} "
        f"({len(forecast_rows)} meses)."
    )
    print(f"💾 Guardado en {OUTPUT_PATH}")
    print(forecast_rows[[DATE_COL, TARGET_COL]].to_string(index=False))


if __name__ == "__main__":
    main()
