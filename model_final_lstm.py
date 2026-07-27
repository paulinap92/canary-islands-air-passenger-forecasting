"""Generate the Total Canarias forecast with the persisted LSTM model.

This file performs inference only. It reads the current post-COVID monthly
history, loads ``models/lstm_best.h5`` and ``models/scaler_y.pkl`` and forecasts
the next months iteratively without retraining.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path(os.getenv("LSTM_FORECAST_DATA_PATH", "result_total.csv"))
MODEL_PATH = Path(os.getenv("LSTM_MODEL_PATH", "models/lstm_best.h5"))
SCALER_PATH = Path(os.getenv("LSTM_SCALER_PATH", "models/scaler_y.pkl"))
OUTPUT_PATH = Path(
    os.getenv("LSTM_FORECAST_OUTPUT_PATH", "forecast_total_canarias_lstm.csv")
)
TRAIN_START_DATE = pd.Timestamp(os.getenv("TRAIN_START_DATE", "2022-01-01"))
FORECAST_MONTHS = int(os.getenv("FORECAST_MONTHS", "12"))

WIN = 12
FEATURE_COLS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]


def validate_monthly_history(df: pd.DataFrame) -> None:
    """Reject duplicated or missing months."""
    if df.empty:
        raise ValueError(f"No hay datos para {ISLAND_NAME} desde {TRAIN_START_DATE.date()}.")

    if df[DATE_COL].duplicated().any():
        duplicated = df.loc[df[DATE_COL].duplicated(keep=False), DATE_COL]
        raise ValueError(
            f"Hay meses duplicados: {duplicated.dt.strftime('%Y-%m').tolist()}"
        )

    periods = pd.PeriodIndex(df[DATE_COL], freq="M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    missing_periods = expected.difference(periods)
    if len(missing_periods):
        raise ValueError(
            "La serie mensual no es continua. Meses ausentes: "
            f"{[str(period) for period in missing_periods]}"
        )

    if len(df) < WIN:
        raise ValueError(
            f"Se requieren al menos {WIN} meses para crear la secuencia inicial."
        )


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load current Total Canarias history from January 2022 onward."""
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
        raw[
            (raw["Isla"] == ISLAND_NAME)
            & (raw[DATE_COL] >= TRAIN_START_DATE)
        ]
        .dropna(subset=[DATE_COL, TARGET_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    validate_monthly_history(history)
    return history


def add_features(history: pd.DataFrame, scaler: object) -> pd.DataFrame:
    """Create exactly the four LSTM input features used during training."""
    featured = history.copy()
    month = featured[DATE_COL].dt.month
    featured["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    featured["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    featured["year_norm"] = (
        featured[DATE_COL].dt.year - TRAIN_START_DATE.year
    ).astype(float)
    featured["_x_pasaj"] = scaler.transform(featured[[TARGET_COL]]).reshape(-1)
    return featured


def generate_forecast(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    forecast_months: int = FORECAST_MONTHS,
) -> pd.DataFrame:
    """Forecast the next months recursively without fitting the model."""
    if forecast_months < 1:
        raise ValueError("FORECAST_MONTHS debe ser mayor que cero.")
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo guardado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No se encontró el scaler guardado: {scaler_path}")

    history = load_history(data_path)
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    featured = add_features(history, scaler)

    expected_shape = (WIN, len(FEATURE_COLS))
    model_input_shape = tuple(model.input_shape[1:])
    if model_input_shape != expected_shape:
        raise ValueError(
            f"El modelo espera secuencias {model_input_shape}, "
            f"pero el forecast genera {expected_shape}."
        )

    sequence = featured[FEATURE_COLS].tail(WIN).to_numpy(dtype=float)
    sequence = sequence.reshape(1, WIN, len(FEATURE_COLS))

    last_real_date = history[DATE_COL].max()
    future_dates = pd.date_range(
        start=last_real_date + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )

    forecast_rows: list[dict[str, object]] = []
    for next_date in future_dates:
        predicted_scaled = float(model.predict(sequence, verbose=0)[0][0])
        predicted = max(
            float(scaler.inverse_transform([[predicted_scaled]])[0][0]),
            0.0,
        )

        month_sin = float(np.sin(2 * np.pi * next_date.month / 12.0))
        month_cos = float(np.cos(2 * np.pi * next_date.month / 12.0))
        year_norm = float(next_date.year - TRAIN_START_DATE.year)

        forecast_rows.append(
            {
                "Isla": ISLAND_NAME,
                DATE_COL: next_date,
                TARGET_COL: predicted,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "year_norm": year_norm,
                "_x_pasaj": predicted_scaled,
            }
        )

        next_step = np.asarray(
            [[predicted_scaled, month_sin, month_cos, year_norm]],
            dtype=float,
        ).reshape(1, 1, len(FEATURE_COLS))
        sequence = np.concatenate([sequence[:, 1:, :], next_step], axis=1)

    history_output = history.copy()
    history_output["Phase"] = "History"
    forecast_output = pd.DataFrame(forecast_rows)
    forecast_output["Phase"] = "Forecast"

    output_columns = ["Isla", DATE_COL, TARGET_COL, "Phase"]
    return pd.concat(
        [history_output[output_columns], forecast_output[output_columns]],
        ignore_index=True,
    )


def main() -> None:
    forecast = generate_forecast()
    forecast.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    history_rows = forecast[forecast["Phase"] == "History"]
    forecast_rows = forecast[forecast["Phase"] == "Forecast"]
    print(f"✅ Modelo LSTM cargado desde {MODEL_PATH}")
    print(
        f"📅 Último mes real: {history_rows[DATE_COL].max().date()} | "
        f"pronóstico: {forecast_rows[DATE_COL].min().date()} → "
        f"{forecast_rows[DATE_COL].max().date()}"
    )
    print(f"💾 Guardados {len(forecast_rows)} meses en {OUTPUT_PATH}")
    print(forecast_rows[[DATE_COL, TARGET_COL]].to_string(index=False))


if __name__ == "__main__":
    main()
