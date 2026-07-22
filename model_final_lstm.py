"""Generate an LSTM forecast using persisted production artifacts.

This module performs inference only. LSTM retraining is isolated in
``training/train_lstm.py`` and must not run during monthly data updates.
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

DATA_PATH = Path("result_total.csv")
MODEL_PATH = Path("models/lstm_best.h5")
SCALER_PATH = Path("models/scaler_y.pkl")
OUTPUT_PATH = Path("forecast_total_canarias_lstm.csv")

WINDOW_SIZE = 12
FORECAST_HORIZON_MONTHS = int(os.getenv("FORECAST_HORIZON_MONTHS", "12"))
FEATURE_COLUMNS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and prepare historical data exactly as required by the LSTM."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing historical dataset: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    required = {"Isla", DATE_COL, TARGET_COL}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns in {data_path}: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = (
        df[df["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"At least {WINDOW_SIZE} historical months are required for LSTM inference."
        )

    df["month_sin"] = np.sin(2 * np.pi * df[DATE_COL].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df[DATE_COL].dt.month / 12.0)

    # The initial training notebook used a zero-based year normalization.
    base_year = int(df[DATE_COL].dt.year.min())
    df["year_norm"] = (df[DATE_COL].dt.year - base_year).astype(float)
    return df


def generate_forecast(
    horizon_months: int = FORECAST_HORIZON_MONTHS,
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> pd.DataFrame:
    """Generate an iterative forecast without fitting or modifying the model."""
    if horizon_months < 1:
        raise ValueError("horizon_months must be at least 1.")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing production LSTM model: {model_path}. "
            "Run training/train_lstm.py explicitly when retraining is required."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing production target scaler: {scaler_path}")

    df = load_history(data_path)
    model = load_model(model_path, compile=False)
    scaler_y = joblib.load(scaler_path)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) != 3:
        raise ValueError(f"Unexpected LSTM input shape: {input_shape}")
    if input_shape[1] not in (None, WINDOW_SIZE):
        raise ValueError(
            f"The saved model expects a window of {input_shape[1]}, "
            f"but inference uses {WINDOW_SIZE}."
        )
    if input_shape[2] not in (None, len(FEATURE_COLUMNS)):
        raise ValueError(
            f"The saved model expects {input_shape[2]} features, "
            f"but inference provides {len(FEATURE_COLUMNS)}."
        )

    scaled_target = scaler_y.transform(df[[TARGET_COL]]).reshape(-1)
    df["_x_pasaj"] = scaled_target

    sequence = (
        df[FEATURE_COLUMNS]
        .tail(WINDOW_SIZE)
        .to_numpy(dtype=float)
        .reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))
    )

    last_real_date = df[DATE_COL].max()
    future_dates = pd.date_range(
        start=last_real_date + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq="MS",
    )
    base_year = int(df[DATE_COL].dt.year.min())
    df_future = df.copy()

    for next_date in future_dates:
        month_sin = np.sin(2 * np.pi * next_date.month / 12.0)
        month_cos = np.cos(2 * np.pi * next_date.month / 12.0)
        year_norm = float(next_date.year - base_year)

        scaled_prediction = float(model.predict(sequence, verbose=0)[0][0])
        prediction = float(
            scaler_y.inverse_transform([[scaled_prediction]])[0][0]
        )
        prediction = max(prediction, 0.0)

        next_row = {
            "Isla": ISLAND_NAME,
            DATE_COL: next_date,
            TARGET_COL: prediction,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "year_norm": year_norm,
            "_x_pasaj": scaled_prediction,
        }
        df_future = pd.concat(
            [df_future, pd.DataFrame([next_row])],
            ignore_index=True,
        )

        next_step = np.asarray(
            [[scaled_prediction, month_sin, month_cos, year_norm]],
            dtype=float,
        ).reshape(1, 1, len(FEATURE_COLUMNS))
        sequence = np.concatenate([sequence[:, 1:, :], next_step], axis=1)

    df_future["Phase"] = np.where(
        df_future[DATE_COL] <= last_real_date,
        "History",
        "Forecast",
    )
    return df_future


def main() -> None:
    """Generate and save the production LSTM forecast."""
    forecast = generate_forecast()
    forecast.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    forecast_rows = forecast[forecast["Phase"] == "Forecast"]
    last_history_date = forecast.loc[
        forecast["Phase"] == "History", DATE_COL
    ].max()
    print(
        f"Saved {len(forecast_rows)} LSTM forecast months to {OUTPUT_PATH} "
        f"(last real month: {last_history_date.date()})."
    )


if __name__ == "__main__":
    main()
