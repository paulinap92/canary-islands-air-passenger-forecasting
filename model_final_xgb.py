"""Generate an XGBoost forecast using the persisted production model."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast_horizon import build_future_dates

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path("result_total_with_lags_coded.csv")
MODEL_PATH = Path("models/xgb_best.pkl")
OUTPUT_PATH = Path("forecast_total_canarias_xgb.csv")
MIN_FORECAST_HORIZON_MONTHS = int(os.getenv("FORECAST_HORIZON_MONTHS", "12"))
FORECAST_END_DATE = os.getenv("FORECAST_END_DATE")

FEATURES = [
    "month_sin",
    "month_cos",
    "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3",
    "roll6",
]


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load validated historical data for Total Canarias."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run download_agent.build_features() first."
        )

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    required = {DATE_COL, TARGET_COL, "Isla", *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing columns in {data_path}: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = (
        df[df["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL, *FEATURES])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    if len(df) < 12:
        raise ValueError("At least 12 historical months are required for forecasting.")
    return df


def generate_forecast(
    minimum_horizon_months: int = MIN_FORECAST_HORIZON_MONTHS,
    forecast_end_date: str | None = FORECAST_END_DATE,
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> pd.DataFrame:
    """Generate an iterative monthly forecast without retraining the model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing production model {model_path}. "
            "Run training/retrain_models.py explicitly when retraining is required."
        )

    df = load_history(data_path)
    model = joblib.load(model_path)
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and expected_features != len(FEATURES):
        raise ValueError(
            f"The saved XGBoost model expects {expected_features} features, "
            f"but inference provides {len(FEATURES)}."
        )

    df_future = df.copy()
    last_real_date = df_future[DATE_COL].max()
    last_real_year_norm = float(df_future.iloc[-1]["year_norm"])
    future_dates = build_future_dates(
        last_real_date,
        minimum_horizon_months,
        forecast_end_date,
    )

    for next_date in future_dates:
        new_row = df_future.iloc[-1].copy()
        new_row[DATE_COL] = next_date
        new_row["Isla"] = ISLAND_NAME
        new_row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12.0)
        new_row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12.0)
        new_row["year_norm"] = (
            last_real_year_norm + next_date.year - last_real_date.year
        )

        for lag in range(1, 13):
            new_row[f"lag_{lag}"] = df_future[TARGET_COL].iloc[-lag]

        recent_values = df_future[TARGET_COL].tail(6).to_numpy(dtype=float)
        new_row["roll3"] = float(np.mean(recent_values[-3:]))
        new_row["roll6"] = float(np.mean(recent_values[-6:]))

        x_pred = np.asarray(
            [[float(new_row[feature]) for feature in FEATURES]],
            dtype=float,
        )
        new_row[TARGET_COL] = max(float(model.predict(x_pred)[0]), 0.0)
        df_future = pd.concat(
            [df_future, pd.DataFrame([new_row])], ignore_index=True
        )

    df_future["Phase"] = np.where(
        df_future[DATE_COL] <= last_real_date, "History", "Forecast"
    )
    return df_future


def main() -> None:
    forecast = generate_forecast()
    forecast.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    forecast_rows = forecast[forecast["Phase"] == "Forecast"]
    last_history_date = forecast.loc[
        forecast["Phase"] == "History", DATE_COL
    ].max()
    final_forecast_date = forecast_rows[DATE_COL].max()
    print(
        f"Saved {len(forecast_rows)} XGBoost forecast months to {OUTPUT_PATH} "
        f"(last real month: {last_history_date.date()}, "
        f"forecast end: {final_forecast_date.date()})."
    )


if __name__ == "__main__":
    main()
