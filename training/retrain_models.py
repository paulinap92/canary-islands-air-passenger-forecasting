"""Explicit model retraining workflow.

This script is intentionally separate from the monthly update. Run it manually
or from a dedicated scheduled job only when a new model candidate should be
trained and evaluated.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_PATH = Path("result_total_with_lags_coded.csv")
MODEL_DIR = Path("models")
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
ISLAND_NAME = "Total Canarias"

FEATURES = [
    "month_sin",
    "month_cos",
    "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3",
    "roll6",
]


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = (
        df[df["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL, *FEATURES])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    if len(df) < 36:
        raise ValueError("Too few monthly observations for a reliable retraining run.")
    return df


def build_xgb_model() -> TransformedTargetRegressor:
    regressor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.7,
            objective="reg:squarederror",
            random_state=42,
        )),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def main() -> None:
    df = load_training_data()
    split = max(24, len(df) - 12)

    train = df.iloc[:split]
    test = df.iloc[split:]

    candidate = build_xgb_model()
    candidate.fit(train[FEATURES].to_numpy(), train[TARGET_COL].to_numpy())

    predictions = candidate.predict(test[FEATURES].to_numpy())
    mae = mean_absolute_error(test[TARGET_COL], predictions)
    rmse = mean_squared_error(test[TARGET_COL], predictions) ** 0.5

    print(f"Candidate XGBoost MAE: {mae:,.2f}")
    print(f"Candidate XGBoost RMSE: {rmse:,.2f}")
    print("Review these metrics before replacing the production model.")

    MODEL_DIR.mkdir(exist_ok=True)
    candidate_path = MODEL_DIR / "xgb_candidate.pkl"
    joblib.dump(candidate, candidate_path)
    print(f"Candidate saved to {candidate_path}")

    # LSTM retraining remains documented in notebooks/my_models_trials.ipynb.
    # It should be extracted into a dedicated training module before it is
    # automated, so the architecture and validation procedure stay explicit.


if __name__ == "__main__":
    main()
