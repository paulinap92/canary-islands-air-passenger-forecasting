"""Train an LSTM candidate separately from the monthly inference workflow.

The architecture and sequence features are extracted from the initial model
selection notebook. Retraining keeps the selected LSTM architecture by default;
it does not repeat the original LSTM/GRU/Transformer model search.

The script writes candidate artifacts only and never overwrites the production
LSTM or scaler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
WINDOW_SIZE = 12
FEATURE_COLUMNS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]

DATA_PATH = Path("result_total.csv")
MODEL_DIR = Path("models")
DEFAULT_UNITS = 32
DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 8


def set_reproducible_seed(seed: int = 42) -> None:
    """Set deterministic seeds where supported."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and prepare chronological Total Canarias observations."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training dataset: {data_path}")

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

    if len(df) < 48:
        raise ValueError(
            "At least 48 monthly observations are recommended for LSTM retraining."
        )

    df["month_sin"] = np.sin(2 * np.pi * df[DATE_COL].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df[DATE_COL].dt.month / 12.0)
    base_year = int(df[DATE_COL].dt.year.min())
    df["year_norm"] = (df[DATE_COL].dt.year - base_year).astype(float)
    return df


def build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological rolling sequences and next-month targets."""
    x_values: list[np.ndarray] = []
    y_values: list[float] = []

    for index in range(window_size, len(features)):
        x_values.append(features[index - window_size:index])
        y_values.append(float(target[index]))

    if not x_values:
        raise ValueError("Not enough observations to build LSTM sequences.")

    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def build_lstm_model(input_shape: tuple[int, int], units: int) -> Sequential:
    """Build the selected LSTM architecture from the initial notebook."""
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_candidate(
    units: int = DEFAULT_UNITS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float | int | str]:
    """Train and evaluate a candidate while preserving temporal order."""
    set_reproducible_seed()
    df = load_history()

    holdout_months = 12
    split_row = len(df) - holdout_months
    if split_row <= WINDOW_SIZE:
        raise ValueError("Training period is too short after reserving holdout data.")

    scaler_y = MinMaxScaler()
    scaler_y.fit(df.loc[: split_row - 1, [TARGET_COL]])

    df["_x_pasaj"] = scaler_y.transform(df[[TARGET_COL]]).reshape(-1)
    feature_values = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    target_values = df["_x_pasaj"].to_numpy(dtype=float)

    x_all, y_all = build_sequences(feature_values, target_values)
    sequence_target_rows = np.arange(WINDOW_SIZE, len(df))
    train_mask = sequence_target_rows < split_row
    validation_mask = sequence_target_rows >= split_row

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_validation, y_validation = x_all[validation_mask], y_all[validation_mask]

    if len(x_validation) != holdout_months:
        raise ValueError(
            f"Expected {holdout_months} holdout sequences, got {len(x_validation)}."
        )

    model = build_lstm_model(
        input_shape=(WINDOW_SIZE, len(FEATURE_COLUMNS)),
        units=units,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=10,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1,
        callbacks=callbacks,
    )

    scaled_predictions = model.predict(x_validation, verbose=0).reshape(-1, 1)
    predictions = scaler_y.inverse_transform(scaled_predictions).reshape(-1)
    actuals = scaler_y.inverse_transform(y_validation.reshape(-1, 1)).reshape(-1)

    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(mean_squared_error(actuals, predictions) ** 0.5)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "lstm_candidate.keras"
    scaler_path = MODEL_DIR / "scaler_y_candidate.pkl"
    metrics_path = MODEL_DIR / "lstm_candidate_metrics.json"

    model.save(model_path)
    joblib.dump(scaler_y, scaler_path)

    metrics: dict[str, float | int | str] = {
        "model": "LSTM",
        "units": units,
        "window_size": WINDOW_SIZE,
        "feature_count": len(FEATURE_COLUMNS),
        "train_sequences": int(len(x_train)),
        "validation_sequences": int(len(x_validation)),
        "epochs_run": int(len(history.history["loss"])),
        "mae": mae,
        "rmse": rmse,
        "last_training_month": str(df.iloc[split_row - 1][DATE_COL].date()),
        "first_validation_month": str(df.iloc[split_row][DATE_COL].date()),
        "last_validation_month": str(df.iloc[-1][DATE_COL].date()),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Candidate LSTM MAE: {mae:,.2f}")
    print(f"Candidate LSTM RMSE: {rmse:,.2f}")
    print(f"Candidate model saved to {model_path}")
    print(f"Candidate scaler saved to {scaler_path}")
    print(f"Metrics saved to {metrics_path}")
    print("Review the candidate before replacing production artifacts.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM candidate model.")
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_candidate(
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
