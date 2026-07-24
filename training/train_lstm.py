"""Train and compare LSTM candidates without replacing production."""

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
from tensorflow.keras.layers import Dense, LSTM

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"
WINDOW_SIZE = 12
HOLDOUT_MONTHS = 12
POST_COVID_START = pd.Timestamp("2022-01-01")
FEATURE_COLUMNS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]
DATA_PATH = Path("result_total.csv")
MODEL_DIR = Path("models")
DEFAULT_UNITS = 32
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 8


def set_reproducible_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training dataset: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = (
        df[df["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    if len(df) < 48:
        raise ValueError("At least 48 monthly observations are required.")
    return df


def prepare_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = df if scope == "all_history" else df[df[DATE_COL] >= POST_COVID_START]
    scoped = scoped.reset_index(drop=True).copy()
    minimum_rows = WINDOW_SIZE + HOLDOUT_MONTHS + 24
    if len(scoped) < minimum_rows:
        raise ValueError(f"Not enough observations for scope {scope}: {len(scoped)}")
    scoped["month_sin"] = np.sin(2 * np.pi * scoped[DATE_COL].dt.month / 12.0)
    scoped["month_cos"] = np.cos(2 * np.pi * scoped[DATE_COL].dt.month / 12.0)
    base_year = int(scoped[DATE_COL].dt.year.min())
    scoped["year_norm"] = (scoped[DATE_COL].dt.year - base_year).astype(float)
    return scoped


def build_sequences(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_values = [features[index - WINDOW_SIZE:index] for index in range(WINDOW_SIZE, len(features))]
    y_values = [float(target[index]) for index in range(WINDOW_SIZE, len(features))]
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def build_lstm_model(input_shape: tuple[int, int], units: int) -> Sequential:
    model = Sequential(
        [
            LSTM(units, input_shape=input_shape),
            Dense(max(units // 2, 1), activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_scope(
    source: pd.DataFrame,
    scope: str,
    units: int,
    epochs: int,
    batch_size: int,
) -> tuple[Sequential, MinMaxScaler, dict[str, object]]:
    set_reproducible_seed()
    df = prepare_scope(source, scope)
    split_row = len(df) - HOLDOUT_MONTHS

    scaler = MinMaxScaler()
    scaler.fit(df.loc[: split_row - 1, [TARGET_COL]])
    df["_x_pasaj"] = scaler.transform(df[[TARGET_COL]]).reshape(-1)
    x_all, y_all = build_sequences(
        df[FEATURE_COLUMNS].to_numpy(dtype=float),
        df["_x_pasaj"].to_numpy(dtype=float),
    )
    target_rows = np.arange(WINDOW_SIZE, len(df))
    train_mask = target_rows < split_row
    holdout_mask = target_rows >= split_row
    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_holdout, y_holdout = x_all[holdout_mask], y_all[holdout_mask]

    model = build_lstm_model((WINDOW_SIZE, len(FEATURE_COLUMNS)), units)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_holdout, y_holdout),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5
            ),
        ],
    )

    predictions = scaler.inverse_transform(
        model.predict(x_holdout, verbose=0).reshape(-1, 1)
    ).reshape(-1)
    actuals = scaler.inverse_transform(y_holdout.reshape(-1, 1)).reshape(-1)
    metrics: dict[str, object] = {
        "scope": scope,
        "units": units,
        "epochs_run": int(len(history.history["loss"])),
        "training_start": str(df.iloc[0][DATE_COL].date()),
        "training_end": str(df.iloc[split_row - 1][DATE_COL].date()),
        "holdout_start": str(df.iloc[split_row][DATE_COL].date()),
        "holdout_end": str(df.iloc[-1][DATE_COL].date()),
        "training_sequences": int(len(x_train)),
        "holdout_sequences": int(len(x_holdout)),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "rmse": float(mean_squared_error(actuals, predictions) ** 0.5),
    }
    return model, scaler, metrics


def train_candidates(units: int, epochs: int, batch_size: int) -> dict[str, object]:
    source = load_history()
    results = []
    for scope in ("all_history", "post_covid"):
        model, scaler, metrics = train_scope(
            source, scope, units, epochs, batch_size
        )
        results.append((model, scaler, metrics))
        print(
            f"LSTM {scope}: MAE={metrics['mae']:,.2f}, "
            f"RMSE={metrics['rmse']:,.2f}"
        )

    best_model, best_scaler, best_metrics = min(
        results, key=lambda item: float(item[2]["rmse"])
    )
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "lstm_candidate.keras"
    scaler_path = MODEL_DIR / "scaler_y_candidate.pkl"
    metrics_path = MODEL_DIR / "lstm_candidate_metrics.json"
    best_model.save(model_path)
    joblib.dump(best_scaler, scaler_path)

    report: dict[str, object] = {
        "model": "LSTM",
        "selection_metric": "rmse",
        "selected_scope": best_metrics["scope"],
        "automatic_production_replacement": False,
        "candidates": [metrics for _, _, metrics in results],
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected candidate scope: {best_metrics['scope']}")
    print("Production model was not replaced.")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM candidates.")
    parser.add_argument("--units", type=int, default=DEFAULT_UNITS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_candidates(args.units, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
