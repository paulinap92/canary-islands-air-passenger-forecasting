"""Train and persist the final LSTM model for Total Canarias.

The architecture and training parameters come from ``my_models_trials.ipynb``:
12-month sequences, four input features and the best LSTM variant (32 units).
COVID rows are excluded from both the target and the sequence memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path(os.getenv("LSTM_TRAINING_DATA_PATH", "result_total.csv"))
MODEL_PATH = Path(os.getenv("LSTM_MODEL_PATH", "models/lstm_best.h5"))
SCALER_PATH = Path(os.getenv("LSTM_SCALER_PATH", "models/scaler_y.pkl"))
METRICS_PATH = Path(
    os.getenv("LSTM_METRICS_PATH", "models/lstm_training_metrics.json")
)
TRAIN_START_DATE = pd.Timestamp(os.getenv("TRAIN_START_DATE", "2022-01-01"))
TEST_MONTHS = int(os.getenv("TEST_MONTHS", "12"))

WIN = 12
UNITS = 32
VAL_FRAC = 0.15
EPOCHS = 150
BATCH_SIZE = 32
SEED = 42
FEATURE_COLS = ["_x_pasaj", "month_sin", "month_cos", "year_norm"]


def set_random_seed() -> None:
    """Set deterministic seeds used in the model-trials notebook."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)


def validate_monthly_history(df: pd.DataFrame) -> None:
    """Reject duplicated or missing months."""
    if df.empty:
        raise ValueError(f"No hay datos para {ISLAND_NAME}.")

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


def load_post_covid_history(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load Total Canarias and retain only rows from January 2022 onward."""
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {data_path}")

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

    minimum_rows = WIN + TEST_MONTHS + 2
    if len(history) < minimum_rows:
        raise ValueError(
            f"Se requieren al menos {minimum_rows} meses desde "
            f"{TRAIN_START_DATE.date()}, pero hay {len(history)}."
        )
    return history


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three calendar features used in model trials."""
    featured = df.copy()
    month = featured[DATE_COL].dt.month
    featured["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    featured["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    featured["year_norm"] = (
        featured[DATE_COL].dt.year - TRAIN_START_DATE.year
    ).astype(float)
    return featured


def prepare_scaled_features(
    history: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Transform the target and create the four LSTM input features."""
    featured = add_calendar_features(history)
    featured["_y_scaled"] = scaler.transform(featured[[TARGET_COL]]).reshape(-1)
    featured["_x_pasaj"] = featured["_y_scaled"]
    return featured


def make_sequences(
    featured: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create 12-month input windows and their target dates."""
    values = featured[FEATURE_COLS].to_numpy(dtype=float)
    targets = featured["_y_scaled"].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(featured[DATE_COL])

    x_sequences: list[np.ndarray] = []
    y_sequences: list[float] = []
    target_dates: list[pd.Timestamp] = []
    for index in range(WIN, len(featured)):
        x_sequences.append(values[index - WIN : index])
        y_sequences.append(float(targets[index]))
        target_dates.append(dates[index])

    return (
        np.asarray(x_sequences, dtype=float),
        np.asarray(y_sequences, dtype=float),
        pd.DatetimeIndex(target_dates),
    )


def build_model() -> models.Model:
    """Build the LSTM_32 architecture selected in model trials."""
    inputs = layers.Input(shape=(WIN, len(FEATURE_COLS)))
    hidden = layers.LSTM(UNITS)(inputs)
    hidden = layers.Dense(UNITS // 2, activation="relu")(hidden)
    output = layers.Dense(1)(hidden)
    model = models.Model(inputs, output, name=f"LSTM_{UNITS}")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_callbacks() -> list[callbacks.Callback]:
    """Return the callbacks used in the model-trials notebook."""
    return [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=8),
    ]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate holdout errors in the original passenger scale."""
    if np.any(y_true == 0):
        raise ValueError("MAPE no se puede calcular porque el holdout contiene ceros.")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mape_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
    }


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    """Evaluate on the latest 12 months, then train the persisted final model."""
    set_random_seed()
    history = load_post_covid_history(data_path)
    last_date = history[DATE_COL].max()
    holdout_start = (
        last_date.to_period("M") - (TEST_MONTHS - 1)
    ).to_timestamp()

    evaluation_train_rows = history[history[DATE_COL] < holdout_start].copy()
    evaluation_scaler = StandardScaler()
    evaluation_scaler.fit(evaluation_train_rows[[TARGET_COL]])
    evaluation_features = prepare_scaled_features(history, evaluation_scaler)
    x_all, y_all, target_dates = make_sequences(evaluation_features)

    train_mask = target_dates < holdout_start
    test_mask = target_dates >= holdout_start
    x_train_all, y_train_all = x_all[train_mask], y_all[train_mask]
    x_test, y_test_scaled = x_all[test_mask], y_all[test_mask]
    test_dates = target_dates[test_mask]

    if len(x_test) != TEST_MONTHS:
        raise ValueError(
            f"El holdout debería contener {TEST_MONTHS} secuencias, "
            f"pero contiene {len(x_test)}."
        )

    validation_rows = max(1, int(len(x_train_all) * VAL_FRAC))
    if len(x_train_all) <= validation_rows:
        raise ValueError("No hay suficientes secuencias para entrenamiento y validación.")

    x_fit = x_train_all[:-validation_rows]
    y_fit = y_train_all[:-validation_rows]
    x_val = x_train_all[-validation_rows:]
    y_val = y_train_all[-validation_rows:]

    evaluation_model = build_model()
    history_fit = evaluation_model.fit(
        x_fit,
        y_fit,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=build_callbacks(),
        shuffle=False,
    )
    best_epoch = int(np.argmin(history_fit.history["val_loss"]) + 1)

    predicted_scaled = evaluation_model.predict(x_test, verbose=0).reshape(-1, 1)
    predicted = evaluation_scaler.inverse_transform(predicted_scaled).reshape(-1)
    actual = evaluation_scaler.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).reshape(-1)
    metrics = calculate_metrics(actual, predicted)

    final_scaler = StandardScaler()
    final_scaler.fit(history[[TARGET_COL]])
    final_features = prepare_scaled_features(history, final_scaler)
    x_final, y_final, final_target_dates = make_sequences(final_features)

    set_random_seed()
    final_model = build_model()
    final_model.fit(
        x_final,
        y_final,
        epochs=best_epoch,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(model_path)
    joblib.dump(final_scaler, scaler_path)

    report: dict[str, object] = {
        "island": ISLAND_NAME,
        "source_data": str(data_path),
        "history_start": history[DATE_COL].min().strftime("%Y-%m-%d"),
        "history_end": history[DATE_COL].max().strftime("%Y-%m-%d"),
        "first_sequence_target": final_target_dates.min().strftime("%Y-%m-%d"),
        "final_training_sequences": int(len(x_final)),
        "holdout_start": test_dates.min().strftime("%Y-%m-%d"),
        "holdout_end": test_dates.max().strftime("%Y-%m-%d"),
        "holdout_months": int(len(test_dates)),
        "best_epoch": best_epoch,
        "window_months": WIN,
        "units": UNITS,
        **metrics,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = train_and_save()
    print(
        "✅ LSTM_32 entrenado sin el período COVID "
        f"({report['history_start']} → {report['history_end']})."
    )
    print(
        "🧪 Holdout automático: "
        f"{report['holdout_start']} → {report['holdout_end']} "
        f"({report['holdout_months']} meses)."
    )
    print(
        f"MAE={report['mae']:.2f} | RMSE={report['rmse']:.2f} | "
        f"MAPE={report['mape_pct']:.2f}% | best_epoch={report['best_epoch']}"
    )
    print(f"💾 Modelo guardado en {report['model_path']}")
    print(f"💾 Scaler guardado en {report['scaler_path']}")


if __name__ == "__main__":
    main()
