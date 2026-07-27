"""Train and persist the final XGBoost model for Total Canarias.

The script reads the current monthly totals, creates the lag and rolling
features used in ``my_models_trials.ipynb``, excludes COVID target rows,
evaluates on the latest 12 months and finally fits the persisted model on all
eligible post-COVID target rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path(os.getenv("XGB_TRAINING_DATA_PATH", "result_total.csv"))
MODEL_PATH = Path(os.getenv("XGB_MODEL_PATH", "models/xgb_best.pkl"))
METRICS_PATH = Path(os.getenv("XGB_METRICS_PATH", "models/xgb_training_metrics.json"))
TRAIN_START_DATE = pd.Timestamp(os.getenv("TRAIN_START_DATE", "2022-01-01"))
TEST_MONTHS = int(os.getenv("TEST_MONTHS", "12"))

FEATURES = [
    "month_sin",
    "month_cos",
    "year_norm",
    *[f"lag_{i}" for i in range(1, 13)],
    "roll3",
    "roll6",
]


def build_model() -> Pipeline:
    """Create the XGBoost pipeline used in ``my_models_trials.ipynb``."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    objective="reg:squarederror",
                ),
            ),
        ]
    )


def validate_monthly_history(df: pd.DataFrame) -> None:
    """Reject duplicated or missing months before creating lag features."""
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


def create_features(history: pd.DataFrame) -> pd.DataFrame:
    """Create seasonality, yearly trend, lag and rolling features."""
    df = history.copy()
    month = df[DATE_COL].dt.month
    first_history_year = int(df[DATE_COL].dt.year.min())

    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["year_norm"] = df[DATE_COL].dt.year - first_history_year

    for lag in range(1, 13):
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    shifted_target = df[TARGET_COL].shift(1)
    df["roll3"] = shifted_target.rolling(window=3).mean()
    df["roll6"] = shifted_target.rolling(window=6).mean()
    return df


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Build features from current totals, then retain post-COVID target rows."""
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
        raw[raw["Isla"] == ISLAND_NAME]
        .dropna(subset=[DATE_COL, TARGET_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    validate_monthly_history(history)

    featured = create_features(history)
    training = (
        featured[featured[DATE_COL] >= TRAIN_START_DATE]
        .dropna(subset=[TARGET_COL, *FEATURES])
        .reset_index(drop=True)
    )

    minimum_rows = TEST_MONTHS + 12
    if len(training) < minimum_rows:
        raise ValueError(
            f"Se requieren al menos {minimum_rows} meses de target posteriores a "
            f"{TRAIN_START_DATE.date()}, pero hay {len(training)}."
        )

    return training


def split_dynamic_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the latest complete TEST_MONTHS as a chronological holdout."""
    last_date = df[DATE_COL].max()
    test_start = (last_date.to_period("M") - (TEST_MONTHS - 1)).to_timestamp()

    train_df = df[df[DATE_COL] < test_start].copy()
    test_df = df[df[DATE_COL] >= test_start].copy()

    if len(test_df) != TEST_MONTHS:
        raise ValueError(
            f"El holdout debería contener {TEST_MONTHS} meses, pero contiene {len(test_df)}."
        )
    if train_df.empty:
        raise ValueError("El conjunto de entrenamiento quedó vacío.")

    return train_df, test_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics for the automatic holdout."""
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
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    """Evaluate, then refit and save using all eligible post-COVID target rows."""
    df = load_training_data(data_path)
    train_df, test_df = split_dynamic_holdout(df)

    evaluation_model = build_model()
    evaluation_model.fit(
        train_df[FEATURES].to_numpy(dtype=float),
        train_df[TARGET_COL].to_numpy(dtype=float),
    )
    predictions = evaluation_model.predict(test_df[FEATURES].to_numpy(dtype=float))
    metrics = calculate_metrics(
        test_df[TARGET_COL].to_numpy(dtype=float),
        np.asarray(predictions, dtype=float),
    )

    final_model = build_model()
    final_model.fit(
        df[FEATURES].to_numpy(dtype=float),
        df[TARGET_COL].to_numpy(dtype=float),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)

    report: dict[str, object] = {
        "island": ISLAND_NAME,
        "source_data": str(data_path),
        "training_start": df[DATE_COL].min().strftime("%Y-%m-%d"),
        "training_end": df[DATE_COL].max().strftime("%Y-%m-%d"),
        "training_rows_final_model": int(len(df)),
        "holdout_start": test_df[DATE_COL].min().strftime("%Y-%m-%d"),
        "holdout_end": test_df[DATE_COL].max().strftime("%Y-%m-%d"),
        "holdout_months": int(len(test_df)),
        **metrics,
        "model_path": str(model_path),
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = train_and_save()
    print(
        "✅ XGBoost entrenado con targets posteriores al COVID "
        f"({report['training_start']} → {report['training_end']})."
    )
    print(
        "🧪 Holdout automático: "
        f"{report['holdout_start']} → {report['holdout_end']} "
        f"({report['holdout_months']} meses)."
    )
    print(
        f"MAE={report['mae']:.2f} | RMSE={report['rmse']:.2f} | "
        f"MAPE={report['mape_pct']:.2f}%"
    )
    print(f"💾 Modelo guardado en {report['model_path']}")


if __name__ == "__main__":
    main()
