"""Train and persist the final XGBoost model for Total Canarias.

The training range starts after the COVID period. The end date and the
12-month holdout are calculated automatically from the newest available row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ISLAND_NAME = "Total Canarias"
TARGET_COL = "Pasajeros"
DATE_COL = "Fecha"

DATA_PATH = Path(os.getenv("XGB_TRAINING_DATA_PATH", "result_total_with_lags_coded.csv"))
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


def build_model() -> TransformedTargetRegressor:
    """Create the final XGBoost configuration selected in model trials."""
    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.03,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=xgb,
        transformer=StandardScaler(),
    )


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load monthly Total Canarias rows and remove the COVID training period."""
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrenamiento: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    required = {"Isla", DATE_COL, TARGET_COL, *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Faltan columnas en {data_path}: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = (
        df[(df["Isla"] == ISLAND_NAME) & (df[DATE_COL] >= TRAIN_START_DATE)]
        .dropna(subset=[DATE_COL, TARGET_COL, *FEATURES])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    if df[DATE_COL].duplicated().any():
        duplicated = df.loc[df[DATE_COL].duplicated(keep=False), DATE_COL]
        raise ValueError(f"Hay meses duplicados: {duplicated.dt.strftime('%Y-%m').tolist()}")

    periods = df[DATE_COL].dt.to_period("M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if not periods.reset_index(drop=True).equals(pd.Series(expected)):
        missing_periods = expected.difference(pd.PeriodIndex(periods))
        raise ValueError(
            "La serie mensual no es continua. Meses ausentes: "
            f"{[str(period) for period in missing_periods]}"
        )

    minimum_rows = TEST_MONTHS + 12
    if len(df) < minimum_rows:
        raise ValueError(
            f"Se requieren al menos {minimum_rows} meses posteriores a "
            f"{TRAIN_START_DATE.date()}, pero hay {len(df)}."
        )

    return df


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
    """Calculate interpretable regression metrics for the holdout."""
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
    """Evaluate on a dynamic holdout, then refit and save on all post-COVID rows."""
    df = load_training_data(data_path)
    train_df, test_df = split_dynamic_holdout(df)

    evaluation_model = build_model()
    evaluation_model.fit(
        train_df[FEATURES].to_numpy(dtype=float),
        train_df[TARGET_COL].to_numpy(dtype=float),
    )
    predictions = evaluation_model.predict(
        test_df[FEATURES].to_numpy(dtype=float)
    )
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
        "✅ XGBoost entrenado sin el período COVID "
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
