"""Train and compare XGBoost candidates without replacing production."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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
HOLDOUT_MONTHS = 12
POST_COVID_START = pd.Timestamp("2022-01-01")

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
        raise ValueError("Too few monthly observations for reliable retraining.")
    return df


def build_xgb_model() -> TransformedTargetRegressor:
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBRegressor(
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def evaluate_scope(df: pd.DataFrame, scope: str) -> tuple[object, dict[str, object]]:
    scoped = df if scope == "all_history" else df[df[DATE_COL] >= POST_COVID_START]
    scoped = scoped.reset_index(drop=True)
    if len(scoped) < HOLDOUT_MONTHS + 24:
        raise ValueError(f"Not enough observations for scope {scope}: {len(scoped)}")

    split = len(scoped) - HOLDOUT_MONTHS
    train = scoped.iloc[:split]
    holdout = scoped.iloc[split:]
    model = build_xgb_model()
    model.fit(train[FEATURES].to_numpy(), train[TARGET_COL].to_numpy())
    predictions = model.predict(holdout[FEATURES].to_numpy())

    metrics: dict[str, object] = {
        "scope": scope,
        "training_start": str(train.iloc[0][DATE_COL].date()),
        "training_end": str(train.iloc[-1][DATE_COL].date()),
        "holdout_start": str(holdout.iloc[0][DATE_COL].date()),
        "holdout_end": str(holdout.iloc[-1][DATE_COL].date()),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "mae": float(mean_absolute_error(holdout[TARGET_COL], predictions)),
        "rmse": float(mean_squared_error(holdout[TARGET_COL], predictions) ** 0.5),
    }
    return model, metrics


def main() -> None:
    df = load_training_data()
    results: list[tuple[object, dict[str, object]]] = []
    for scope in ("all_history", "post_covid"):
        model, metrics = evaluate_scope(df, scope)
        results.append((model, metrics))
        print(
            f"XGBoost {scope}: MAE={metrics['mae']:,.2f}, "
            f"RMSE={metrics['rmse']:,.2f}"
        )

    best_model, best_metrics = min(results, key=lambda item: float(item[1]["rmse"]))
    MODEL_DIR.mkdir(exist_ok=True)
    candidate_path = MODEL_DIR / "xgb_candidate.pkl"
    metrics_path = MODEL_DIR / "xgb_candidate_metrics.json"
    joblib.dump(best_model, candidate_path)

    report = {
        "model": "XGBoost",
        "selection_metric": "rmse",
        "selected_scope": best_metrics["scope"],
        "automatic_production_replacement": False,
        "candidates": [metrics for _, metrics in results],
        "candidate_path": str(candidate_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected candidate scope: {best_metrics['scope']}")
    print(f"Candidate saved to {candidate_path}")
    print(f"Comparison report saved to {metrics_path}")
    print("Production model was not replaced.")


if __name__ == "__main__":
    main()
