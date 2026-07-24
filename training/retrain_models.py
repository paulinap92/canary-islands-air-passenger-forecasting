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


def scope_data(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    scoped = df if scope == "all_history" else df[df[DATE_COL] >= POST_COVID_START]
    return scoped.reset_index(drop=True)


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


def evaluate_scope(df: pd.DataFrame, scope: str) -> dict[str, object]:
    scoped = scope_data(df, scope)
    if len(scoped) < HOLDOUT_MONTHS + 24:
        raise ValueError(f"Not enough observations for scope {scope}: {len(scoped)}")

    split = len(scoped) - HOLDOUT_MONTHS
    train = scoped.iloc[:split]
    holdout = scoped.iloc[split:]
    model = build_xgb_model()
    model.fit(train[FEATURES].to_numpy(), train[TARGET_COL].to_numpy())
    predictions = model.predict(holdout[FEATURES].to_numpy())

    return {
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


def main() -> None:
    df = load_training_data()
    metrics = [evaluate_scope(df, scope) for scope in ("all_history", "post_covid")]
    for result in metrics:
        print(
            f"XGBoost {result['scope']}: MAE={result['mae']:,.2f}, "
            f"RMSE={result['rmse']:,.2f}"
        )

    best_metrics = min(metrics, key=lambda item: float(item["rmse"]))
    selected_scope = str(best_metrics["scope"])
    final_training_data = scope_data(df, selected_scope)
    final_model = build_xgb_model()
    final_model.fit(
        final_training_data[FEATURES].to_numpy(),
        final_training_data[TARGET_COL].to_numpy(),
    )

    MODEL_DIR.mkdir(exist_ok=True)
    candidate_path = MODEL_DIR / "xgb_candidate.pkl"
    metrics_path = MODEL_DIR / "xgb_candidate_metrics.json"
    joblib.dump(final_model, candidate_path)

    report = {
        "model": "XGBoost",
        "selection_metric": "rmse",
        "selected_scope": selected_scope,
        "final_training_start": str(final_training_data.iloc[0][DATE_COL].date()),
        "final_training_end": str(final_training_data.iloc[-1][DATE_COL].date()),
        "final_training_rows": int(len(final_training_data)),
        "automatic_production_replacement": False,
        "candidates": metrics,
        "candidate_path": str(candidate_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected candidate scope: {selected_scope}")
    print(f"Candidate retrained on all selected-scope data and saved to {candidate_path}")
    print(f"Comparison report saved to {metrics_path}")
    print("Production model was not replaced.")


if __name__ == "__main__":
    main()
