"""Append production forecasts to an immutable forecast history.

The current forecast CSV files are regenerated on every monthly update. This
module preserves what each production model predicted at each forecast origin
so future actual values can be compared with forecasts made earlier.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

HISTORY_PATH = Path("forecast_history.csv")
XGB_FORECAST_PATH = Path("forecast_total_canarias_xgb.csv")
LSTM_FORECAST_PATH = Path("forecast_total_canarias_lstm.csv")
XGB_MODEL_PATH = Path("models/xgb_best.pkl")
LSTM_MODEL_PATH = Path("models/lstm_best.h5")
LSTM_SCALER_PATH = Path("models/scaler_y.pkl")

DATE_COL = "Fecha"
TARGET_COL = "Pasajeros"
PHASE_COL = "Phase"

HISTORY_COLUMNS = [
    "generated_at_utc",
    "forecast_origin",
    "target_date",
    "horizon_months",
    "model",
    "model_version",
    "prediction",
]


def _hash_files(paths: Iterable[Path]) -> str:
    """Return a short stable version hash for one or more model artifacts."""
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:16]


def _month_distance(origin: pd.Timestamp, target: pd.Timestamp) -> int:
    """Return the number of calendar months between origin and target."""
    return (target.year - origin.year) * 12 + target.month - origin.month


def _history_rows(
    forecast_path: Path,
    model_name: str,
    model_artifacts: list[Path],
    generated_at_utc: str,
) -> pd.DataFrame:
    """Convert one current forecast file into append-only history rows."""
    if not forecast_path.exists():
        raise FileNotFoundError(f"Missing forecast file: {forecast_path}")

    forecast = pd.read_csv(forecast_path, encoding="utf-8-sig")
    required = {DATE_COL, TARGET_COL, PHASE_COL}
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise KeyError(f"{forecast_path} is missing columns: {missing}")

    forecast[DATE_COL] = pd.to_datetime(forecast[DATE_COL], errors="coerce")
    forecast[TARGET_COL] = pd.to_numeric(forecast[TARGET_COL], errors="coerce")
    forecast = forecast.dropna(subset=[DATE_COL, TARGET_COL])

    history_dates = forecast.loc[forecast[PHASE_COL] == "History", DATE_COL]
    if history_dates.empty:
        raise ValueError(f"{forecast_path} contains no History rows")

    origin = history_dates.max()
    future = forecast[
        (forecast[PHASE_COL] == "Forecast") & (forecast[DATE_COL] > origin)
    ].copy()
    if future.empty:
        raise ValueError(f"{forecast_path} contains no future Forecast rows")

    model_version = _hash_files(model_artifacts)
    rows = pd.DataFrame(
        {
            "generated_at_utc": generated_at_utc,
            "forecast_origin": origin.strftime("%Y-%m-%d"),
            "target_date": future[DATE_COL].dt.strftime("%Y-%m-%d"),
            "horizon_months": [
                _month_distance(origin, target) for target in future[DATE_COL]
            ],
            "model": model_name,
            "model_version": model_version,
            "prediction": future[TARGET_COL].astype(float).to_numpy(),
        }
    )
    return rows[HISTORY_COLUMNS]


def append_forecast_history(history_path: Path = HISTORY_PATH) -> int:
    """Append the current XGBoost and LSTM forecasts without duplicating runs."""
    generated_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    new_rows = pd.concat(
        [
            _history_rows(
                XGB_FORECAST_PATH,
                "XGB",
                [XGB_MODEL_PATH],
                generated_at_utc,
            ),
            _history_rows(
                LSTM_FORECAST_PATH,
                "LSTM",
                [LSTM_MODEL_PATH, LSTM_SCALER_PATH],
                generated_at_utc,
            ),
        ],
        ignore_index=True,
    )

    if history_path.exists() and history_path.stat().st_size > 0:
        existing = pd.read_csv(history_path, encoding="utf-8-sig")
    else:
        existing = pd.DataFrame(columns=HISTORY_COLUMNS)

    combined = pd.concat([existing, new_rows], ignore_index=True)
    deduplication_key = [
        "forecast_origin",
        "target_date",
        "model",
        "model_version",
    ]
    before = len(combined)
    combined = combined.drop_duplicates(subset=deduplication_key, keep="first")
    added = before - len(existing) - (before - len(combined))

    combined = combined[HISTORY_COLUMNS].sort_values(
        ["forecast_origin", "model", "horizon_months"]
    )
    combined.to_csv(history_path, index=False, encoding="utf-8-sig")
    return max(added, 0)


def main() -> None:
    added = append_forecast_history()
    print(f"Forecast history updated: {added} new rows added to {HISTORY_PATH}.")


if __name__ == "__main__":
    main()
