"""Append the current XGB and LSTM forecasts to an immutable history file.

Each model run is stored as a snapshot in ``forecast_history.csv``. Re-running
with identical forecasts does not create duplicate rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATE_COL = "Fecha"
TARGET_COL = "Pasajeros"
HISTORY_PATH = Path(os.getenv("FORECAST_HISTORY_PATH", "forecast_history.csv"))
FORECAST_PATHS = {
    "XGB": Path(os.getenv("XGB_FORECAST_OUTPUT_PATH", "forecast_total_canarias_xgb.csv")),
    "LSTM": Path(os.getenv("LSTM_FORECAST_OUTPUT_PATH", "forecast_total_canarias_lstm.csv")),
}
HISTORY_COLUMNS = [
    "run_id",
    "generated_at_utc",
    "data_through",
    "model",
    "target_month",
    "horizon_months",
    "predicted_passengers",
    "source_commit",
]


def resolve_source_commit() -> str:
    """Return the workflow SHA or the current Git commit when available."""
    workflow_sha = os.getenv("GITHUB_SHA", "").strip()
    if workflow_sha:
        return workflow_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve_generated_at() -> str:
    """Return a stable ISO timestamp for this snapshot."""
    configured = os.getenv("FORECAST_GENERATED_AT_UTC", "").strip()
    if configured:
        timestamp = pd.Timestamp(configured)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat()

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_model_forecast(model: str, path: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Load and validate one model forecast file."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el forecast {model}: {path}")

    forecast = pd.read_csv(path, encoding="utf-8-sig")
    required = {DATE_COL, TARGET_COL, "Phase"}
    missing = sorted(required.difference(forecast.columns))
    if missing:
        raise KeyError(f"Faltan columnas en {path}: {missing}")

    forecast[DATE_COL] = pd.to_datetime(forecast[DATE_COL], errors="coerce")
    forecast[TARGET_COL] = pd.to_numeric(forecast[TARGET_COL], errors="coerce")
    forecast = forecast.dropna(subset=[DATE_COL, TARGET_COL]).copy()

    phase = forecast["Phase"].astype(str).str.casefold()
    historical = forecast[phase == "history"].sort_values(DATE_COL)
    future = forecast[phase == "forecast"].sort_values(DATE_COL).reset_index(drop=True)

    if historical.empty:
        raise ValueError(f"El forecast {model} no contiene filas History.")
    if future.empty:
        raise ValueError(f"El forecast {model} no contiene filas Forecast.")

    data_through = historical[DATE_COL].max().to_period("M").to_timestamp()
    expected_dates = pd.Series(
        pd.date_range(
            start=data_through + pd.offsets.MonthBegin(1),
            periods=len(future),
            freq="MS",
        )
    )
    actual_dates = future[DATE_COL].dt.to_period("M").dt.to_timestamp()
    if not actual_dates.reset_index(drop=True).equals(expected_dates):
        raise ValueError(f"Las fechas futuras de {model} no son mensuales y consecutivas.")
    if (future[TARGET_COL] < 0).any():
        raise ValueError(f"El forecast {model} contiene valores negativos.")

    future[DATE_COL] = actual_dates
    return data_through, future


def build_snapshot() -> pd.DataFrame:
    """Build one combined XGB/LSTM snapshot."""
    source_commit = resolve_source_commit()
    generated_at = resolve_generated_at()
    loaded: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {
        model: load_model_forecast(model, path)
        for model, path in FORECAST_PATHS.items()
    }

    data_through_values = {value[0] for value in loaded.values()}
    if len(data_through_values) != 1:
        raise ValueError("XGB y LSTM no utilizan el mismo último mes histórico.")
    data_through = data_through_values.pop()

    payload = {
        "source_commit": source_commit,
        "data_through": data_through.strftime("%Y-%m-%d"),
        "forecasts": {
            model: [
                [row[DATE_COL].strftime("%Y-%m-%d"), round(float(row[TARGET_COL]), 8)]
                for _, row in future.iterrows()
            ]
            for model, (_, future) in loaded.items()
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    run_id = f"{data_through.strftime('%Y%m')}-{digest}"

    records: list[dict[str, object]] = []
    for model, (_, future) in loaded.items():
        for horizon, (_, row) in enumerate(future.iterrows(), start=1):
            records.append(
                {
                    "run_id": run_id,
                    "generated_at_utc": generated_at,
                    "data_through": data_through.strftime("%Y-%m-%d"),
                    "model": model,
                    "target_month": row[DATE_COL].strftime("%Y-%m-%d"),
                    "horizon_months": horizon,
                    "predicted_passengers": float(row[TARGET_COL]),
                    "source_commit": source_commit,
                }
            )

    return pd.DataFrame(records, columns=HISTORY_COLUMNS)


def load_existing_history() -> pd.DataFrame:
    """Load the existing history, or return an empty table."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig")
    missing = sorted(set(HISTORY_COLUMNS).difference(history.columns))
    if missing:
        raise KeyError(f"Faltan columnas en {HISTORY_PATH}: {missing}")
    return history[HISTORY_COLUMNS].copy()


def append_snapshot() -> tuple[str, int]:
    """Append the current snapshot unless the same run already exists."""
    new_snapshot = build_snapshot()
    run_id = str(new_snapshot["run_id"].iloc[0])
    existing = load_existing_history()

    if not existing.empty and run_id in set(existing["run_id"].astype(str)):
        print(f"ℹ️ Snapshot {run_id} ya existe; no se añaden duplicados.")
        return run_id, 0

    combined = pd.concat([existing, new_snapshot], ignore_index=True)
    combined["generated_at_utc"] = pd.to_datetime(
        combined["generated_at_utc"], utc=True, errors="coerce"
    )
    combined["data_through"] = pd.to_datetime(combined["data_through"], errors="coerce")
    combined["target_month"] = pd.to_datetime(combined["target_month"], errors="coerce")
    if combined[["generated_at_utc", "data_through", "target_month"]].isna().any().any():
        raise ValueError("La historia contiene fechas inválidas.")

    combined = combined.sort_values(
        ["generated_at_utc", "model", "target_month"],
        kind="stable",
    )
    combined["generated_at_utc"] = combined["generated_at_utc"].map(
        lambda value: value.isoformat()
    )
    combined["data_through"] = combined["data_through"].dt.strftime("%Y-%m-%d")
    combined["target_month"] = combined["target_month"].dt.strftime("%Y-%m-%d")

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = HISTORY_PATH.with_suffix(HISTORY_PATH.suffix + ".tmp")
    combined.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(HISTORY_PATH)

    print(f"✅ Añadido snapshot {run_id}: {len(new_snapshot)} filas.")
    print(f"💾 Historia guardada en {HISTORY_PATH}")
    return run_id, len(new_snapshot)


def main() -> None:
    append_snapshot()


if __name__ == "__main__":
    main()
