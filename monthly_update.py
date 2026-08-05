"""Refresh passenger data and forecasts without retraining production models."""

from __future__ import annotations

import os
import subprocess
import sys

from download_agent import PassengerAgent, build_features
from update_forecast_history import append_snapshot

NO_NEW_DATA_MESSAGE = "No se encontró archivo para"


def run_command(script: str) -> None:
    """Run one inference script and fail immediately on errors."""
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    """Download new data, rebuild features, and refresh production forecasts."""
    os.environ.pop("RUN_RETRAIN", None)

    new_data_available = True
    agent = PassengerAgent()
    try:
        agent.run()
    except RuntimeError as exc:
        if NO_NEW_DATA_MESSAGE not in str(exc):
            raise
        new_data_available = False
        print("No new WebTenerife monthly file is available.")

    # Keep derived files synchronized even when no remote file is available.
    build_features()

    # Inference only: persisted production models are loaded without fitting.
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    if new_data_available:
        run_id, added_rows = append_snapshot()
        print(f"Forecast history run: {run_id}; new rows: {added_rows}.")
        print("Monthly source data and derived features were updated.")
    else:
        print("No history snapshot added because the source data did not change.")

    print("Production forecasts were refreshed without retraining.")


if __name__ == "__main__":
    main()
