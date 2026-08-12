"""Refresh passenger data and forecasts only when new source data is available."""

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
    """Check for new data; update everything only when a new XLSX is found."""
    os.environ.pop("RUN_RETRAIN", None)

    agent = PassengerAgent()
    try:
        agent.run()
    except RuntimeError as exc:
        if NO_NEW_DATA_MESSAGE not in str(exc):
            raise
        print("No new WebTenerife monthly file is available. Nothing else to do.")
        return

    # A new source file was downloaded and result CSVs were updated.
    build_features()

    # Inference only: persisted production models are loaded without fitting.
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    run_id, added_rows = append_snapshot()
    print(f"Forecast history run: {run_id}; new rows: {added_rows}.")
    print("Monthly source data, derived features, and forecasts were updated.")
    print("No model retraining was performed.")


if __name__ == "__main__":
    main()
