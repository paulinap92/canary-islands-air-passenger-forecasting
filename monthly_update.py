"""Refresh production forecasts and preserve their history without retraining.

When WebTenerife publishes a new monthly XLSX file, this workflow updates the
historical datasets and features first. Even when no new source file exists, it
can still regenerate and archive the current production forecast snapshot.
"""

from __future__ import annotations

import os
import subprocess
import sys

from download_agent import PassengerAgent, build_features
from forecast_history import append_forecast_history


NO_NEW_DATA_MESSAGE = "No se encontró archivo para"


def run_command(script: str) -> None:
    """Run one inference script and fail immediately on errors."""
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    # The legacy agent still supports RUN_RETRAIN for backward compatibility.
    # This production workflow explicitly disables it.
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

    if new_data_available:
        # Feature engineering must use the newly downloaded historical month.
        build_features()

    # Inference only: neither script is allowed to call model.fit().
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    # The first run starts the real forecast history. Repeated runs with the
    # same origin and model artifacts are deduplicated.
    added_rows = append_forecast_history()

    if new_data_available:
        print("Monthly data and features were updated.")
    else:
        print("Historical datasets were unchanged.")

    print("Production forecasts were refreshed.")
    print(f"Forecast history updated with {added_rows} new rows.")
    print("No model retraining was performed.")


if __name__ == "__main__":
    main()
