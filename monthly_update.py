"""Refresh production forecasts and preserve their history without retraining.

When WebTenerife publishes a new monthly XLSX file, this workflow updates the
historical datasets first. Lag and rolling features are rebuilt on every run so
XGBoost always uses the same latest real month as the LSTM and dashboard data.
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
    """Refresh data-derived features, forecasts, and immutable history."""
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

    # Always rebuild derived features. This also repairs stale lag files when
    # result_total.csv is newer even though no new remote XLSX is available.
    build_features()

    # Inference only: neither script is allowed to call model.fit().
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    added_rows = append_forecast_history()

    if new_data_available:
        print("Monthly source data and derived features were updated.")
    else:
        print("Source data was unchanged; derived features were synchronized.")

    print("Production forecasts were refreshed.")
    print(f"Forecast history updated with {added_rows} new rows.")
    print("No model retraining was performed.")


if __name__ == "__main__":
    main()
