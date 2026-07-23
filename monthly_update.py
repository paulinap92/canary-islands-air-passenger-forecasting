"""Monthly data and forecast update without model retraining.

Run this workflow after WebTenerife publishes a new monthly XLSX file. It updates
historical datasets, rebuilds features, refreshes forecasts with persisted
production models, and appends the run to the immutable forecast history.
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
    # This production monthly workflow explicitly disables it.
    os.environ.pop("RUN_RETRAIN", None)

    agent = PassengerAgent()
    try:
        agent.run()
    except RuntimeError as exc:
        if NO_NEW_DATA_MESSAGE in str(exc):
            print("No new WebTenerife monthly file is available. Nothing changed.")
            return
        raise

    # Feature engineering must use the newly downloaded historical month.
    build_features()

    # Inference only: neither script is allowed to call model.fit().
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    added_rows = append_forecast_history()

    print("Monthly update completed: data, features, and forecasts refreshed.")
    print(f"Forecast history updated with {added_rows} new rows.")
    print("No model retraining was performed.")


if __name__ == "__main__":
    main()
