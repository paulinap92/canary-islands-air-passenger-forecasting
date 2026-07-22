"""Monthly data and forecast update without model retraining.

Run this workflow after a new WebTenerife XLSX file is available. It updates the
historical datasets, rebuilds lag features, and generates fresh forecasts using
the persisted production models.
"""

from __future__ import annotations

import subprocess
import sys

from download_agent import PassengerAgent, build_features


def run_command(script: str) -> None:
    """Run one inference script and fail immediately on errors."""
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    agent = PassengerAgent()
    agent.run()

    # Feature engineering must use the newly downloaded historical month.
    build_features()

    # Inference only: neither script is allowed to call model.fit().
    run_command("model_final_xgb.py")
    run_command("model_final_lstm.py")

    print("Monthly update completed: data, features, and forecasts refreshed.")
    print("No model retraining was performed.")


if __name__ == "__main__":
    main()
