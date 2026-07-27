"""Generate the Total Canarias forecast with the persisted XGBoost model.

This file performs inference only. It reads the current monthly totals, creates
the same features as ``model_train_xgb.py``, loads ``models/xgb_best.pkl`` and
forecasts the next months iteratively without retraining the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ISLAND_NAME = "Total Canarias"