# Model development notebooks

## `my_models_trials.ipynb`

This notebook documents the **initial model development, comparison, selection, and first training run** for the project.

It includes:

- feature and dataset preparation for forecasting,
- comparison of classical regression models,
- time-series validation and holdout evaluation,
- experiments with XGBoost, LSTM, GRU, and Transformer architectures,
- selection of the production XGBoost and LSTM approaches,
- creation of the initial persisted artifacts in `models/`.

The notebook is **not part of the monthly production update** and should not be executed automatically whenever a new month of data arrives.

## Production lifecycle

- `monthly_update.py` updates data, rebuilds features, and generates forecasts with existing production models.
- `training/retrain_models.py` is the explicit retraining entry point for new model candidates.
- A candidate model should replace the production model only after its validation metrics have been reviewed.
