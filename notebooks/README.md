# Model development notebooks

The main model-development record is currently stored at the repository root as `my_models_trials.ipynb`.

It documents the initial:

- feature and dataset preparation,
- classical-model comparison,
- time-series validation and holdout evaluation,
- XGBoost, LSTM, GRU, and Transformer experiments,
- selection of the production XGBoost and LSTM approaches,
- creation of the first persisted model artifacts.

The notebook is not part of the monthly production update and is not executed automatically when a new month arrives.

## Production lifecycle

- `monthly_update.py` updates data and generates forecasts with existing production artifacts.
- `training/retrain_models.py` compares full-history and post-2021 XGBoost candidates.
- `training/train_lstm.py` compares the same data scopes with the selected LSTM architecture.
- The better holdout scope is retrained on all data in that scope and saved only as a candidate.
- Candidate artifacts remain separate until their validation reports and forecast behaviour are reviewed.

Routine retraining does not repeat the full architecture search. A new XGBoost/LSTM/GRU/Transformer comparison should be treated as a deliberate model-development experiment.
