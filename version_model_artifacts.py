"""Create immutable version folders for trained model artifacts and results.

This script does not train or alter models. It copies the artifacts produced by
an existing training run into a versioned directory and updates a lightweight
registry plus per-model current pointers.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

VERSIONS_ROOT = Path("models/versions")
REGISTRY_PATH = Path("models/model_registry.csv")

MODEL_CONFIG = {
    "XGB": {
        "model_path": Path("models/xgb_best.pkl"),
        "metrics_path": Path("models/xgb_training_metrics.json"),
        "forecast_path": Path("forecast_total_canarias_xgb.csv"),
        "extra_artifacts": {},
    },
    "LSTM": {
        "model_path": Path("models/lstm_best.h5"),
        "metrics_path": Path("models/lstm_training_metrics.json"),
        "forecast_path": Path("forecast_total_canarias_lstm.csv"),
        "extra_artifacts": {
            "scaler_y.pkl": Path("models/scaler_y.pkl"),
        },
    },
}

REGISTRY_COLUMNS = [
    "model_version",
    "model",
    "created_at_utc",
    "data_through",
    "training_start",
    "training_end",
    "holdout_start",
    "holdout_end",
    "holdout_months",
    "mae",
    "rmse",
    "mape_pct",
    "status",
    "git_commit",
    "version_path",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _source_commit() -> str:
    return os.getenv("GITHUB_SHA", "unknown").strip() or "unknown"


def _run_token(now: datetime) -> str:
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    if run_id:
        return run_id
    return now.strftime("%Y%m%dT%H%M%SZ")


def _load_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _data_through(metrics: dict[str, object]) -> str:
    value = metrics.get("training_end", metrics.get("history_end"))
    if not value:
        raise KeyError("Metrics do not contain training_end or history_end")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _training_start(metrics: dict[str, object]) -> str:
    value = metrics.get("training_start", metrics.get("history_start", ""))
    return pd.Timestamp(value).strftime("%Y-%m-%d") if value else ""


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_xgb_feature_importance(model_path: Path, output_path: Path) -> None:
    """Write feature importances when the persisted XGB pipeline exposes them."""
    pipeline = joblib.load(model_path)
    estimator = getattr(pipeline, "named_steps", {}).get("xgb")
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return

    feature_names = [
        "month_sin",
        "month_cos",
        "year_norm",
        *[f"lag_{index}" for index in range(1, 13)],
        "roll3",
        "roll6",
    ]
    if len(importances) != len(feature_names):
        feature_names = [f"feature_{index}" for index in range(len(importances))]

    importance = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_path, index=False, encoding="utf-8-sig")


def _registry_row(
    model: str,
    version: str,
    created_at: str,
    version_dir: Path,
    metrics: dict[str, object],
    source_commit: str,
) -> dict[str, object]:
    return {
        "model_version": version,
        "model": model,
        "created_at_utc": created_at,
        "data_through": _data_through(metrics),
        "training_start": _training_start(metrics),
        "training_end": _data_through(metrics),
        "holdout_start": metrics.get("holdout_start", ""),
        "holdout_end": metrics.get("holdout_end", ""),
        "holdout_months": metrics.get("holdout_months", ""),
        "mae": metrics.get("mae", ""),
        "rmse": metrics.get("rmse", ""),
        "mape_pct": metrics.get("mape_pct", ""),
        "status": "active",
        "git_commit": source_commit,
        "version_path": version_dir.as_posix(),
    }


def _update_registry(new_rows: list[dict[str, object]]) -> None:
    if REGISTRY_PATH.exists():
        registry = pd.read_csv(REGISTRY_PATH, encoding="utf-8-sig")
    else:
        registry = pd.DataFrame(columns=REGISTRY_COLUMNS)

    new_df = pd.DataFrame(new_rows, columns=REGISTRY_COLUMNS)
    registry = pd.concat([registry, new_df], ignore_index=True)
    registry = registry.drop_duplicates(
        subset=["model_version", "model"], keep="last"
    )

    # Only the newest registered version of each model is active.
    registry["status"] = "archived"
    for model in registry["model"].dropna().unique():
        model_rows = registry[registry["model"] == model]
        newest_index = model_rows["created_at_utc"].astype(str).idxmax()
        registry.loc[newest_index, "status"] = "active"

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry[REGISTRY_COLUMNS].sort_values(
        ["created_at_utc", "model"]
    ).to_csv(REGISTRY_PATH, index=False, encoding="utf-8-sig")


def version_model(model: str, config: dict[str, object], now: datetime) -> dict[str, object]:
    metrics_path = Path(config["metrics_path"])
    metrics = _load_metrics(metrics_path)
    data_through = _data_through(metrics)
    source_commit = _source_commit()
    version = f"{data_through[:7].replace('-', '')}_{_run_token(now)}"
    version_dir = VERSIONS_ROOT / model.lower() / version

    if version_dir.exists():
        raise FileExistsError(f"Version directory already exists: {version_dir}")
    version_dir.mkdir(parents=True)

    model_source = Path(config["model_path"])
    model_filename = "model.pkl" if model == "XGB" else "model.h5"
    _copy_required(model_source, version_dir / model_filename)
    _copy_required(metrics_path, version_dir / "metrics.json")
    _copy_required(Path(config["forecast_path"]), version_dir / "forecast.csv")

    for destination_name, source_path in dict(config["extra_artifacts"]).items():
        _copy_required(Path(source_path), version_dir / destination_name)

    if model == "XGB":
        _write_xgb_feature_importance(
            model_source, version_dir / "feature_importance.csv"
        )

    created_at = now.isoformat()
    metadata = {
        "model_version": version,
        "model": model,
        "created_at_utc": created_at,
        "data_through": data_through,
        "training_start": _training_start(metrics),
        "training_end": data_through,
        "holdout_start": metrics.get("holdout_start"),
        "holdout_end": metrics.get("holdout_end"),
        "holdout_months": metrics.get("holdout_months"),
        "metrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape_pct": metrics.get("mape_pct"),
        },
        "git_commit": source_commit,
        "status": "active",
        "artifacts": sorted(path.name for path in version_dir.iterdir()),
    }
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pointer_path = VERSIONS_ROOT / model.lower() / "current.json"
    pointer_path.write_text(
        json.dumps(
            {
                "model": model,
                "model_version": version,
                "version_path": version_dir.as_posix(),
                "updated_at_utc": created_at,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return _registry_row(
        model=model,
        version=version,
        created_at=created_at,
        version_dir=version_dir,
        metrics=metrics,
        source_commit=source_commit,
    )


def main() -> None:
    now = _utc_now()
    rows = [
        version_model(model, config, now)
        for model, config in MODEL_CONFIG.items()
    ]
    _update_registry(rows)
    for row in rows:
        print(
            f"✅ {row['model']} versioned as {row['model_version']} "
            f"in {row['version_path']}"
        )
    print(f"📋 Registry updated: {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
