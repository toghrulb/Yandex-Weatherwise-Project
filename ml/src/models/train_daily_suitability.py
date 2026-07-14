from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.daily_common import (
    fit_daily_pipeline,
    load_daily_target_frames,
    save_daily_model,
    save_daily_training_report,
)
from ml.src.utils.paths import DAILY_SUITABILITY_MODEL_PATH, DAILY_SUITABILITY_TRAINING_REPORT_PATH


@dataclass
class DailySuitabilityCandidate:
    name: str
    alpha: float


def _candidate_grid() -> list[DailySuitabilityCandidate]:
    return [
        DailySuitabilityCandidate(name="ridge_alpha_0.3", alpha=0.3),
        DailySuitabilityCandidate(name="ridge_alpha_1.0", alpha=1.0),
        DailySuitabilityCandidate(name="ridge_alpha_3.0", alpha=3.0),
        DailySuitabilityCandidate(name="ridge_alpha_10.0", alpha=10.0),
    ]


def train_daily_suitability_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_daily_target_frames(
        target_column="avg_outdoor_suitability",
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = pd.to_numeric(frames.y_train, errors="coerce").fillna(0.0)
    y_val = pd.to_numeric(frames.y_val, errors="coerce").fillna(0.0)

    best = {"score": float("inf"), "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = Ridge(alpha=candidate.alpha)
        model = fit_daily_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
        val_pred = np.clip(model.predict(frames.x_val), 0, 10)

        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        val_r2 = r2_score(y_val, val_pred)

        trial = {
            "candidate": asdict(candidate),
            "val_mae": float(val_mae),
            "val_rmse": val_rmse,
            "val_r2": float(val_r2),
        }
        trials.append(trial)

        if val_mae < best["score"]:
            best = {
                "score": float(val_mae),
                "candidate": candidate,
                "model": model,
                "val_rmse": val_rmse,
                "val_r2": float(val_r2),
            }

    save_daily_model(DAILY_SUITABILITY_MODEL_PATH, best["model"])
    report = {
        "target": "avg_outdoor_suitability",
        "model_path": str(DAILY_SUITABILITY_MODEL_PATH),
        "train_rows": frames.train_rows,
        "val_rows": frames.val_rows,
        "test_rows": frames.test_rows,
        "selection_metric": "mae",
        "split_config": asdict(split_config or SplitConfig()),
        "best_candidate": asdict(best["candidate"]),
        "validation_metrics": {
            "mae": best["score"],
            "rmse": best["val_rmse"],
            "r2": best["val_r2"],
        },
        "candidate_trials": trials,
    }
    save_daily_training_report(DAILY_SUITABILITY_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    out = train_daily_suitability_model(force_rebuild_split=False)
    print("Daily suitability training complete.")
    print(out["validation_metrics"])

