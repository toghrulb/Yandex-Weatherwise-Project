from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.activity_common import (
    fit_activity_pipeline,
    load_activity_target_frames,
    save_activity_model,
    save_activity_training_report,
)
from ml.src.models.activity_postprocess import quantize_activity_suitability
from ml.src.utils.paths import ACTIVITY_SUITABILITY_MODEL_PATH, ACTIVITY_SUITABILITY_TRAINING_REPORT_PATH


@dataclass
class ActivitySuitabilityCandidate:
    name: str
    alpha: float


def _candidate_grid() -> list[ActivitySuitabilityCandidate]:
    return [
        ActivitySuitabilityCandidate(name="ridge_alpha_0.3", alpha=0.3),
        ActivitySuitabilityCandidate(name="ridge_alpha_1.0", alpha=1.0),
        ActivitySuitabilityCandidate(name="ridge_alpha_3.0", alpha=3.0),
        ActivitySuitabilityCandidate(name="ridge_alpha_10.0", alpha=10.0),
    ]


def train_activity_suitability_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_activity_target_frames(
        target_column="activity_suitability_score",
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = pd.to_numeric(frames.y_train, errors="coerce").fillna(0.0)
    y_val = pd.to_numeric(frames.y_val, errors="coerce").fillna(0.0)

    best = {"score": float("inf"), "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = Ridge(alpha=candidate.alpha)
        model = fit_activity_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)

        val_pred_raw = model.predict(frames.x_val)
        val_pred = quantize_activity_suitability(val_pred_raw)

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

    save_activity_model(ACTIVITY_SUITABILITY_MODEL_PATH, best["model"])
    report = {
        "target": "activity_suitability_score",
        "model_path": str(ACTIVITY_SUITABILITY_MODEL_PATH),
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
        "postprocessing": {
            "type": "nearest_integer",
            "min_score": 0,
            "max_score": 10,
        },
        "candidate_trials": trials,
    }
    save_activity_training_report(ACTIVITY_SUITABILITY_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    out = train_activity_suitability_model(force_rebuild_split=False)
    print("Activity suitability training complete.")
    print(out["validation_metrics"])

