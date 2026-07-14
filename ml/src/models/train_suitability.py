from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.common import fit_pipeline, load_target_frames, save_model, save_training_report
from ml.src.models.suitability_postprocess import DEFAULT_SUITABILITY_LEVELS, quantize_suitability
from ml.src.utils.paths import SUITABILITY_MODEL_PATH, SUITABILITY_TRAINING_REPORT_PATH


@dataclass
class SuitabilityCandidate:
    name: str
    alpha: float


def _candidate_grid() -> list[SuitabilityCandidate]:
    return [
        SuitabilityCandidate(name="ridge_alpha_0.3", alpha=0.3),
        SuitabilityCandidate(name="ridge_alpha_1.0", alpha=1.0),
        SuitabilityCandidate(name="ridge_alpha_3.0", alpha=3.0),
        SuitabilityCandidate(name="ridge_alpha_10.0", alpha=10.0),
    ]


def train_suitability_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_target_frames(
        target_column="outdoor_suitability_score",
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = pd.to_numeric(frames.y_train, errors="coerce").fillna(0.0)
    y_val = pd.to_numeric(frames.y_val, errors="coerce").fillna(0.0)

    best = {"score": float("inf"), "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = Ridge(alpha=candidate.alpha)
        model = fit_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
        val_pred_raw = model.predict(frames.x_val)
        val_pred = quantize_suitability(val_pred_raw)

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

    save_model(SUITABILITY_MODEL_PATH, best["model"])

    report = {
        "target": "outdoor_suitability_score",
        "model_path": str(SUITABILITY_MODEL_PATH),
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
            "type": "nearest_fixed_levels",
            "levels": [float(x) for x in DEFAULT_SUITABILITY_LEVELS.tolist()],
        },
        "candidate_trials": trials,
    }
    save_training_report(SUITABILITY_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    output = train_suitability_model(force_rebuild_split=False)
    print("Suitability training complete.")
    print(output["validation_metrics"])
