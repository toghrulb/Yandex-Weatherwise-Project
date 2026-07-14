from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.daily_common import (
    fit_daily_pipeline,
    load_daily_target_frames,
    save_daily_model,
    save_daily_training_report,
)
from ml.src.utils.paths import DAILY_BEST_HOUR_MODEL_PATH, DAILY_BEST_HOUR_TRAINING_REPORT_PATH


@dataclass
class DailyBestHourCandidate:
    name: str
    c: float


def _candidate_grid() -> list[DailyBestHourCandidate]:
    return [
        DailyBestHourCandidate(name="logreg_c0.5", c=0.5),
        DailyBestHourCandidate(name="logreg_c1.0", c=1.0),
        DailyBestHourCandidate(name="logreg_c2.0", c=2.0),
    ]


def _within_1h_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def train_daily_best_hour_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_daily_target_frames(
        target_column="best_outdoor_hour",
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = pd.to_numeric(frames.y_train, errors="coerce").fillna(12).astype(int)
    y_val = pd.to_numeric(frames.y_val, errors="coerce").fillna(12).astype(int)

    best = {"score": -1.0, "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = LogisticRegression(
            C=candidate.c,
            max_iter=1600,
            random_state=42,
        )
        model = fit_daily_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
        val_pred = model.predict(frames.x_val).astype(int)

        val_acc = accuracy_score(y_val, val_pred)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_within_1h = _within_1h_accuracy(y_true=y_val.to_numpy(), y_pred=val_pred)

        trial = {
            "candidate": asdict(candidate),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_within_1h_accuracy": float(val_within_1h),
        }
        trials.append(trial)

        if val_acc > best["score"]:
            best = {
                "score": float(val_acc),
                "candidate": candidate,
                "model": model,
                "val_macro_f1": float(val_macro_f1),
                "val_within_1h_accuracy": float(val_within_1h),
            }

    save_daily_model(DAILY_BEST_HOUR_MODEL_PATH, best["model"])
    report = {
        "target": "best_outdoor_hour",
        "model_path": str(DAILY_BEST_HOUR_MODEL_PATH),
        "train_rows": frames.train_rows,
        "val_rows": frames.val_rows,
        "test_rows": frames.test_rows,
        "selection_metric": "accuracy",
        "split_config": asdict(split_config or SplitConfig()),
        "best_candidate": asdict(best["candidate"]),
        "validation_metrics": {
            "accuracy": best["score"],
            "macro_f1": best["val_macro_f1"],
            "within_1h_accuracy": best["val_within_1h_accuracy"],
        },
        "candidate_trials": trials,
    }
    save_daily_training_report(DAILY_BEST_HOUR_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    out = train_daily_best_hour_model(force_rebuild_split=False)
    print("Daily best-hour training complete.")
    print(out["validation_metrics"])

