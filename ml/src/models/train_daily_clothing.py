from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.daily_common import (
    fit_daily_pipeline,
    load_daily_target_frames,
    save_daily_model,
    save_daily_training_report,
)
from ml.src.utils.paths import (
    DAILY_CLOTHING_AFTERNOON_MODEL_PATH,
    DAILY_CLOTHING_EVENING_MODEL_PATH,
    DAILY_CLOTHING_MORNING_MODEL_PATH,
    DAILY_CLOTHING_TRAINING_REPORT_PATH,
)


@dataclass
class DailyClothingCandidate:
    name: str
    c: float
    class_weight: str | None


def _candidate_grid() -> list[DailyClothingCandidate]:
    return [
        DailyClothingCandidate(name="logreg_c0.5", c=0.5, class_weight=None),
        DailyClothingCandidate(name="logreg_c1.0", c=1.0, class_weight=None),
        DailyClothingCandidate(name="logreg_c2.0", c=2.0, class_weight=None),
        DailyClothingCandidate(name="logreg_c1.0_balanced", c=1.0, class_weight="balanced"),
    ]


def _train_single_clothing_target(
    target_column: str,
    model_path,
    split_config: SplitConfig | None,
    force_rebuild_split: bool,
) -> dict:
    frames = load_daily_target_frames(
        target_column=target_column,
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = frames.y_train.astype(str)
    y_val = frames.y_val.astype(str)

    best = {"score": -1.0, "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = LogisticRegression(
            C=candidate.c,
            class_weight=candidate.class_weight,
            max_iter=1600,
            random_state=42,
        )
        model = fit_daily_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
        val_pred = model.predict(frames.x_val)

        val_acc = accuracy_score(y_val, val_pred)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        trial = {
            "candidate": asdict(candidate),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_weighted_f1": float(val_weighted_f1),
        }
        trials.append(trial)

        if val_acc > best["score"]:
            best = {
                "score": float(val_acc),
                "candidate": candidate,
                "model": model,
                "val_macro_f1": float(val_macro_f1),
                "val_weighted_f1": float(val_weighted_f1),
                "rows": {
                    "train": frames.train_rows,
                    "val": frames.val_rows,
                    "test": frames.test_rows,
                },
            }

    save_daily_model(model_path, best["model"])
    return {
        "target": target_column,
        "model_path": str(model_path),
        "selection_metric": "accuracy",
        "best_candidate": asdict(best["candidate"]),
        "validation_metrics": {
            "accuracy": best["score"],
            "macro_f1": best["val_macro_f1"],
            "weighted_f1": best["val_weighted_f1"],
        },
        "rows": best["rows"],
        "candidate_trials": trials,
    }


def train_daily_clothing_models(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    morning = _train_single_clothing_target(
        target_column="clothing_morning",
        model_path=DAILY_CLOTHING_MORNING_MODEL_PATH,
        split_config=split_config,
        force_rebuild_split=force_rebuild_split,
    )
    afternoon = _train_single_clothing_target(
        target_column="clothing_afternoon",
        model_path=DAILY_CLOTHING_AFTERNOON_MODEL_PATH,
        split_config=split_config,
        force_rebuild_split=False,
    )
    evening = _train_single_clothing_target(
        target_column="clothing_evening",
        model_path=DAILY_CLOTHING_EVENING_MODEL_PATH,
        split_config=split_config,
        force_rebuild_split=False,
    )

    report = {
        "target_group": "daily_clothing",
        "split_config": asdict(split_config or SplitConfig()),
        "morning": morning,
        "afternoon": afternoon,
        "evening": evening,
    }
    save_daily_training_report(DAILY_CLOTHING_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    out = train_daily_clothing_models(force_rebuild_split=False)
    print("Daily clothing training complete.")
    print(
        {
            "morning": out["morning"]["validation_metrics"],
            "afternoon": out["afternoon"]["validation_metrics"],
            "evening": out["evening"]["validation_metrics"],
        }
    )

