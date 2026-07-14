from __future__ import annotations

from dataclasses import asdict, dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.common import fit_pipeline, load_target_frames, save_model, save_training_report
from ml.src.utils.paths import CLOTHING_MODEL_PATH, CLOTHING_TRAINING_REPORT_PATH


@dataclass
class ClothingCandidate:
    name: str
    c: float
    class_weight: str | None


def _candidate_grid() -> list[ClothingCandidate]:
    return [
        ClothingCandidate(name="logreg_c0.5", c=0.5, class_weight=None),
        ClothingCandidate(name="logreg_c1.0", c=1.0, class_weight=None),
        ClothingCandidate(name="logreg_c2.0", c=2.0, class_weight=None),
        ClothingCandidate(name="logreg_c1.0_balanced", c=1.0, class_weight="balanced"),
    ]


def train_clothing_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_target_frames(
        target_column="clothing_recommendation",
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
            max_iter=1800,
            n_jobs=None,
            random_state=42,
        )
        model = fit_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
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
            }

    save_model(CLOTHING_MODEL_PATH, best["model"])

    report = {
        "target": "clothing_recommendation",
        "model_path": str(CLOTHING_MODEL_PATH),
        "train_rows": frames.train_rows,
        "val_rows": frames.val_rows,
        "test_rows": frames.test_rows,
        "selection_metric": "accuracy",
        "split_config": asdict(split_config or SplitConfig()),
        "best_candidate": asdict(best["candidate"]),
        "validation_metrics": {
            "accuracy": best["score"],
            "macro_f1": best["val_macro_f1"],
            "weighted_f1": best["val_weighted_f1"],
        },
        "candidate_trials": trials,
    }
    save_training_report(CLOTHING_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    output = train_clothing_model(force_rebuild_split=False)
    print("Clothing training complete.")
    print(output["validation_metrics"])
