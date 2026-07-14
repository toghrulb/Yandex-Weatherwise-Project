from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from ml.src.data.split_manager import SplitConfig
from ml.src.models.common import fit_pipeline, load_target_frames, save_model, save_training_report
from ml.src.utils.paths import UMBRELLA_MODEL_PATH, UMBRELLA_TRAINING_REPORT_PATH


@dataclass
class UmbrellaCandidate:
    name: str
    c: float
    class_weight: str | None


def _candidate_grid() -> list[UmbrellaCandidate]:
    return [
        UmbrellaCandidate(name="logreg_c0.3_balanced", c=0.3, class_weight="balanced"),
        UmbrellaCandidate(name="logreg_c1.0_balanced", c=1.0, class_weight="balanced"),
        UmbrellaCandidate(name="logreg_c3.0_balanced", c=3.0, class_weight="balanced"),
        UmbrellaCandidate(name="logreg_c1.0_unweighted", c=1.0, class_weight=None),
    ]


def train_umbrella_model(
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> dict:
    frames = load_target_frames(
        target_column="umbrella_needed",
        force_rebuild_split=force_rebuild_split,
        split_config=split_config,
    )
    y_train = pd.to_numeric(frames.y_train, errors="coerce").fillna(0).astype(int)
    y_val = pd.to_numeric(frames.y_val, errors="coerce").fillna(0).astype(int)

    best = {"score": -1.0, "candidate": None, "model": None}
    trials = []

    for candidate in _candidate_grid():
        estimator = LogisticRegression(
            C=candidate.c,
            class_weight=candidate.class_weight,
            max_iter=1200,
            random_state=42,
        )
        model = fit_pipeline(estimator=estimator, x_train=frames.x_train, y_train=y_train)
        val_pred = model.predict(frames.x_val)

        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        val_precision = precision_score(y_val, val_pred, zero_division=0)
        val_recall = recall_score(y_val, val_pred, zero_division=0)

        trial = {
            "candidate": asdict(candidate),
            "val_f1": float(val_f1),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
        }
        trials.append(trial)

        if val_f1 > best["score"]:
            best = {
                "score": float(val_f1),
                "candidate": candidate,
                "model": model,
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
            }

    save_model(UMBRELLA_MODEL_PATH, best["model"])

    report = {
        "target": "umbrella_needed",
        "model_path": str(UMBRELLA_MODEL_PATH),
        "train_rows": frames.train_rows,
        "val_rows": frames.val_rows,
        "test_rows": frames.test_rows,
        "selection_metric": "f1",
        "split_config": asdict(split_config or SplitConfig()),
        "best_candidate": asdict(best["candidate"]),
        "validation_metrics": {
            "f1": best["score"],
            "precision": best["val_precision"],
            "recall": best["val_recall"],
        },
        "candidate_trials": trials,
    }
    save_training_report(UMBRELLA_TRAINING_REPORT_PATH, report)
    return report


if __name__ == "__main__":
    output = train_umbrella_model(force_rebuild_split=False)
    print("Umbrella training complete.")
    print(output["validation_metrics"])
