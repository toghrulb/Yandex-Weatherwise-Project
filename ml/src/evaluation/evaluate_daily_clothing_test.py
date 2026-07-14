from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import load_model, write_markdown_report
from ml.src.models.daily_common import load_daily_target_frames
from ml.src.utils.paths import (
    DAILY_CLOTHING_AFTERNOON_MODEL_PATH,
    DAILY_CLOTHING_EVENING_MODEL_PATH,
    DAILY_CLOTHING_MORNING_MODEL_PATH,
    DAILY_CLOTHING_TEST_REPORT_PATH,
)


def _evaluate_single(model_path, target_column: str, split_config: SplitConfig | None) -> dict:
    model = load_model(model_path)
    frames = load_daily_target_frames(
        target_column=target_column,
        force_rebuild_split=False,
        split_config=split_config,
    )
    y_test = frames.y_test.astype(str)
    pred = model.predict(frames.x_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }


def evaluate_daily_clothing_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    morning = _evaluate_single(DAILY_CLOTHING_MORNING_MODEL_PATH, "clothing_morning", split_config)
    afternoon = _evaluate_single(DAILY_CLOTHING_AFTERNOON_MODEL_PATH, "clothing_afternoon", split_config)
    evening = _evaluate_single(DAILY_CLOTHING_EVENING_MODEL_PATH, "clothing_evening", split_config)

    result = {
        "morning": morning,
        "afternoon": afternoon,
        "evening": evening,
    }

    if write_report:
        write_markdown_report(
            path=report_path or DAILY_CLOTHING_TEST_REPORT_PATH,
            title="Daily Clothing Test Metrics",
            rows=[
                ("Morning Accuracy", f"{morning['accuracy']:.4f}"),
                ("Morning Macro F1", f"{morning['macro_f1']:.4f}"),
                ("Afternoon Accuracy", f"{afternoon['accuracy']:.4f}"),
                ("Afternoon Macro F1", f"{afternoon['macro_f1']:.4f}"),
                ("Evening Accuracy", f"{evening['accuracy']:.4f}"),
                ("Evening Macro F1", f"{evening['macro_f1']:.4f}"),
            ],
        )
    return result


if __name__ == "__main__":
    result = evaluate_daily_clothing_test()
    print("Daily clothing test evaluation complete.")
    print(json.dumps(result, indent=2))

