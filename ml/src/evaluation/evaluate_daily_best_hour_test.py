from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import load_model, write_markdown_report
from ml.src.models.daily_common import load_daily_target_frames
from ml.src.utils.paths import DAILY_BEST_HOUR_MODEL_PATH, DAILY_BEST_HOUR_TEST_REPORT_PATH


def _within_1h_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= 1))


def evaluate_daily_best_hour_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(DAILY_BEST_HOUR_MODEL_PATH)
    frames = load_daily_target_frames(
        target_column="best_outdoor_hour",
        force_rebuild_split=False,
        split_config=split_config,
    )
    y_test = pd.to_numeric(frames.y_test, errors="coerce").fillna(12).astype(int)
    pred = model.predict(frames.x_test).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "within_1h_accuracy": _within_1h_accuracy(y_true=y_test.to_numpy(), y_pred=pred),
    }

    if write_report:
        write_markdown_report(
            path=report_path or DAILY_BEST_HOUR_TEST_REPORT_PATH,
            title="Daily Best Outdoor Hour Test Metrics",
            rows=[
                ("Accuracy", f"{metrics['accuracy']:.4f}"),
                ("Macro F1", f"{metrics['macro_f1']:.4f}"),
                ("Within +/-1h Accuracy", f"{metrics['within_1h_accuracy']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_daily_best_hour_test()
    print("Daily best-hour test evaluation complete.")
    print(json.dumps(result, indent=2))

