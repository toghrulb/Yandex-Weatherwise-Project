from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import get_test_frames, load_model, write_markdown_report
from ml.src.utils.paths import UMBRELLA_MODEL_PATH, UMBRELLA_TEST_REPORT_PATH


def evaluate_umbrella_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(UMBRELLA_MODEL_PATH)
    x_test, y_test_raw = get_test_frames(
        target_column="umbrella_needed",
        split_config=split_config,
    )
    y_test = pd.to_numeric(y_test_raw, errors="coerce").fillna(0).astype(int)

    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
    }

    if write_report:
        write_markdown_report(
            path=report_path or UMBRELLA_TEST_REPORT_PATH,
            title="Umbrella Test Metrics",
            rows=[
                ("Accuracy", f"{metrics['accuracy']:.4f}"),
                ("F1", f"{metrics['f1']:.4f}"),
                ("Precision", f"{metrics['precision']:.4f}"),
                ("Recall", f"{metrics['recall']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_umbrella_test()
    print("Umbrella test evaluation complete.")
    print(json.dumps(result, indent=2))
