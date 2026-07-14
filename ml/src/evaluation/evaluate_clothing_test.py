from __future__ import annotations

import json

from sklearn.metrics import accuracy_score, f1_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import get_test_frames, load_model, write_markdown_report
from ml.src.utils.paths import CLOTHING_MODEL_PATH, CLOTHING_TEST_REPORT_PATH


def evaluate_clothing_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(CLOTHING_MODEL_PATH)
    x_test, y_test = get_test_frames(
        target_column="clothing_recommendation",
        split_config=split_config,
    )
    y_test = y_test.astype(str)

    pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

    if write_report:
        write_markdown_report(
            path=report_path or CLOTHING_TEST_REPORT_PATH,
            title="Clothing Test Metrics",
            rows=[
                ("Accuracy", f"{metrics['accuracy']:.4f}"),
                ("Macro F1", f"{metrics['macro_f1']:.4f}"),
                ("Weighted F1", f"{metrics['weighted_f1']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_clothing_test()
    print("Clothing test evaluation complete.")
    print(json.dumps(result, indent=2))
