from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import get_test_frames, load_model, write_markdown_report
from ml.src.models.suitability_postprocess import quantize_suitability
from ml.src.utils.paths import SUITABILITY_MODEL_PATH, SUITABILITY_TEST_REPORT_PATH


def evaluate_suitability_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(SUITABILITY_MODEL_PATH)
    x_test, y_test_raw = get_test_frames(
        target_column="outdoor_suitability_score",
        split_config=split_config,
    )
    y_test = pd.to_numeric(y_test_raw, errors="coerce").fillna(0.0)

    pred_raw = model.predict(x_test)
    pred = quantize_suitability(pred_raw)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }

    if write_report:
        write_markdown_report(
            path=report_path or SUITABILITY_TEST_REPORT_PATH,
            title="Suitability Test Metrics",
            rows=[
                ("MAE", f"{metrics['mae']:.4f}"),
                ("RMSE", f"{metrics['rmse']:.4f}"),
                ("R2", f"{metrics['r2']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_suitability_test()
    print("Suitability test evaluation complete.")
    print(json.dumps(result, indent=2))
