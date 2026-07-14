from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import load_model, write_markdown_report
from ml.src.models.activity_common import load_activity_target_frames
from ml.src.models.activity_postprocess import quantize_activity_suitability
from ml.src.utils.paths import ACTIVITY_SUITABILITY_MODEL_PATH, ACTIVITY_SUITABILITY_TEST_REPORT_PATH


def evaluate_activity_suitability_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(ACTIVITY_SUITABILITY_MODEL_PATH)
    frames = load_activity_target_frames(
        target_column="activity_suitability_score",
        force_rebuild_split=False,
        split_config=split_config,
    )
    y_test = pd.to_numeric(frames.y_test, errors="coerce").fillna(0.0)
    pred_raw = model.predict(frames.x_test)
    pred = quantize_activity_suitability(pred_raw)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }

    if write_report:
        write_markdown_report(
            path=report_path or ACTIVITY_SUITABILITY_TEST_REPORT_PATH,
            title="Activity Suitability Test Metrics",
            rows=[
                ("MAE", f"{metrics['mae']:.4f}"),
                ("RMSE", f"{metrics['rmse']:.4f}"),
                ("R2", f"{metrics['r2']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_activity_suitability_test()
    print("Activity suitability test evaluation complete.")
    print(json.dumps(result, indent=2))

