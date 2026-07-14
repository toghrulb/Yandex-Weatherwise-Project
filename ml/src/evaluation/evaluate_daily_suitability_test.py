from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import load_model, write_markdown_report
from ml.src.models.daily_common import load_daily_target_frames
from ml.src.utils.paths import DAILY_SUITABILITY_MODEL_PATH, DAILY_SUITABILITY_TEST_REPORT_PATH


def evaluate_daily_suitability_test(
    split_config: SplitConfig | None = None,
    report_path=None,
    write_report: bool = True,
) -> dict:
    model = load_model(DAILY_SUITABILITY_MODEL_PATH)
    frames = load_daily_target_frames(
        target_column="avg_outdoor_suitability",
        force_rebuild_split=False,
        split_config=split_config,
    )
    y_test = pd.to_numeric(frames.y_test, errors="coerce").fillna(0.0)
    pred = np.clip(model.predict(frames.x_test), 0, 10)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }

    if write_report:
        write_markdown_report(
            path=report_path or DAILY_SUITABILITY_TEST_REPORT_PATH,
            title="Daily Avg Outdoor Suitability Test Metrics",
            rows=[
                ("MAE", f"{metrics['mae']:.4f}"),
                ("RMSE", f"{metrics['rmse']:.4f}"),
                ("R2", f"{metrics['r2']:.4f}"),
            ],
        )
    return metrics


if __name__ == "__main__":
    result = evaluate_daily_suitability_test()
    print("Daily suitability test evaluation complete.")
    print(json.dumps(result, indent=2))

