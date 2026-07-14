from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import write_markdown_report
from ml.src.evaluation.evaluate_activity_go_no_test import evaluate_activity_go_no_test
from ml.src.evaluation.evaluate_activity_suitability_test import evaluate_activity_suitability_test
from ml.src.utils.paths import ACTIVITY_ALL_TEST_REPORT_PATH


def evaluate_activity_all_test(
    split_config: SplitConfig | None = None,
    write_reports: bool = True,
    all_report_path: Path | None = None,
) -> dict:
    go_no = evaluate_activity_go_no_test(split_config=split_config, write_report=write_reports)
    suitability = evaluate_activity_suitability_test(split_config=split_config, write_report=write_reports)

    if write_reports:
        write_markdown_report(
            path=all_report_path or ACTIVITY_ALL_TEST_REPORT_PATH,
            title="Activity Test Metrics",
            rows=[
                ("Go/No-Go F1", f"{go_no['f1']:.4f}"),
                ("Go/No-Go Accuracy", f"{go_no['accuracy']:.4f}"),
                ("Activity Suitability MAE", f"{suitability['mae']:.4f}"),
                ("Activity Suitability RMSE", f"{suitability['rmse']:.4f}"),
                ("Activity Suitability R2", f"{suitability['r2']:.4f}"),
            ],
        )

    return {
        "go_no": go_no,
        "suitability": suitability,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate activity models on test split.")
    parser.add_argument(
        "--strategy",
        choices=["chronological_by_date_windows", "monthly_chronological_ratio"],
        default="monthly_chronological_ratio",
    )
    parser.add_argument("--val-days", type=int, default=10)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--monthly-train-ratio", type=float, default=0.70)
    parser.add_argument("--monthly-val-ratio", type=float, default=0.15)
    parser.add_argument("--monthly-test-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = SplitConfig(
        strategy=args.strategy,
        val_days=args.val_days,
        test_days=args.test_days,
        monthly_train_ratio=args.monthly_train_ratio,
        monthly_val_ratio=args.monthly_val_ratio,
        monthly_test_ratio=args.monthly_test_ratio,
    )
    result = evaluate_activity_all_test(split_config=config, write_reports=True)
    print("Activity all-test evaluation complete.")
    print(json.dumps(result, indent=2))

