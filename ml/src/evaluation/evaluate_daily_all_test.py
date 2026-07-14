from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import write_markdown_report
from ml.src.evaluation.evaluate_daily_best_hour_test import evaluate_daily_best_hour_test
from ml.src.evaluation.evaluate_daily_clothing_test import evaluate_daily_clothing_test
from ml.src.evaluation.evaluate_daily_suitability_test import evaluate_daily_suitability_test
from ml.src.evaluation.evaluate_daily_umbrella_test import evaluate_daily_umbrella_test
from ml.src.utils.paths import DAILY_ALL_TEST_REPORT_PATH


def evaluate_daily_all_test(
    split_config: SplitConfig | None = None,
    write_reports: bool = True,
    all_report_path: Path | None = None,
) -> dict:
    umbrella = evaluate_daily_umbrella_test(split_config=split_config, write_report=write_reports)
    best_hour = evaluate_daily_best_hour_test(split_config=split_config, write_report=write_reports)
    clothing = evaluate_daily_clothing_test(split_config=split_config, write_report=write_reports)
    suitability = evaluate_daily_suitability_test(split_config=split_config, write_report=write_reports)

    if write_reports:
        write_markdown_report(
            path=all_report_path or DAILY_ALL_TEST_REPORT_PATH,
            title="Daily Summary Test Metrics",
            rows=[
                ("Umbrella F1", f"{umbrella['f1']:.4f}"),
                ("Best Hour Accuracy", f"{best_hour['accuracy']:.4f}"),
                ("Best Hour Within +/-1h", f"{best_hour['within_1h_accuracy']:.4f}"),
                ("Clothing Morning Accuracy", f"{clothing['morning']['accuracy']:.4f}"),
                ("Clothing Afternoon Accuracy", f"{clothing['afternoon']['accuracy']:.4f}"),
                ("Clothing Evening Accuracy", f"{clothing['evening']['accuracy']:.4f}"),
                ("Avg Suitability MAE", f"{suitability['mae']:.4f}"),
                ("Avg Suitability R2", f"{suitability['r2']:.4f}"),
            ],
        )

    return {
        "umbrella": umbrella,
        "best_hour": best_hour,
        "clothing": clothing,
        "suitability": suitability,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate daily summary models on test split.")
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
    result = evaluate_daily_all_test(split_config=config, write_reports=True)
    print("Daily all-test evaluation complete.")
    print(json.dumps(result, indent=2))

