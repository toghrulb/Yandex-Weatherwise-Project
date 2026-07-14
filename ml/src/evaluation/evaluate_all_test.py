from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.src.data.split_manager import SplitConfig
from ml.src.evaluation.common import write_markdown_report
from ml.src.evaluation.evaluate_clothing_test import evaluate_clothing_test
from ml.src.evaluation.evaluate_suitability_test import evaluate_suitability_test
from ml.src.evaluation.evaluate_umbrella_test import evaluate_umbrella_test
from ml.src.utils.paths import ALL_TEST_REPORT_PATH


def evaluate_all_test(
    split_config: SplitConfig | None = None,
    write_reports: bool = True,
    all_report_path: Path | None = None,
) -> dict:
    umbrella = evaluate_umbrella_test(split_config=split_config, write_report=write_reports)
    clothing = evaluate_clothing_test(split_config=split_config, write_report=write_reports)
    suitability = evaluate_suitability_test(split_config=split_config, write_report=write_reports)

    if write_reports:
        write_markdown_report(
            path=all_report_path or ALL_TEST_REPORT_PATH,
            title="All Test Metrics",
            rows=[
                ("Umbrella F1", f"{umbrella['f1']:.4f}"),
                ("Umbrella Accuracy", f"{umbrella['accuracy']:.4f}"),
                ("Clothing Accuracy", f"{clothing['accuracy']:.4f}"),
                ("Clothing Macro F1", f"{clothing['macro_f1']:.4f}"),
                ("Suitability MAE", f"{suitability['mae']:.4f}"),
                ("Suitability RMSE", f"{suitability['rmse']:.4f}"),
                ("Suitability R2", f"{suitability['r2']:.4f}"),
            ],
        )

    return {
        "umbrella": umbrella,
        "clothing": clothing,
        "suitability": suitability,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all models on test split.")
    parser.add_argument(
        "--strategy",
        choices=["chronological_by_date_windows", "monthly_chronological_ratio"],
        default="chronological_by_date_windows",
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
    result = evaluate_all_test(
        split_config=config,
        write_reports=True,
    )
    print("All test evaluations complete.")
    print(json.dumps(result, indent=2))

