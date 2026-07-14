from __future__ import annotations

import argparse
import json

from ml.src.data.activity_split_manager import ensure_activity_split_assignments
from ml.src.data.load_activity_data import load_activity_data
from ml.src.data.split_manager import SplitConfig


def create_activity_split(cfg: SplitConfig, force_rebuild: bool = False) -> dict:
    df = load_activity_data()
    assignments = ensure_activity_split_assignments(
        df=df,
        cfg=cfg,
        force_rebuild=force_rebuild,
    )
    counts = assignments["split"].value_counts().to_dict()
    return {k: int(v) for k, v in counts.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create persisted train/val/test split for activity dataset.")
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
    parser.add_argument("--force-rebuild", action="store_true")
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
    stats = create_activity_split(cfg=config, force_rebuild=args.force_rebuild)
    print("Activity split creation complete.")
    print(json.dumps(stats, indent=2))

