from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

import pandas as pd

from ml.src.data.split_manager import SplitConfig
from ml.src.utils.paths import ACTIVITY_SPLIT_ASSIGNMENTS_PATH, ACTIVITY_SPLIT_MANIFEST_PATH


def _config_dict(cfg: SplitConfig) -> dict[str, Any]:
    return asdict(cfg)


def _prepare_working_df(df: pd.DataFrame) -> pd.DataFrame:
    if "rec_id" not in df.columns:
        raise ValueError("Input dataframe must include `rec_id` column for stable split mapping.")
    if "timestamp" not in df.columns:
        raise ValueError("Input dataframe must include `timestamp` for chronological split.")

    working = df.copy()
    working["rec_id"] = working["rec_id"].astype(str)
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["rec_id", "timestamp"]).copy()
    working["date"] = working["timestamp"].dt.floor("D")
    working["month_period"] = working["timestamp"].dt.to_period("M").astype(str)
    return working


def _build_split_by_global_date_windows(working: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    unique_dates = sorted(working["date"].dropna().unique().tolist())
    required_days = cfg.val_days + cfg.test_days + 1
    if len(unique_dates) < required_days:
        raise ValueError(
            f"Not enough unique dates ({len(unique_dates)}) for split with "
            f"val_days={cfg.val_days} and test_days={cfg.test_days}."
        )

    test_dates = set(unique_dates[-cfg.test_days :])
    val_dates = set(unique_dates[-(cfg.test_days + cfg.val_days) : -cfg.test_days])

    working["split"] = "train"
    working.loc[working["date"].isin(val_dates), "split"] = "val"
    working.loc[working["date"].isin(test_dates), "split"] = "test"
    return working


def _monthly_day_counts(n_days: int, cfg: SplitConfig) -> tuple[int, int, int]:
    train_days = max(1, int(math.floor(n_days * cfg.monthly_train_ratio)))
    val_days = max(1, int(math.floor(n_days * cfg.monthly_val_ratio)))
    test_days = n_days - train_days - val_days

    if test_days < 1:
        deficit = 1 - test_days
        reduction_from_train = min(deficit, max(0, train_days - 1))
        train_days -= reduction_from_train
        deficit -= reduction_from_train
        if deficit > 0:
            reduction_from_val = min(deficit, max(0, val_days - 1))
            val_days -= reduction_from_val
            deficit -= reduction_from_val
        test_days = n_days - train_days - val_days

    if train_days < 1 or val_days < 1 or test_days < 1:
        raise ValueError(f"Could not allocate monthly split for n_days={n_days}.")
    return train_days, val_days, test_days


def _build_split_by_monthly_ratio(working: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    working["split"] = ""
    unique_months = sorted(working["month_period"].dropna().unique().tolist())

    for month in unique_months:
        month_mask = working["month_period"] == month
        month_dates = sorted(working.loc[month_mask, "date"].dropna().unique().tolist())
        n_days = len(month_dates)
        if n_days < 3:
            raise ValueError(
                f"Month {month} has only {n_days} unique day(s), need at least 3 for train/val/test."
            )

        train_n, val_n, _ = _monthly_day_counts(n_days=n_days, cfg=cfg)
        train_dates = set(month_dates[:train_n])
        val_dates = set(month_dates[train_n : train_n + val_n])
        test_dates = set(month_dates[train_n + val_n :])

        month_indices = working.index[month_mask]
        month_df = working.loc[month_indices]
        working.loc[month_df.index[month_df["date"].isin(train_dates)], "split"] = "train"
        working.loc[month_df.index[month_df["date"].isin(val_dates)], "split"] = "val"
        working.loc[month_df.index[month_df["date"].isin(test_dates)], "split"] = "test"

    if (working["split"] == "").any():
        missing = int((working["split"] == "").sum())
        raise ValueError(f"Split assignment failed for {missing} rows.")
    return working


def _build_split_assignments(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    cfg.validate()
    working = _prepare_working_df(df)

    if cfg.strategy == "chronological_by_date_windows":
        working = _build_split_by_global_date_windows(working=working, cfg=cfg)
    elif cfg.strategy == "monthly_chronological_ratio":
        working = _build_split_by_monthly_ratio(working=working, cfg=cfg)
    else:
        raise ValueError(f"Unsupported split strategy: {cfg.strategy}")

    assignments = working[["rec_id", "timestamp", "split"]].sort_values(["timestamp", "rec_id"])
    return assignments.reset_index(drop=True)


def _save_split_artifacts(assignments: pd.DataFrame, cfg: SplitConfig) -> None:
    ACTIVITY_SPLIT_ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(ACTIVITY_SPLIT_ASSIGNMENTS_PATH, index=False)

    split_counts = assignments["split"].value_counts().to_dict()
    manifest = {
        "version": 1,
        "strategy": cfg.strategy,
        "config": _config_dict(cfg),
        "rows": int(len(assignments)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "boundaries": {
            "train_start": str(assignments.loc[assignments["split"] == "train", "timestamp"].min()),
            "train_end": str(assignments.loc[assignments["split"] == "train", "timestamp"].max()),
            "val_start": str(assignments.loc[assignments["split"] == "val", "timestamp"].min()),
            "val_end": str(assignments.loc[assignments["split"] == "val", "timestamp"].max()),
            "test_start": str(assignments.loc[assignments["split"] == "test", "timestamp"].min()),
            "test_end": str(assignments.loc[assignments["split"] == "test", "timestamp"].max()),
        },
        "files": {
            "assignments_csv": str(ACTIVITY_SPLIT_ASSIGNMENTS_PATH),
        },
    }

    ACTIVITY_SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVITY_SPLIT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _manifest_matches_config(cfg: SplitConfig) -> bool:
    if not ACTIVITY_SPLIT_MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(ACTIVITY_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False

    if manifest.get("strategy") != cfg.strategy:
        return False
    return manifest.get("config", {}) == _config_dict(cfg)


def ensure_activity_split_assignments(
    df: pd.DataFrame,
    cfg: SplitConfig | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cfg = cfg or SplitConfig()
    cfg.validate()

    if ACTIVITY_SPLIT_ASSIGNMENTS_PATH.exists() and not force_rebuild:
        assignments = pd.read_csv(ACTIVITY_SPLIT_ASSIGNMENTS_PATH)
        assignments["rec_id"] = assignments["rec_id"].astype(str)
        assignments["timestamp"] = pd.to_datetime(assignments["timestamp"], errors="coerce")

        source_ids = set(df["rec_id"].astype(str).tolist())
        split_ids = set(assignments["rec_id"].astype(str).tolist())
        is_complete = source_ids == split_ids and len(assignments) == len(df)
        has_valid_cfg = _manifest_matches_config(cfg)

        if is_complete and has_valid_cfg:
            return assignments

    assignments = _build_split_assignments(df=df, cfg=cfg)
    _save_split_artifacts(assignments=assignments, cfg=cfg)
    return assignments


def apply_activity_split_assignments(
    df: pd.DataFrame,
    cfg: SplitConfig | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if "rec_id" not in df.columns:
        raise ValueError("Input dataframe must include `rec_id` column.")
    if "timestamp" not in df.columns:
        raise ValueError("Input dataframe must include `timestamp` column.")

    assignments = ensure_activity_split_assignments(df=df, cfg=cfg, force_rebuild=force_rebuild)
    merged = df.copy()
    merged["rec_id"] = merged["rec_id"].astype(str)
    merged = merged.merge(assignments[["rec_id", "split"]], on="rec_id", how="left", validate="1:1")

    missing = merged["split"].isna().sum()
    if missing:
        raise ValueError(f"Split mapping missing for {missing} rows. Rebuild split artifacts.")
    return merged

