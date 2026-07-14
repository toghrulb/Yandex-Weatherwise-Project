from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.src.data.daily_split_manager import apply_daily_split_assignments
from ml.src.data.load_daily_data import load_daily_data
from ml.src.data.split_manager import SplitConfig
from ml.src.features.build_daily_features import (
    DAILY_CATEGORICAL_FEATURES,
    DAILY_NUMERIC_FEATURES,
    build_daily_feature_frame,
)
from ml.src.utils.paths import ARTIFACTS_DIR


@dataclass
class DailyTargetFrames:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_rows: int
    val_rows: int
    test_rows: int


def build_daily_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                DAILY_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                DAILY_CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_daily_target_frames(
    target_column: str,
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> DailyTargetFrames:
    df = load_daily_data()
    if "date" not in df.columns:
        raise ValueError("Daily dataset must contain `date` column.")
    df = df.dropna(subset=["summary_id", "date"]).copy()

    df = apply_daily_split_assignments(
        df=df,
        cfg=split_config,
        force_rebuild=force_rebuild_split,
    )
    df = df.dropna(subset=[target_column]).copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    return DailyTargetFrames(
        x_train=build_daily_feature_frame(train_df),
        y_train=train_df[target_column],
        x_val=build_daily_feature_frame(val_df),
        y_val=val_df[target_column],
        x_test=build_daily_feature_frame(test_df),
        y_test=test_df[target_column],
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        test_rows=int(len(test_df)),
    )


def fit_daily_pipeline(estimator: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_daily_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def fit_daily_pipeline_with_params(
    estimator: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    fit_params: dict[str, Any] | None = None,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_daily_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train, **(fit_params or {}))
    return pipeline


def save_daily_model(path: Path, model: Any) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_daily_training_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

