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

from ml.src.data.activity_split_manager import apply_activity_split_assignments
from ml.src.data.load_activity_data import load_activity_data
from ml.src.data.split_manager import SplitConfig
from ml.src.features.build_activity_features import (
    ACTIVITY_CATEGORICAL_FEATURES,
    ACTIVITY_NUMERIC_FEATURES,
    build_activity_feature_frame,
)
from ml.src.utils.paths import ARTIFACTS_DIR


@dataclass
class ActivityTargetFrames:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_rows: int
    val_rows: int
    test_rows: int


def build_activity_preprocessor() -> ColumnTransformer:
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
                ACTIVITY_NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ACTIVITY_CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_activity_target_frames(
    target_column: str,
    force_rebuild_split: bool = False,
    split_config: SplitConfig | None = None,
) -> ActivityTargetFrames:
    df = load_activity_data()
    if "timestamp" not in df.columns:
        raise ValueError("Activity dataset must contain `timestamp` column.")
    df = df.dropna(subset=["rec_id", "timestamp"]).copy()

    df = apply_activity_split_assignments(
        df=df,
        cfg=split_config,
        force_rebuild=force_rebuild_split,
    )
    df = df.dropna(subset=[target_column]).copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    return ActivityTargetFrames(
        x_train=build_activity_feature_frame(train_df),
        y_train=train_df[target_column],
        x_val=build_activity_feature_frame(val_df),
        y_val=val_df[target_column],
        x_test=build_activity_feature_frame(test_df),
        y_test=test_df[target_column],
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        test_rows=int(len(test_df)),
    )


def fit_activity_pipeline(estimator: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_activity_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def save_activity_model(path: Path, model: Any) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_activity_training_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

