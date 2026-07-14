from __future__ import annotations

import numpy as np
import pandas as pd


ACTIVITY_NUMERIC_FEATURES = [
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "cloud_cover_pct",
    "uv_index",
    "hour_of_day",
    "month",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

ACTIVITY_CATEGORICAL_FEATURES = [
    "station_id",
    "season",
    "weather_condition",
    "activity_type",
]

ACTIVITY_ALL_FEATURES = ACTIVITY_NUMERIC_FEATURES + ACTIVITY_CATEGORICAL_FEATURES


def _ensure_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "hour_of_day" not in df.columns:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour_of_day"] = ts.dt.hour
        else:
            df["hour_of_day"] = np.nan

    if "month" not in df.columns:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df["month"] = ts.dt.month
        else:
            df["month"] = np.nan

    if "day_of_week" not in df.columns:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df["day_of_week"] = ts.dt.dayofweek
        else:
            df["day_of_week"] = np.nan

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (
            pd.to_numeric(df["day_of_week"], errors="coerce").fillna(-1).isin([5, 6]).astype(int)
        )

    return df


def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = pd.to_numeric(df["hour_of_day"], errors="coerce").fillna(0)
    month = pd.to_numeric(df["month"], errors="coerce").fillna(1)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def build_activity_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = _ensure_temporal_columns(df)
    df = _add_cyclical_time_features(df)

    for col in ACTIVITY_ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    for col in ACTIVITY_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[ACTIVITY_ALL_FEATURES]

