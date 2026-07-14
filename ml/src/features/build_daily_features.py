from __future__ import annotations

import numpy as np
import pandas as pd


DAILY_NUMERIC_FEATURES = [
    "month",
    "day_of_week",
    "is_weekend",
    "temp_min_c",
    "temp_max_c",
    "temp_avg_c",
    "feels_like_min_c",
    "feels_like_max_c",
    "total_precipitation_mm",
    "snow_hours",
    "rain_hours",
    "max_wind_kmh",
    "max_gust_kmh",
    "avg_humidity_pct",
    "max_uv_index",
    "thunderstorm_occurred",
    "month_sin",
    "month_cos",
]

DAILY_CATEGORICAL_FEATURES = [
    "station_id",
    "climate_zone",
    "season",
    "dominant_condition",
]

DAILY_ALL_FEATURES = DAILY_NUMERIC_FEATURES + DAILY_CATEGORICAL_FEATURES


def _ensure_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "month" not in df.columns:
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            df["month"] = d.dt.month
        else:
            df["month"] = np.nan

    if "day_of_week" not in df.columns:
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            df["day_of_week"] = d.dt.dayofweek
        else:
            df["day_of_week"] = np.nan

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (
            pd.to_numeric(df["day_of_week"], errors="coerce").fillna(-1).isin([5, 6]).astype(int)
        )
    return df


def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_numeric(df["month"], errors="coerce").fillna(1)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def build_daily_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = _ensure_temporal_columns(df)
    df = _add_cyclical_time_features(df)

    for col in DAILY_ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    for col in DAILY_NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[DAILY_ALL_FEATURES]

