"""
Load CSV datasets into memory at application startup.
"""
from __future__ import annotations

import pandas as pd
from backend.app.core.config import DATA_RAW


class DataStore:
    """Singleton-style container for in-memory DataFrames."""

    hourly: pd.DataFrame | None = None
    daily: pd.DataFrame | None = None
    activities: pd.DataFrame | None = None
    forecasts: pd.DataFrame | None = None

    @classmethod
    def load(cls) -> None:
        """Read all CSV files and do basic type coercion."""

        hourly_path = DATA_RAW / "hourly_observations.csv"
        daily_path = DATA_RAW / "daily_summaries.csv"
        activity_path = DATA_RAW / "activity_recommendations.csv"
        forecast_path = DATA_RAW / "forecast_vs_actual.csv"

        if hourly_path.exists():
            cls.hourly = pd.read_csv(hourly_path, parse_dates=["timestamp", "date"])
        if daily_path.exists():
            cls.daily = pd.read_csv(daily_path, parse_dates=["date"])
        if activity_path.exists():
            cls.activities = pd.read_csv(activity_path, parse_dates=["timestamp"])
        if forecast_path.exists():
            cls.forecasts = pd.read_csv(
                forecast_path,
                parse_dates=["target_timestamp", "forecast_issued_at"],
            )

        loaded = sum(1 for df in [cls.hourly, cls.daily, cls.activities, cls.forecasts] if df is not None)
        print(f"[DataStore] Loaded {loaded}/4 datasets from {DATA_RAW}")
