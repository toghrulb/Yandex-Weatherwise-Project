from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.utils.paths import HOURLY_DATA_PATH


def load_hourly_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the main hourly WeatherWise dataset with parsed date columns."""
    data_path = path or HOURLY_DATA_PATH
    df = pd.read_csv(data_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df
