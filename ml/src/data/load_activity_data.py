from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.utils.paths import ACTIVITY_DATA_PATH


def load_activity_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load activity recommendation dataset with parsed timestamp."""
    data_path = path or ACTIVITY_DATA_PATH
    df = pd.read_csv(data_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df

