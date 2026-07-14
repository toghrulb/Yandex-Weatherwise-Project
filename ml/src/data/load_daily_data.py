from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.utils.paths import DAILY_DATA_PATH


def load_daily_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load daily summary dataset with parsed date."""
    data_path = path or DAILY_DATA_PATH
    df = pd.read_csv(data_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df

