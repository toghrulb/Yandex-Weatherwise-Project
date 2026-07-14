from __future__ import annotations

import numpy as np


def quantize_activity_suitability(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, 10.0)
    return np.rint(arr)

