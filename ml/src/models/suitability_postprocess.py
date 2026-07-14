from __future__ import annotations

import numpy as np


DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_SCORE = 10.0
DEFAULT_SUITABILITY_LEVELS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)


def quantize_suitability(
    values: np.ndarray | list[float],
    allowed_values: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Clip suitability predictions and snap to discrete score levels."""
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, DEFAULT_MIN_SCORE, DEFAULT_MAX_SCORE)

    if allowed_values is None:
        allowed = DEFAULT_SUITABILITY_LEVELS
    else:
        allowed = np.asarray(allowed_values, dtype=float)
        allowed = np.unique(np.clip(allowed, DEFAULT_MIN_SCORE, DEFAULT_MAX_SCORE))

    if allowed.size == 0:
        allowed = DEFAULT_SUITABILITY_LEVELS

    nearest_idx = np.argmin(np.abs(arr.reshape(-1, 1) - allowed.reshape(1, -1)), axis=1)
    return allowed[nearest_idx]
