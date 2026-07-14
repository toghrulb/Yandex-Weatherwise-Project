from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml.src.features.build_features import build_feature_frame
from ml.src.models.suitability_postprocess import quantize_suitability
from ml.src.utils.paths import CLOTHING_MODEL_PATH, SUITABILITY_MODEL_PATH, UMBRELLA_MODEL_PATH


class BaselinePredictor:
    """Lightweight local predictor for quick script-level checks."""

    def __init__(self, artifact_dir: Path | None = None) -> None:
        if artifact_dir is None:
            umbrella_path = UMBRELLA_MODEL_PATH
            clothing_path = CLOTHING_MODEL_PATH
            suitability_path = SUITABILITY_MODEL_PATH
        else:
            umbrella_path = artifact_dir / UMBRELLA_MODEL_PATH.name
            clothing_path = artifact_dir / CLOTHING_MODEL_PATH.name
            suitability_path = artifact_dir / SUITABILITY_MODEL_PATH.name

        self.umbrella_model = joblib.load(umbrella_path)
        self.clothing_model = joblib.load(clothing_path)
        self.suitability_model = joblib.load(suitability_path)

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([payload])
        features = build_feature_frame(frame)

        umbrella = int(self.umbrella_model.predict(features)[0])
        clothing = str(self.clothing_model.predict(features)[0])
        suitability_raw = float(self.suitability_model.predict(features)[0])
        suitability = float(quantize_suitability([suitability_raw])[0])

        return {
            "umbrella_needed": umbrella,
            "clothing_recommendation": clothing,
            "outdoor_suitability_score": suitability,
        }
