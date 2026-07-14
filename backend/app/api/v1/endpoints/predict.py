from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.predict import PredictRequest, PredictResponse
from backend.app.services.inference import inference_service


router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        prediction = inference_service.predict(data)
        return PredictResponse(**prediction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

