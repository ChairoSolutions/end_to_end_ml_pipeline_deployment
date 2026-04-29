"""FastAPI application for the end-to-end ML deployment project."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.schemas import PredictionRequest, PredictionResponse
from src.service import get_model, predict
from src.utils import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="End-to-End ML Pipeline API", version="1.0.0")

try:
    model = get_model()
except Exception as exc:  # pragma: no cover
    logger.warning("Model could not be loaded at startup: %s", exc)
    model = None


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the pipeline first.")

    try:
        result = predict(request.features, model)
        return PredictionResponse(**result)
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Invalid input for prediction: {exc}") from exc
