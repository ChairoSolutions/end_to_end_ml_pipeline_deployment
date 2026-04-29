"""Service helpers for model inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from src.pipeline import load_pipeline


logger = logging.getLogger(__name__)
MODEL_PATH = Path("artifacts") / "model_pipeline.joblib"


def get_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    return load_pipeline(model_path)


def predict(features: Dict[str, Any], model: Pipeline) -> dict:
    frame = pd.DataFrame([features])
    prediction = int(model.predict(frame)[0])
    response = {"prediction": prediction}
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(frame)[0]
        if len(probabilities) == 2:
            response["probability"] = float(probabilities[1])
        else:
            response["class_probabilities"] = {
                str(label): float(score) for label, score in zip(classes, probabilities)
            }
    return response
