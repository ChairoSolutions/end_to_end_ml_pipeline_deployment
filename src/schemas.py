"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary to score.")


class PredictionResponse(BaseModel):
    prediction: Any
    probability: float | None = None
