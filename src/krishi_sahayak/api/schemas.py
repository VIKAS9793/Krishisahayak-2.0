# src/krishi_sahayak/api/schemas.py
"""Pydantic schemas for API request and response models."""
from pydantic import BaseModel, Field
from typing import List, Optional

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., example="healthy")
    version: str = Field("1.0.0", example="1.0.0")
    device: str = Field(..., example="cuda")

class Prediction(BaseModel):
    """A single prediction with class name and confidence."""
    class_name: str = Field(..., example="Tomato___Late_blight")
    display_name: str = Field(..., example="Tomato Late Blight")
    confidence: float = Field(..., example=0.987, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    """The final prediction response returned to the user."""
    filename: str = Field(..., example="uploaded_image.jpg")
    predictions: List[Prediction]
    model_checkpoint: str
    explanation_image: Optional[str] = Field(None, example="data:image/png;base64,iVBORw0KGgoAAA...", description="Base64 encoded explanation heatmap image.")