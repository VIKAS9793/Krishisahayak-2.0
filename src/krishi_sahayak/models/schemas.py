# src/krishi_sahayak/models/schemas.py
"""
Pydantic models for model-related data structures in KrishiSahayak. (Refactored)

This module is the single source of truth for the model's data contracts (input/
output), its metadata, and specific configurations like distillation. It has been
purified to exclude pipeline-level training configurations.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    # Teacher Models
    EFFICIENTNET_B0 = "efficientnet_b0"
    
    # Student Models
    MOBILENET_V3_SMALL_075 = "mobilenet_v3_small_075"
    MOBILENET_V3_SMALL_050 = "mobilenet_v3_small_050"
    SHUFFLENET_V2_X05 = "shufflenet_v2_x05"
    
    # Legacy/Alternative Models
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    EFFICIENTNET_B4 = "efficientnet_b4"
    VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
    SWIN_BASE_PATCH4_WINDOW7_224 = "swin_base_patch4_window7_224"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Supported task types."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"
    GAN = "gan"

# -----------------------------------------------------------------------------
# Configuration Models Specific to the Model Package
# -----------------------------------------------------------------------------

class DistillationConfig(BaseModel):
    """Configuration for knowledge distillation."""
    enabled: bool = False
    teacher_architecture: ModelArchitecture = ModelArchitecture.EFFICIENTNET_B0
    teacher_checkpoint: Optional[Path] = None
    temperature: float = Field(2.0, gt=0, description="Temperature for softening probabilities.")
    alpha: float = Field(0.5, ge=0, le=1, description="Weight balance between student and distillation loss.")
    hard_label_weight: float = Field(0.5, ge=0, le=1, description="Weight of the hard label loss within the student's loss component.")

# -----------------------------------------------------------------------------
# Core Data Contract Models
# -----------------------------------------------------------------------------

class ModelInput(BaseModel):
    """Base model for model input data."""
    image: np.ndarray = Field(..., description="Input image as numpy array")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the input")

    class Config:
        arbitrary_types_allowed = True

    @field_validator('image')
    @classmethod
    def validate_image(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("Image must be a numpy array")
        if v.ndim not in (3, 4):
            raise ValueError(f"Expected image with 3 or 4 dimensions (H,W,C) or (N,H,W,C), got {v.shape}")
        return v

class ModelOutput(BaseModel):
    """Base model for model output data."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results with class probabilities")
    logits: Optional[torch.Tensor] = Field(None, description="Raw model output logits")
    features: Optional[torch.Tensor] = Field(None, description="Feature embeddings from the model")

    class Config:
        arbitrary_types_allowed = True

class ModelMetadata(BaseModel):
    """Metadata about a trained model, suitable for a model registry."""
    name: str = Field(..., description="Model name, often including architecture and dataset.")
    version: str = Field("1.0.0", description="Model version (e.g., semantic versioning).")
    run_id: Optional[str] = Field(None, description="Unique ID from the experiment tracking run (e.g., WandB or MLflow).")
    architecture: ModelArchitecture
    task: TaskType
    num_classes: int = Field(..., gt=0)
    input_size: Tuple[int, int] = Field((224, 224), description="Input image size as (height, width).")
    mean: List[float] = Field([0.485, 0.456, 0.406], description="Normalization mean for each channel.")
    std: List[float] = Field([0.229, 0.224, 0.225], description="Normalization standard deviation for each channel.")
    class_names: List[str] = Field(..., description="List of class names in the order of model outputs.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the model artifact was created.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Key evaluation metrics (e.g., accuracy, F1-score).")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters.")

    @field_validator('class_names')
    @classmethod
    def validate_class_names_length(cls, v: List[str], info) -> List[str]:
        if 'num_classes' in info.data and len(v) != info.data['num_classes']:
            raise ValueError("Number of class names must match the 'num_classes' field.")
        return v