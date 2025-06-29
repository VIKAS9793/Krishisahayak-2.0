# src/krishi_sahayak/models/schemas.py
"""
Pydantic Schemas for CONCRETE Model-Related Configurations and Data Contracts.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# --- Enums for Type-Safe Configuration ---
class ModelArchitecture(str, Enum):
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENET_V3_SMALL_075 = "mobilenet_v3_small_075"
    RESNET18 = "resnet18"

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    GAN = "gan"

# --- Unified Model Architecture Schemas ---
class StreamConfig(BaseModel):
    channels: int = Field(..., gt=0)
    adapter_out: Optional[int] = Field(None, gt=0)
    pretrained: bool = False

class FusionConfig(BaseModel):
    method: Literal["concat", "add", "attention", "cross_attention"] = "concat"
    num_heads: int = Field(8, gt=0)
    dropout_rate: float = Field(0.1, ge=0, le=1)

class ModelConfig(BaseModel):
    backbone_name: str
    streams: Dict[str, StreamConfig]
    fusion: Optional[FusionConfig] = None
    classifier_hidden_dim: Optional[int] = Field(None, gt=0)
    classifier_dropout: float = Field(0.0, ge=0, le=1)

# --- Distillation Schema ---
class DistillationConfig(BaseModel):
    enabled: bool = False
    teacher_architecture: Optional[ModelArchitecture] = None
    teacher_checkpoint: Optional[str] = None
    temperature: float = Field(2.0, gt=0)
    alpha: float = Field(0.5, ge=0, le=1)

# NOTE: BaseModelConfig was moved to src/krishi_sahayak/models/base/base.py
# to be co-located with its consumer, the BaseModel class.