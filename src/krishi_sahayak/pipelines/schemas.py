"""
KrishiSahayak - Centralized Pydantic Configuration Schemas (Source of Truth)

This module is the single source of truth for the structure and validation of all
project configurations.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import confloat, conint, constr

# --- Core Enumerations ---
class JobType(str, Enum):
    CLASSIFICATION = "classification"
    GAN = "gan"
    DISTILLATION = "distillation"

# --- Shared Nested Configuration Models ---
class PathsConfig(BaseModel):
    data_root: Path
    output_root: Path
    log_dir: Path
    checkpoint_dir: Path
    teacher_checkpoint: Optional[Path] = None

class CallbacksConfig(BaseModel):
    early_stopping: Dict[str, Any] = Field(default_factory=dict)
    model_checkpoint: Dict[str, Any] = Field(default_factory=dict)

class DataLoaderParams(BaseModel):
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True

class TrainingParams(BaseModel):
    batch_size: int = Field(32, gt=0)
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)

# --- Master Training Job Schema ---
class TrainingConfig(BaseModel):
    """The canonical Pydantic model for a single training job."""
    model_config = ConfigDict(extra="allow", use_enum_values=True)

    type: JobType
    description: Optional[str] = None
    
    # Nested Configs
    data_loader_params: DataLoaderParams = Field(default_factory=DataLoaderParams)
    data_params: Dict[str, Any] = Field(default_factory=dict)
    model_config: Dict[str, Any] = Field(default_factory=dict)
    training_params: TrainingParams = Field(default_factory=TrainingParams)
    trainer_config: Dict[str, Any] = Field(default_factory=dict)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    
    # Experiment Tracking
    project_name: str = "krishi-sahayak"
    experiment_name: str
    run_test_after_fit: bool = True
