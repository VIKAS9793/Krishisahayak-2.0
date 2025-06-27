# src/krishi_sahayak/config/schemas.py
"""
KrishiSahayak - Centralized Pydantic Configuration Schemas

This module is the single source of truth for the structure and validation of all
project configurations, including training pipelines.
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from krishi_sahayak.models.schemas import BaseModelConfig, ModelConfig

class JobType(str, Enum):
    CLASSIFICATION = "classification"
    GAN = "gan"
    DISTILLATION = "distillation"

class PathsConfig(BaseModel):
    data_root: Path
    output_root: Path
    log_dir: Path
    checkpoint_dir: Path
    teacher_checkpoint: Optional[Path] = None
    output_dir: Path = Field(default_factory=lambda: Path("output/default_job_output"))

class CallbacksConfig(BaseModel):
    early_stopping: Dict[str, Any] = Field(default_factory=dict)
    model_checkpoint: Dict[str, Any] = Field(default_factory=dict)

class DataLoaderParams(BaseModel):
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True

class TrainingParams(BaseModel):
    batch_size: int = Field(32, gt=0)

class DistillationConfig(BaseModel):
    """Configuration for knowledge distillation."""
    enabled: bool = True
    teacher_checkpoint: Optional[Path] = None
    temperature: float = Field(2.0, gt=0, description="Temperature for softening the teacher's outputs")
    alpha: float = Field(0.3, ge=0, le=1, description="Weight for distillation loss (1-alpha) for student loss")
    hard_label_weight: float = Field(0.1, ge=0, description="Weight for hard labels in the loss")
    
    class Config:
        json_encoders = {Path: str}


class TrainingConfig(BaseModel):
    """The canonical Pydantic model for a single training job."""
    model_config = ConfigDict(extra="allow", use_enum_values=True)
    type: JobType
    description: Optional[str] = None
    data_loader_params: DataLoaderParams = Field(default_factory=DataLoaderParams)
    data_params: Dict[str, Any] = Field(default_factory=dict)
    model_config: ModelConfig = Field(default_factory=ModelConfig)
    training_params: BaseModelConfig = Field(default_factory=BaseModelConfig)
    trainer_config: Dict[str, Any] = Field(default_factory=dict)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    project_name: str = "krishi_sahayak"
    experiment_name: str
    run_test_after_fit: bool = True

class MasterConfig(BaseModel):
    """The root schema for the entire project configuration."""
    project_name: str
    seed: int
    paths: PathsConfig
    data_preparation: Dict[str, Dict[str, Any]]
    training_pipelines: Dict[str, Union[TrainingConfig, Dict[str, Any]]]
