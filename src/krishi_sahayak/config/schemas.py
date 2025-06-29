# src/krishi_sahayak/config/schemas.py
"""KrishiSahayak - Centralized Pydantic Schemas for high-level pipeline configuration."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from krishi_sahayak.models.base import BaseModelConfig
from krishi_sahayak.models.schemas import ModelConfig, DistillationConfig

class JobType(str, Enum):
    CLASSIFICATION = "classification"
    GAN = "gan"

class PrepareJobConfig(BaseModel):
    handler_type: str
    source_subdir: str
    output_filename: str
    splitting_config: Optional[Dict[str, Any]] = None
    dataset_prefix: Optional[str] = None

class PathsConfig(BaseModel):
    data_root: Path
    output_root: Path
    log_dir: Path
    checkpoint_dir: Path
    output_dir: Path = Field(default_factory=lambda: Path("output/default_job_output"))
    teacher_checkpoint: Optional[Path] = None

class CallbacksConfig(BaseModel):
    early_stopping: Dict[str, Any] = Field(default_factory=dict)
    model_checkpoint: Dict[str, Any] = Field(default_factory=dict)

class DataLoaderParams(BaseModel):
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True

class TrainingConfig(BaseModel):
    type: str
    description: Optional[str] = None
    experiment_name: str
    run_test_after_fit: bool = True
    data_loader_params: DataLoaderParams = Field(default_factory=DataLoaderParams)
    data_params: Dict[str, Any] = Field(default_factory=dict)
    architecture_config: ModelConfig
    training_params: BaseModelConfig
    distillation: Optional[DistillationConfig] = None
    trainer_config: Dict[str, Any] = Field(default_factory=dict)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)

class MasterConfig(BaseModel):
    project_name: str
    seed: int
    paths: PathsConfig
    data_preparation: Dict[str, PrepareJobConfig]
    training_pipelines: Dict[str, TrainingConfig]