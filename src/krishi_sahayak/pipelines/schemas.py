"""
Pydantic schemas and enumerations for configuring and managing training pipelines.

This module serves as the single source of truth for all configuration contracts,
ensuring that any job configuration is validated, self-documenting, and robust.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import confloat, conint, constr


# --- Core Enumerations ---

class JobType(str, Enum):
    """Supported training job types."""
    CLASSIFICATION = "classification"
    GAN = "gan"
    REGRESSION = "regression"


# --- Configuration Models ---

class PathsConfig(BaseModel):
    """
    Configuration for file paths.
    Generates dynamic default paths for outputs and checkpoints if not provided.
    """
    data_dir: Path
    root_output_dir: Path = Path("output")
    log_dir: Path | None = None
    checkpoint_dir: Path | None = None
    output_dir: Path | None = None
    
    @field_validator('data_dir')
    @classmethod
    def data_dir_must_exist(cls, v: Path) -> Path:
        """Validates that the source data directory exists."""
        if not v.exists():
            raise ValueError(f"Data directory does not exist: {v.resolve()}")
        return v
    
    @model_validator(mode='after')
    def set_dynamic_paths(self) -> 'PathsConfig':
        """Sets default output/checkpoint paths based on the root output dir."""
        if self.log_dir is None:
            self.log_dir = self.root_output_dir / "logs"
        if self.output_dir is None:
            self.output_dir = self.root_output_dir / "jobs"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.root_output_dir / "checkpoints"
        return self


class MonitoringConfig(BaseModel):
    """Configuration for resource monitoring and alerting thresholds."""
    enable_resource_monitoring: bool = True
    memory_threshold_mb: confloat(gt=0) | None = None
    cpu_threshold_percent: confloat(gt=0, le=100) | None = None


class LoggerConfig(BaseModel):
    """Configuration for the experiment logger."""
    type: str = Field("tensorboard", pattern=r"^(tensorboard|wandb)$")


class ModelCheckpointConfig(BaseModel):
    """Configuration for the ModelCheckpoint callback."""
    enable: bool = True
    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: conint(ge=1) = 1
    save_last: bool = True


class EarlyStoppingConfig(BaseModel):
    """Configuration for the EarlyStopping callback."""
    enable: bool = True
    monitor: str = "val/loss"
    mode: str = "min"
    patience: conint(ge=1) = 5


class CallbacksConfig(BaseModel):
    """Container for all callback configurations."""
    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class TrainingConfig(BaseModel):
    """
    The root Pydantic model for complete training configuration validation.
    This is the single source of truth for defining a training job.
    """
    # Allow extra fields for model-specific hyperparameters
    model_config = ConfigDict(extra="allow", use_enum_values=True, validate_assignment=True)

    # Core Job Details
    type: JobType
    name: constr(min_length=1, max_length=100, regex=r'^[a-zA-Z0-9_-]+$')
    description: str | None = None
    project_name: constr(min_length=1) = "krishi-sahayak"
    experiment_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    resume_from_checkpoint: Path | None = None

    # Training Hyperparameters
    epochs: conint(ge=1) = 100
    batch_size: conint(ge=1) = 32
    learning_rate: confloat(gt=0.0) = 1e-3
    
    # Nested Configuration Models
    paths: PathsConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
        
    @field_validator('experiment_name', mode='before')
    @classmethod
    def set_experiment_name_if_none(cls, v: str | None, info: 'ValidationInfo') -> str:
        """Generates a unique experiment name if one is not provided."""
        if v is None:
            job_name = info.data.get('name', 'unnamed_job')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{job_name}_{timestamp}"
        return v
    
    @field_validator('resume_from_checkpoint')
    @classmethod
    def validate_checkpoint_exists(cls, v: Path | None) -> Path | None:
        """Validates that the checkpoint file to resume from actually exists."""
        if v and not v.exists():
            raise ValueError(f"Checkpoint file to resume from does not exist: {v.resolve()}")
        return v
