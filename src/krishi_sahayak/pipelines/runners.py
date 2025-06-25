"""
Training pipeline runners for different model types.

This module provides specialized runner classes that encapsulate the training loop
and related functionality. It is built on a robust, Pydantic-based configuration
system to ensure validation and reproducibility.
"""
from __future__ import annotations
import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pydantic import BaseModel, Field
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Project components (assuming these are correctly defined elsewhere)
from krishi_sahayak.data.data_module import DataModuleConfig, PlantDiseaseDataModule
from krishi_sahayak.models import UnifiedModel
from krishi_sahayak.models.core import ModelConfig

# Optional GAN imports handled gracefully
try:
    from krishi_sahayak.models.gan.pix2pix import Pix2PixGAN, Pix2PixConfig
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    Pix2PixGAN = None
    Pix2PixConfig = None

logger = logging.getLogger(__name__)

# --- Pydantic Configuration Models for Validation and Clarity ---

class PathsConfig(BaseModel):
    """Configuration for all relevant file paths."""
    processed_data_dir: Path
    log_dir: Path = Path("output/logs")
    checkpoint_dir: Path = Path("output/checkpoints")

class LoggerConfig(BaseModel):
    """Configuration for the experiment logger."""
    type: str = "tensorboard" # 'wandb' or 'tensorboard'

class ModelCheckpointConfig(BaseModel):
    """Configuration for the ModelCheckpoint callback."""
    enable: bool = True
    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: int = 1
    save_last: bool = True

class EarlyStoppingConfig(BaseModel):
    """Configuration for the EarlyStopping callback."""
    enable: bool = True
    monitor: str = "val/loss"
    mode: str = "min"
    patience: int = 5

class CallbacksConfig(BaseModel):
    """Container for all callback configurations."""
    model_checkpoint: ModelCheckpointConfig = Field(default_factory=ModelCheckpointConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)

class ExperimentConfig(BaseModel):
    """The root configuration model for a training run."""
    paths: PathsConfig
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)
    data_loader_params: dict[str, Any] = Field(default_factory=dict)
    data_params: dict[str, Any] = Field(default_factory=dict)
    model_config: dict[str, Any] # Flexible dict, parsed by the specific runner
    training_params: dict[str, Any] = Field(default_factory=dict)
    trainer_config: dict[str, Any] = Field(default_factory=dict)
    experiment_name: str = "default_experiment"
    project_name: str = "krishi-sahayak"
    run_test_after_fit: bool = True

# --- Runner Implementations ---

class BaseRunner(ABC):
    """An abstract base class for a training task runner."""
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: ExperimentConfig = ExperimentConfig(**config)
        self.experiment_dir = self.config.paths.checkpoint_dir / self.config.experiment_name

    def run(self) -> None:
        """Executes the full training pipeline for the specific task."""
        logger.info(f"Starting training run for job: {self.config.experiment_name}")
        self._save_config()

        data_module = self._setup_data()
        model = self._setup_model(data_module)
        trainer = self._setup_trainer()
        
        logger.info("Starting model training...")
        trainer.fit(model, datamodule=data_module)
        
        if self.config.run_test_after_fit:
            logger.info("Training finished. Starting testing with the best checkpoint...")
            trainer.test(model, datamodule=data_module, ckpt_path="best")
        logger.info(f"Run '{self.config.experiment_name}' completed successfully.")

    @abstractmethod
    def _setup_data(self) -> pl.LightningDataModule:
        """Sets up the LightningDataModule."""
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule:
        """Sets up the LightningModule."""
        raise NotImplementedError

    def _save_config(self) -> None:
        """Saves the validated run configuration for reproducibility."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.experiment_dir / "run_config.yaml"
        # Convert Path objects to strings for clean YAML export
        config_dict = self.config.dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, indent=4)
        logger.info(f"Run configuration saved to {config_path}")

    def _setup_trainer(self) -> pl.Trainer:
        """Configures and returns the PyTorch Lightning Trainer."""
        logger_instance = self._setup_logger()
        callbacks = self._setup_callbacks()
        return pl.Trainer(logger=logger_instance, callbacks=callbacks, **self.config.trainer_config)

    def _setup_logger(self) -> WandbLogger | TensorBoardLogger:
        """Sets up the experiment logger based on the validated config."""
        if self.config.logger.type == "wandb":
            # For CI/CD, WANDB_API_KEY should be set as a secure environment variable
            return WandbLogger(
                project=self.config.project_name, 
                name=self.config.experiment_name, 
                save_dir=str(self.config.paths.log_dir)
            )
        return TensorBoardLogger(
            save_dir=str(self.config.paths.log_dir), 
            name=self.config.experiment_name
        )

    def _setup_callbacks(self) -> list[Callback]:
        """Sets up PyTorch Lightning callbacks from the validated config."""
        callbacks: list[Callback] = [RichProgressBar(), LearningRateMonitor(logging_interval="step")]
        
        # Declarative setup based on the validated Pydantic model
        if self.config.callbacks.model_checkpoint.enable:
            cfg = self.config.callbacks.model_checkpoint
            callbacks.append(ModelCheckpoint(dirpath=str(self.experiment_dir), **cfg.dict(exclude={'enable'})))
        
        if self.config.callbacks.early_stopping.enable:
            cfg = self.config.callbacks.early_stopping
            callbacks.append(EarlyStopping(**cfg.dict(exclude={'enable'})))
            
        return callbacks


class ClassificationRunner(BaseRunner):
    """Runs a classification training pipeline."""
    def _setup_data(self) -> pl.LightningDataModule:
        """Initializes the DataModule from configuration."""
        config_dict = {
            **self.config.data_loader_params,
            **self.config.data_params,
            "data_dir": self.config.paths.processed_data_dir,
        }
        dm_config = DataModuleConfig(**config_dict)
        return PlantDiseaseDataModule(config=dm_config)
        
    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule:
        """Initializes the classification model."""
        # REFACTORED: The model config (including num_classes) is now validated
        # by its own Pydantic model, decoupling it from the data module's state.
        model_pydantic_config = ModelConfig(**self.config.model_config)
        
        # CORRECTED: The manual call to data_module.setup() has been removed.
        # The Trainer is responsible for calling this hook at the correct time.
        return UnifiedModel(
            model_config=model_pydantic_config, 
            **self.config.training_params
        )


class GanRunner(BaseRunner):
    """Runs a GAN training pipeline."""
    def _setup_data(self) -> pl.LightningDataModule:
        """Initializes the DataModule for a GAN task."""
        config_dict = {
            **self.config.data_loader_params,
            **self.config.data_params,
            "data_dir": self.config.paths.processed_data_dir,
            "task": "gan",
        }
        dm_config = DataModuleConfig(**config_dict)
        return PlantDiseaseDataModule(config=dm_config)

    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule:
        """Initializes the GAN model."""
        if not GAN_AVAILABLE:
            raise ImportError("GAN dependencies are not installed. Please install with 'pip install .[gan]'")
        
        # The specific model config is parsed by its Pydantic model
        model_pydantic_config = Pix2PixConfig(**self.config.model_config)
        return Pix2PixGAN(config=model_pydantic_config)


__all__ = ["BaseRunner", "ClassificationRunner", "GanRunner", "ExperimentConfig"]