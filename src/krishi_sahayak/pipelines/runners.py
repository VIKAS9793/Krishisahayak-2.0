# src/krishi_sahayak/pipelines/runners.py
"""
Training pipeline runners for different model types. (Refactored)
This module uses a single, canonical configuration schema and correctly
instantiates models according to their defined contracts.
"""
from __future__ import annotations
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from krishi_sahayak.config.schemas import PathsConfig, TrainingConfig
from krishi_sahayak.data.data_module import DataModuleConfig, PlantDiseaseDataModule
from krishi_sahayak.models.architectures import UnifiedModel
from krishi_sahayak.models.schemas import ModelArchitecture # For teacher model architecture

logger = logging.getLogger(__name__)

class BaseRunner(ABC):
    """An abstract base class for a training task runner."""
    def __init__(self, config: TrainingConfig, paths: PathsConfig, job_name: str) -> None:
        self.config = config
        self.paths = paths
        self.job_name = job_name
        self.experiment_dir = self.paths.checkpoint_dir / self.config.experiment_name

    def run(self) -> Dict[str, Any]:
        """Executes the full training pipeline and returns test results."""
        logger.info(f"Starting training run for job: {self.job_name}")
        self._save_config()
        data_module = self._setup_data()
        model = self._setup_model(data_module)
        trainer = self._setup_trainer()
        
        logger.info("Starting model training...")
        trainer.fit(model, datamodule=data_module)
        
        test_results = {}
        if self.config.run_test_after_fit:
            logger.info("Training finished. Starting testing with the best checkpoint...")
            test_results_list = trainer.test(model, datamodule=data_module, ckpt_path="best")
            if test_results_list:
                test_results = test_results_list[0]
                logger.info(f"Test results: {test_results}")

        logger.info(f"Run '{self.config.experiment_name}' completed successfully.")
        return test_results

    @abstractmethod
    def _setup_data(self) -> pl.LightningDataModule:
        raise NotImplementedError

    @abstractmethod
    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule:
        raise NotImplementedError

    def _save_config(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.experiment_dir / "run_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config.model_dump(mode='json'), f, indent=4)
        logger.info(f"Run configuration saved to {config_path}")

    def _setup_trainer(self) -> pl.Trainer:
        logger_instance = self._setup_logger()
        callbacks = self._setup_callbacks()
        return pl.Trainer(
            default_root_dir=self.experiment_dir,
            logger=logger_instance,
            callbacks=callbacks,
            **self.config.trainer_config
        )

    def _setup_logger(self) -> WandbLogger | TensorBoardLogger:
        logger_cfg = self.config.trainer_config.get('logger', {})
        if logger_cfg.get('type') == "wandb":
            return WandbLogger(project=self.config.project_name, name=self.config.experiment_name, save_dir=str(self.paths.log_dir))
        return TensorBoardLogger(save_dir=str(self.paths.log_dir), name=self.config.experiment_name)

    def _setup_callbacks(self) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = [RichProgressBar(), LearningRateMonitor(logging_interval="step")]
        if self.config.callbacks.model_checkpoint.get('enable', True):
            params = self.config.callbacks.model_checkpoint.copy(); params.pop('enable', None)
            callbacks.append(ModelCheckpoint(dirpath=str(self.experiment_dir), **params))
        if self.config.callbacks.early_stopping.get('enable', True):
            params = self.config.callbacks.early_stopping.copy(); params.pop('enable', None)
            callbacks.append(EarlyStopping(**params))
        return callbacks

class ClassificationRunner(BaseRunner):
    """Runner for classification tasks with optional knowledge distillation."""
    
    def __init__(self, config: TrainingConfig, paths: PathsConfig, job_name: str):
        super().__init__(config, paths, job_name)
        self.teacher_model = None
        
    def _load_teacher_model(self, num_classes: int) -> UnifiedModel | None:
        """Loads the teacher model for knowledge distillation if configured."""
        distill_config = self.config.model_dump().get("distillation", {})
        if not distill_config or not distill_config.get("enabled"):
            return None
        
        teacher_checkpoint = distill_config.get("teacher_checkpoint")
        if not teacher_checkpoint:
            raise ValueError("Distillation is enabled, but 'teacher_checkpoint' is not specified.")

        logger.info(f"Loading teacher model from {teacher_checkpoint}")
        
        teacher_model_config = self.config.model_config.copy(deep=True)
        # The teacher's architecture is now configurable.
        teacher_arch = distill_config.get("teacher_architecture")
        if not teacher_arch:
            raise ValueError("Distillation is enabled, but 'teacher_architecture' is not specified.")
        teacher_model_config.backbone_name = ModelArchitecture(teacher_arch)
        
        teacher = UnifiedModel(
            model_config=teacher_model_config,
            base_config=self.config.training_params,
            num_classes=num_classes,
        )
        
        checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
        teacher.load_state_dict(checkpoint.get('state_dict', checkpoint))
        teacher.eval()
        return teacher
    
    def _setup_data(self) -> PlantDiseaseDataModule:
        """Sets up the data module for the training job."""
        dm_pydantic_config = DataModuleConfig(
            data_dir=self.paths.data_root / "processed",
            metadata_filename=self.config.data_params['metadata_filename'],
            batch_size=self.config.data_params.get('batch_size', 32),
            num_workers=self.config.data_loader_params.num_workers,
            pin_memory=self.config.data_loader_params.pin_memory,
            seed=self.config.seed, # Correctly access the top-level seed
            transforms=self.config.data_params.get('transforms', {})
        )
        return PlantDiseaseDataModule(config=dm_pydantic_config)

    def _setup_model(self, data_module: PlantDiseaseDataModule) -> UnifiedModel:
        """Initializes the classification model with optional teacher for distillation."""
        data_module.prepare_data()
        data_module.setup()
        
        self.teacher_model = self._load_teacher_model(data_module.num_classes)
        
        def classification_batch_processor(batch):
            return batch['image'], batch['target']

        distill_cfg = self.config.model_dump().get("distillation", {})

        model = UnifiedModel(
            model_config=self.config.model_config,
            base_config=self.config.training_params,
            num_classes=data_module.num_classes,
            class_weights=data_module.class_weights,
            batch_processor=classification_batch_processor,
            teacher_model=self.teacher_model,
            distillation_config=distill_cfg if distill_cfg.get("enabled") else None
        )
        return model

class GanRunner(BaseRunner):
    """Placeholder for a GAN training pipeline runner."""
    def _setup_data(self) -> pl.LightningDataModule:
        logger.error("GanRunner _setup_data is not yet implemented.")
        raise NotImplementedError("GAN data setup is not implemented.")

    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule:
        logger.error("GanRunner _setup_model is not yet implemented.")
        raise NotImplementedError("GAN model setup is not implemented.")