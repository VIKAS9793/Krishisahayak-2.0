# src/krishi_sahayak/pipelines/runners.py
"""Training pipeline runners for different model types."""
from __future__ import annotations
import logging
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from krishi_sahayak.config.schemas import PathsConfig, TrainingConfig
from krishi_sahayak.data.data_module import DataModuleConfig, PlantDiseaseDataModule
# REFACTORED: Corrected import path from non-existent 'architectures' to 'core'
from krishi_sahayak.models.core import UnifiedModel

logger = logging.getLogger(__name__)

class BaseRunner(ABC):
    """An abstract base class for a training task runner."""
    def __init__(self, config: TrainingConfig, paths: PathsConfig, job_name: str) -> None:
        self.config = config
        self.paths = paths
        self.job_name = job_name
        self.experiment_dir = self.paths.checkpoint_dir / self.config.experiment_name
    def run(self) -> Dict[str, Any]:
        self._save_config()
        data_module = self._setup_data()
        model = self._setup_model(data_module)
        trainer = self._setup_trainer()
        trainer.fit(model, datamodule=data_module)
        test_results = {}
        if self.config.run_test_after_fit:
            test_results_list = trainer.test(model, datamodule=data_module, ckpt_path="best")
            if test_results_list: test_results = test_results_list[0]
        return test_results
    @abstractmethod
    def _setup_data(self) -> pl.LightningDataModule: raise NotImplementedError
    @abstractmethod
    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule: raise NotImplementedError
    def _save_config(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.experiment_dir / "run_config.yaml"
        with open(config_path, 'w') as f: yaml.dump(self.config.model_dump(mode='json'), f, indent=4)
    def _setup_trainer(self) -> pl.Trainer:
        logger_instance = self._setup_logger()
        callbacks = self._setup_callbacks()
        return pl.Trainer(default_root_dir=self.experiment_dir, logger=logger_instance, callbacks=callbacks, **self.config.trainer_config)
    def _setup_logger(self) -> WandbLogger | TensorBoardLogger:
        logger_cfg = self.config.trainer_config.get('logger', {})
        if logger_cfg.get('type') == "wandb": return WandbLogger(project=self.config.project_name, name=self.config.experiment_name, save_dir=str(self.paths.log_dir))
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
    """Runs a standard classification training pipeline."""
    def _setup_data(self) -> PlantDiseaseDataModule:
        dm_config = DataModuleConfig(
            data_dir=self.paths.data_root / "processed",
            metadata_filename=self.config.data_params['metadata_filename'],
            batch_size=self.config.training_params.batch_size,
            num_workers=self.config.data_loader_params.num_workers,
            pin_memory=self.config.data_loader_params.pin_memory,
            seed=self.config.seed,
            transforms=self.config.data_params.get('transforms', {})
        )
        return PlantDiseaseDataModule(config=dm_config)
    def _setup_model(self, data_module: PlantDiseaseDataModule) -> UnifiedModel:
        data_module.prepare_data(); data_module.setup()
        def batch_processor(batch): return batch['image'], batch['target']
        return UnifiedModel(
            model_config=self.config.architecture_config, base_config=self.config.training_params,
            num_classes=data_module.num_classes, class_weights=data_module.class_weights,
            batch_processor=batch_processor
        )

class GanRunner(BaseRunner):
    """Placeholder for a GAN training pipeline runner."""
    def _setup_data(self) -> pl.LightningDataModule: raise NotImplementedError
    def _setup_model(self, data_module: pl.LightningDataModule) -> pl.LightningModule: raise NotImplementedError