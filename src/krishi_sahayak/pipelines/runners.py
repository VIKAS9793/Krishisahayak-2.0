"""
Training pipeline runners for different model types. (Refactored)
This module uses a single, canonical configuration schema and correctly
instantiates models according to their defined contracts.
"""
from __future__ import annotations
import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# REFACTORED: Import the canonical schemas, not local ones.
from krishisahayak.config.schemas import TrainingConfig
from krishisahayak.data.data_module import DataModuleConfig, PlantDiseaseDataModule
from krishisahayak.models import BaseModelConfig, UnifiedModel, ModelConfig

logger = logging.getLogger(__name__)

# --- Runner Implementations ---
class BaseRunner(ABC):
    """An abstract base class for a training task runner."""
    def __init__(self, config: TrainingConfig, paths: Any, job_name: str) -> None:
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
            test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")[0]
            logger.info(f"Test results: {test_results}")

        logger.info(f"Run '{self.config.experiment_name}' completed successfully.")
        return test_results

    # ... (other methods like _save_config, _setup_logger are similar but use self.config from TrainingConfig) ...

class ClassificationRunner(BaseRunner):
    """Runs a classification training pipeline."""
    def _setup_data(self) -> PlantDiseaseDataModule:
        dm_pydantic_config = DataModuleConfig(
            data_dir=self.paths.data_root / "processed",
            metadata_filename=self.config.data_params['metadata_filename'],
            batch_size=self.config.training_params.batch_size,
            num_workers=self.config.data_loader_params.num_workers,
            pin_memory=self.config.data_loader_params.pin_memory,
            seed=42, # Assuming global seed
            transforms=self.config.data_params.get('transforms', {})
        )
        return PlantDiseaseDataModule(config=dm_pydantic_config)
        
    def _setup_model(self, data_module: PlantDiseaseDataModule) -> UnifiedModel:
        """Initializes the classification model correctly."""
        data_module.prepare_data()
        data_module.setup()
        
        model_pydantic_config = ModelConfig(**self.config.model_config)
        
        # REFACTORED: Correctly create the BaseModelConfig expected by the BaseModel parent class.
        base_model_config = BaseModelConfig(**self.config.training_params.model_dump())
        
        # REFACTORED: Pass the required configs and dependencies to the UnifiedModel.
        return UnifiedModel(
            model_config=model_pydantic_config,
            base_config=base_model_config,
            num_classes=data_module.num_classes,
            class_weights=data_module.class_weights,
            # Assuming a standard batch processor for this model type
            batch_processor=lambda batch: (batch['image'], batch['target'])
        )