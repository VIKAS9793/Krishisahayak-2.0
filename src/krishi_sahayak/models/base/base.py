# src/krishi_sahayak/models/base.py
"""
KrishiSahayak - Abstract Base Model (Refactored & Production-Ready)

This module provides an enhanced abstract base class for all models,
incorporating MAANG-level best practices for flexibility and robustness.
"""
from __future__ import annotations
import abc
from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field
from torch import nn, optim
from torch.optim import lr_scheduler

# Type alias for a function that unpacks a dataloader batch into (inputs, targets)
BatchProcessorCallable = Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]

# --- Pydantic Configuration Schemas ---
class BaseModelConfig(PydanticBaseModel):
    """Configuration schema for the BaseModel's optimizer and scheduler."""
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    optimizer: str = Field("AdamW", pattern="^(AdamW|Adam|SGD)$")
    use_scheduler: bool = True
    scheduler_type: str = Field("cosine", pattern="^(cosine|step|plateau)$")
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)
    scheduler_monitor: str = "val/loss"


class BaseModel(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    An abstract base class for all models in the project. It handles metrics,
    optimization, and the training/validation/test steps.
    """
    def __init__(
        self,
        model: nn.Module, # The actual model is now injected
        num_classes: int,
        config: BaseModelConfig,
        batch_processor: BatchProcessorCallable,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # Use save_hyperparameters to automatically log configs and allow reloading.
        # Ignore complex objects that are not simple hyperparameters.
        self.save_hyperparameters(config.model_dump(), ignore=['model', 'batch_processor', 'class_weights'])

        self.model = model
        self.config = config
        self.batch_processor = batch_processor
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Setup metrics for each stage
        common_metrics = torchmetrics.MetricCollection({
            'acc': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro'),
        })
        self.train_metrics = common_metrics.clone(prefix='train/')
        self.val_metrics = common_metrics.clone(prefix='val/')
        self.test_metrics = common_metrics.clone(prefix='test/')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass by invoking the injected model."""
        return self.model(x)

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        """Performs a shared step for training, validation, and testing."""
        # Use the injected batch_processor to get inputs and targets
        x, y = self.batch_processor(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Update and log metrics
        metrics_collection = getattr(self, f"{stage}_metrics")
        metrics_collection.update(logits.softmax(dim=-1), y)

        self.log(f'{stage}/loss', loss, on_step=(stage == 'train'), on_epoch=True, prog_bar=True)
        self.log_dict(metrics_collection, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, 'val')

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, 'test')

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Performs a prediction step, returning softmax probabilities.
        This step is simplified to not rely on a batch_processor that yields targets.
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        elif isinstance(batch, dict):
            # Assumes a common key like 'image' or 'pixel_values' if the batch is a dict
            x = batch.get('image', next(iter(batch.values())))
        else:
            x = batch
        
        if not isinstance(x, torch.Tensor):
             raise TypeError(f"Input for prediction must be a torch.Tensor, but got {type(x)}")

        return self(x).softmax(dim=-1)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and an optional learning rate scheduler."""
        cfg = self.config
        
        optimizer_map = {
            "AdamW": optim.AdamW,
            "Adam": optim.Adam,
            "SGD": optim.SGD,
        }
        optimizer_class = optimizer_map.get(cfg.optimizer)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

        optimizer_params = {'lr': cfg.learning_rate, 'weight_decay': cfg.weight_decay}
        if cfg.optimizer == "SGD":
            optimizer_params['momentum'] = 0.9 # Add default momentum for SGD

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        if not cfg.use_scheduler:
            return {"optimizer": optimizer}

        scheduler_map = {
            "cosine": lr_scheduler.CosineAnnealingLR,
            "step": lr_scheduler.StepLR,
            "plateau": lr_scheduler.ReduceLROnPlateau,
        }
        scheduler_class = scheduler_map.get(cfg.scheduler_type)
        if scheduler_class is None:
             raise ValueError(f"Unsupported scheduler: {cfg.scheduler_type}")
        
        # Ensure required parameters exist for specific schedulers
        if cfg.scheduler_type == "cosine" and "T_max" not in cfg.scheduler_params:
            raise ValueError("`T_max` must be provided in scheduler_params for CosineAnnealingLR.")

        scheduler = scheduler_class(optimizer, **cfg.scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": cfg.scheduler_monitor, "interval": "epoch"},
        }