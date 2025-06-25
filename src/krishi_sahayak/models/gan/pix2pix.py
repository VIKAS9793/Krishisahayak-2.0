"""
Pix2Pix GAN using PyTorch Lightning (Refactored & Production-Ready)

This module implements the Pix2Pix architecture as a LightningModule. This
enhanced version incorporates type-safe configuration, flexible learning rate
scheduling, and logger-agnostic visual logging for robust training and monitoring.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pydantic import BaseModel, Field
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import lr_scheduler
from torchvision import transforms

from .gan import DiscriminatorConfig, EnhancedDiscriminator, EnhancedGenerator, GeneratorConfig


# =============================================================================
# PART 1: Pydantic Configuration Models (Enhanced)
# =============================================================================
class OptimizerConfig(BaseModel):
    """Configuration for GAN optimizers and schedulers."""
    learning_rate: float = Field(2e-4, gt=0)
    b1: float = Field(0.5, gt=0, lt=1)
    b2: float = Field(0.999, gt=0, lt=1)
    use_scheduler: bool = True
    scheduler_type: str = Field("plateau", pattern="^(plateau|cosine)$")
    scheduler_params: Dict[str, Any] = Field(default_factory=dict)
    scheduler_monitor: str = "val_pixel_loss"

class NormalizationConfig(BaseModel):
    """Stores dataset normalization stats for correct visual logging."""
    mean: List[float] = [0.5, 0.5, 0.5]
    std: List[float] = [0.5, 0.5, 0.5]

class DataConfig(BaseModel):
    """Defines keys and properties for data handling."""
    input_key: str = "image"
    target_key: str = "ms_image"
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)

class Pix2PixConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    l1_lambda: float = Field(100.0, ge=0)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = Field(default_factory=DiscriminatorConfig)


# =============================================================================
# PART 2: The PyTorch Lightning Module (Refactored)
# =============================================================================
class Pix2PixGAN(pl.LightningModule):
    """
    A PyTorch Lightning implementation of the Pix2Pix GAN model, designed for
    robustness, configurability, and seamless integration with the project's
    training pipeline.
    """
    def __init__(self, config: Pix2PixConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.generator = EnhancedGenerator(config.generator)
        self.discriminator = EnhancedDiscriminator(config.discriminator)
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_pixel = nn.L1Loss()

        # Denormalization transform for visualization
        norm_conf = self.config.data.normalization
        self.denormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(norm_conf.mean, norm_conf.std)],
            std=[1/s for s in norm_conf.std]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass, which is the generator's prediction."""
        return self.generator(x)

    def _generator_loss(self, real_a: torch.Tensor, real_b: torch.Tensor, fake_b: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        pred_fake = self.discriminator(real_a, fake_b)
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_pixel = self.criterion_pixel(fake_b, real_b)
        total_loss_g = loss_gan + self.config.l1_lambda * loss_pixel
        return total_loss_g, loss_gan, loss_pixel

    def _discriminator_loss(self, real_a: torch.Tensor, real_b: torch.Tensor, fake_b: torch.Tensor) -> torch.Tensor:
        pred_real = self.discriminator(real_a, real_b)
        loss_real = self.criterion_gan(pred_real, torch.ones_like(pred_real, device=self.device))
        pred_fake = self.discriminator(real_a, fake_b.detach())
        loss_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        return (loss_real + loss_fake) * 0.5

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        real_a = batch[self.config.data.input_key]
        real_b = batch[self.config.data.target_key]
        fake_b = self(real_a)

        if optimizer_idx == 0: # Train Generator
            loss_g, loss_gan, loss_pixel = self._generator_loss(real_a, real_b, fake_b)
            self.log_dict({'g_loss': loss_g, 'g_gan_loss': loss_gan, 'g_pixel_loss': loss_pixel}, prog_bar=True)
            return loss_g

        if optimizer_idx == 1: # Train Discriminator
            loss_d = self._discriminator_loss(real_a, real_b, fake_b)
            self.log('d_loss', loss_d, prog_bar=True)
            return loss_d
        return None

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        real_a = batch[self.config.data.input_key]
        real_b = batch[self.config.data.target_key]
        fake_b = self(real_a)
        
        self.log('val_pixel_loss', self.criterion_pixel(fake_b, real_b), on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0 and self.logger: self._log_image_grid(real_a, real_b, fake_b)

    def _log_image_grid(self, real_a: torch.Tensor, real_b: torch.Tensor, fake_b: torch.Tensor) -> None:
        n_images = min(4, real_a.size(0))
        real_b_display = real_b[:n_images].expand(-1, 3, -1, -1) if real_b.shape[1] == 1 else real_b[:n_images]
        fake_b_display = fake_b[:n_images].expand(-1, 3, -1, -1) if fake_b.shape[1] == 1 else fake_b[:n_images]
        grid = torchvision.utils.make_grid(torch.cat([
            self.denormalize(real_a[:n_images]), self.denormalize(real_b_display), self.denormalize(fake_b_display)
        ]), nrow=n_images)
        caption = "Top: Input (RGB), Middle: Target (Ground Truth), Bottom: Generated"
        
        if isinstance(self.logger, WandbLogger) and hasattr(self.logger.experiment, "Image"):
            self.logger.experiment.log({"val_generated_images": self.logger.experiment.Image(grid, caption=caption)})
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image("val_generated_images", grid, self.current_epoch)

    def configure_optimizers(self) -> List[Dict[str, Any]]:
        opt_conf = self.config.optimizer
        lr, b1, b2 = opt_conf.learning_rate, opt_conf.b1, opt_conf.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        if not opt_conf.use_scheduler:
            return [{"optimizer": opt_g}, {"optimizer": opt_d}]
        
        # Create schedulers based on configuration
        schedulers = []
        for opt in [opt_g, opt_d]:
            if opt_conf.scheduler_type == "plateau":
                scheduler = lr_scheduler.ReduceLROnPlateau(opt, mode='min', **opt_conf.scheduler_params)
            elif opt_conf.scheduler_type == "cosine":
                if "T_max" not in opt_conf.scheduler_params:
                    raise ValueError("`T_max` must be provided in scheduler_params for CosineAnnealingLR.")
                scheduler = lr_scheduler.CosineAnnealingLR(opt, **opt_conf.scheduler_params)
            else:
                raise ValueError(f"Unsupported scheduler type: {opt_conf.scheduler_type}")
            
            schedulers.append({"scheduler": scheduler, "monitor": opt_conf.scheduler_monitor})

        return [{"optimizer": opt_g, "lr_scheduler": schedulers[0]}, {"optimizer": opt_d, "lr_scheduler": schedulers[1]}]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        input_key = self.config.data.input_key
        real_a = batch[input_key] if isinstance(batch, dict) else batch
        return self(real_a)