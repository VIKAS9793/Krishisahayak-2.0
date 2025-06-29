# src/krishi_sahayak/gan/pix2pix.py
"""Pix2Pix GAN using PyTorch Lightning (Refactored & Production-Ready)"""
from __future__ import annotations
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pydantic import BaseModel, Field
from torch.optim import lr_scheduler

from krishi_sahayak.models.base import BaseModelConfig
from .gan import GANConfig, create_enhanced_gan_models

class Pix2PixConfig(BaseModel):
    data: Dict[str, Any] = Field(default_factory=dict)
    optimizer: BaseModelConfig = Field(default_factory=BaseModelConfig)
    l1_lambda: float = Field(100.0, ge=0)
    gan_architectures: GANConfig = Field(default_factory=GANConfig)

class Pix2PixGAN(pl.LightningModule):
    def __init__(self, config: Pix2PixConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.generator, self.discriminator = create_enhanced_gan_models(self.config.gan_architectures)
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_pixel = nn.L1Loss()
        norm_conf = self.config.data
        self.denormalize = torchvision.transforms.Normalize(mean=[-m/s for m, s in zip(norm_conf.get('mean', [0.5]*3), norm_conf.get('std', [0.5]*3))], std=[1/s for s in norm_conf.get('std', [0.5]*3)])
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.generator(x)
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int):
        real_a = batch[self.config.data.get('input_key', 'image')]
        real_b = batch[self.config.data.get('target_key', 'ms_image')]
        fake_b = self(real_a)
        if optimizer_idx == 0:
            loss_g = self._generator_loss(real_a, real_b, fake_b)
            self.log_dict({'g_loss': loss_g}, prog_bar=True, on_step=True, on_epoch=False)
            return loss_g
        if optimizer_idx == 1:
            loss_d = self._discriminator_loss(real_a, real_b, fake_b)
            self.log('d_loss', loss_d, prog_bar=True, on_step=True, on_epoch=False)
            return loss_d
    def _generator_loss(self, real_a, real_b, fake_b):
        pred_fake = self.discriminator(real_a, fake_b)
        loss_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_pixel = self.criterion_pixel(fake_b, real_b)
        return loss_gan + self.config.l1_lambda * loss_pixel
    def _discriminator_loss(self, real_a, real_b, fake_b):
        loss_real = self.criterion_gan(self.discriminator(real_a, real_b), torch.ones_like(self.discriminator(real_a, real_b), device=self.device))
        loss_fake = self.criterion_gan(self.discriminator(real_a, fake_b.detach()), torch.zeros_like(self.discriminator(real_a, fake_b.detach()), device=self.device))
        return (loss_real + loss_fake) * 0.5
    def configure_optimizers(self) -> List[Dict[str, Any]]:
        opt_conf = self.config.optimizer
        lr = opt_conf.learning_rate
        opt_g = torch.optim.AdamW(self.generator.parameters(), lr=lr, weight_decay=opt_conf.weight_decay)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, weight_decay=opt_conf.weight_decay)
        if not opt_conf.use_scheduler: return [{"optimizer": opt_g}, {"optimizer": opt_d}]
        scheduler_g = lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.trainer.max_epochs, eta_min=1e-6)
        scheduler_d = lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return [{"optimizer": opt_g, "lr_scheduler": scheduler_g}, {"optimizer": opt_d, "lr_scheduler": scheduler_d}]