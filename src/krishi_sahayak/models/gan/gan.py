"""
KrishiSahayak - Advanced GAN Models for Image-to-Image Translation (Refactored)

This module provides enhanced Generator and Discriminator architectures incorporating
modern best practices such as Spectral Normalization and Self-Attention.
"""
from __future__ import annotations
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch.nn.utils import spectral_norm

# =============================================================================
# PART 1: Pydantic Configuration Models
# =============================================================================

class GeneratorConfig(BaseModel):
    in_channels: int = Field(3, gt=0)
    out_channels: int = Field(1, gt=0)
    features: int = Field(64, gt=0)
    leaky_relu_slope: float = Field(0.2, description="Negative slope for LeakyReLU.")
    use_attention: bool = True
    attention_projection_ratio: int = Field(8, gt=0)
    use_spectral_norm: bool = False
    dropout_rate: float = Field(0.5, ge=0.0, le=1.0, description="Dropout rate for deeper generator layers.")

class DiscriminatorConfig(BaseModel):
    in_channels: int = Field(4, gt=0, description="Sum of generator input and output channels.")
    features: int = Field(64, gt=0)
    leaky_relu_slope: float = Field(0.2, description="Negative slope for LeakyReLU.")
    use_spectral_norm: bool = True
    use_attention: bool = False
    attention_projection_ratio: int = Field(8, gt=0)

class GANConfig(BaseModel):
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = Field(default_factory=DiscriminatorConfig)


# =============================================================================
# PART 2: Core Model Components
# =============================================================================

def weights_init_xavier(m: nn.Module) -> None:
    """Applies Xavier normal initialization to Conv and ConvTranspose layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)

class SelfAttention(nn.Module):
    """Self-attention mechanism (non-local block) for convolutional layers."""
    # ... (Implementation is unchanged as it was already excellent) ...

class DownBlock(nn.Module):
    """Generalized downsampling block used by both Generator and Discriminator."""
    def __init__(self, in_c: int, out_c: int, use_norm: bool = True, use_sn: bool = False, leaky_relu_slope: float = 0.2) -> None:
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        if use_sn:
            conv = spectral_norm(conv)
        
        layers: list[nn.Module] = [conv]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class UpBlock(nn.Module):
    """Upsampling block for the U-Net Generator with skip connections."""
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.0, use_sn: bool = False) -> None:
        super().__init__()
        conv_t = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        if use_sn:
            conv_t = spectral_norm(conv_t)
        
        layers: list[nn.Module] = [conv_t, nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat([x, skip_input], dim=1)


# =============================================================================
# PART 3: Top-Level Model Architectures
# =============================================================================

class EnhancedGenerator(nn.Module):
    """Enhanced U-Net Generator with self-attention and spectral normalization."""
    # ... (Forward pass is unchanged, __init__ is updated to use DownBlock and config) ...
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        f, dr, sn = config.features, config.dropout_rate, config.use_spectral_norm
        slope = config.leaky_relu_slope

        self.down1 = DownBlock(config.in_channels, f, use_norm=False, use_sn=sn, leaky_relu_slope=slope)
        self.down2 = DownBlock(f, f * 2, use_sn=sn, leaky_relu_slope=slope)
        self.down3 = DownBlock(f * 2, f * 4, use_sn=sn, leaky_relu_slope=slope)
        # ... and so on for other down blocks
        # ... up blocks and final layer remain structurally the same ...

# NOTE: For brevity, the full generator and discriminator __init__ are not repeated,
# but they would be updated to use the new DownBlock and configurable slope.
# The forward passes remain identical.

class EnhancedDiscriminator(nn.Module):
    """Enhanced PatchGAN Discriminator using the shared DownBlock component."""
    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        f, sn = config.features, config.use_spectral_norm
        slope = config.leaky_relu_slope
        
        layers: list[nn.Module] = [
            DownBlock(config.in_channels, f, use_norm=False, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f, f * 2, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f * 2, f * 4, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f * 4, f * 8, use_sn=sn, leaky_relu_slope=slope),
        ]
        if config.use_attention:
            layers.append(SelfAttention(f * 8, config.attention_projection_ratio))
        
        # Final convolution to produce a 1-channel patch output
        layers.append(nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([img_A, img_B], dim=1))

# =============================================================================
# PART 4: Factory Function
# =============================================================================

def create_enhanced_gan_models(config: GANConfig) -> tuple[EnhancedGenerator, EnhancedDiscriminator]:
    """Factory function to create and initialize GAN models from a config object."""
    generator = EnhancedGenerator(config.generator)
    discriminator = EnhancedDiscriminator(config.discriminator)
    generator.apply(weights_init_xavier)
    discriminator.apply(weights_init_xavier)
    return generator, discriminator