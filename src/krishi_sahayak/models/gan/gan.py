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
from pydantic import BaseModel, Field
from torch.nn.utils import spectral_norm

# --- Pydantic Configuration Models ---

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


# --- Core Model Components ---

def weights_init_xavier(m: nn.Module) -> None:
    """Applies Xavier normal initialization to Conv and ConvTranspose layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight)

class SelfAttention(nn.Module):
    """Self-attention mechanism (non-local block) for convolutional layers."""
    def __init__(self, in_channels: int, projection_ratio: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.query = spectral_norm(nn.Conv1d(in_channels, in_channels // projection_ratio, 1))
        self.key = spectral_norm(nn.Conv1d(in_channels, in_channels // projection_ratio, 1))
        self.value = spectral_norm(nn.Conv1d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flatten = x.view(B, C, H * W)
        q = self.query(x_flatten).permute(0, 2, 1)
        k = self.key(x_flatten)
        v = self.value(x_flatten)
        
        attention_matrix = torch.bmm(q, k)
        attention = F.softmax(attention_matrix, dim=-1)
        
        o = torch.bmm(v, attention.permute(0, 2, 1))
        o = o.view(B, C, H, W)
        
        return self.gamma * o + x


class DownBlock(nn.Module):
    """Generalized downsampling block used by both Generator and Discriminator."""
    def __init__(self, in_c: int, out_c: int, use_norm: bool = True, use_sn: bool = False, leaky_relu_slope: float = 0.2) -> None:
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_norm)
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


# --- Top-Level Model Architectures ---

class EnhancedGenerator(nn.Module):
    """Enhanced U-Net Generator with optional self-attention and spectral normalization."""
    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        f, dr, sn, slope = config.features, config.dropout_rate, config.use_spectral_norm, config.leaky_relu_slope

        self.down1 = DownBlock(config.in_channels, f, use_norm=False, use_sn=sn, leaky_relu_slope=slope)
        self.down2 = DownBlock(f, f * 2, use_sn=sn, leaky_relu_slope=slope)
        self.down3 = DownBlock(f * 2, f * 4, use_sn=sn, leaky_relu_slope=slope)
        self.down4 = DownBlock(f * 4, f * 8, use_sn=sn, leaky_relu_slope=slope)
        self.down5 = DownBlock(f * 8, f * 8, use_sn=sn, leaky_relu_slope=slope)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(f * 8, f * 8, 4, 2, 1), nn.ReLU(True))

        self.up1 = UpBlock(f * 8, f * 8, dropout=dr, use_sn=sn)
        self.up2 = UpBlock(f * 16, f * 8, dropout=dr, use_sn=sn)
        
        self.attention = SelfAttention(f * 8, config.attention_projection_ratio) if config.use_attention else nn.Identity()

        self.up3 = UpBlock(f * 16, f * 4, use_sn=sn)
        self.up4 = UpBlock(f * 8, f * 2, use_sn=sn)
        self.up5 = UpBlock(f * 4, f, use_sn=sn)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(f * 2, config.out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        bottleneck = self.bottleneck(d5)
        u1 = self.up1(bottleneck, d5)
        u2 = self.up2(u1, d4)
        u2_att = self.attention(u2)
        u3 = self.up3(u2_att, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final_up(u5)


class EnhancedDiscriminator(nn.Module):
    """Enhanced PatchGAN Discriminator using the shared DownBlock component."""
    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.config = config
        f, sn, slope = config.features, config.use_spectral_norm, config.leaky_relu_slope
        
        layers: list[nn.Module] = [
            DownBlock(config.in_channels, f, use_norm=False, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f, f * 2, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f * 2, f * 4, use_sn=sn, leaky_relu_slope=slope),
            DownBlock(f * 4, f * 8, use_sn=sn, leaky_relu_slope=slope),
        ]
        if config.use_attention:
            layers.append(SelfAttention(f * 8, config.attention_projection_ratio))
        
        layers.append(nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([img_A, img_B], dim=1))


# --- Factory Function ---
def create_enhanced_gan_models(config: GANConfig) -> tuple[EnhancedGenerator, EnhancedDiscriminator]:
    """Factory function to create and initialize GAN models from a config object."""
    generator = EnhancedGenerator(config.generator)
    discriminator = EnhancedDiscriminator(config.discriminator)
    generator.apply(weights_init_xavier)
    discriminator.apply(weights_init_xavier)
    return generator, discriminator