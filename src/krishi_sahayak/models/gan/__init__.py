# src/krishi_sahayak/models/gan/__init__.py
"""
Initializes the GAN package, exporting key model and config components.
"""
from .gan import (
    GANConfig,
    GeneratorConfig,
    DiscriminatorConfig,
    EnhancedGenerator,
    EnhancedDiscriminator,
    SelfAttention, # <-- Add this line
)
from .pix2pix import Pix2PixGAN, Pix2PixConfig

__all__ = [
    "GANConfig",
    "GeneratorConfig",
    "DiscriminatorConfig",
    "EnhancedGenerator",
    "EnhancedDiscriminator",
    "SelfAttention", # <-- And add it here
    "Pix2PixGAN",
    "Pix2PixConfig",
]
