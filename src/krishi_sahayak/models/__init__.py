"""
Model implementations for plant disease classification.

This package contains the model architectures used in the KrishiSahayak project,
including the base model, core unified model, and advanced utility trainers.
"""
# Core model components
from .base import BaseModel
from .core import (
    FusionConfig,
    HybridModel,
    ModelConfig,
    StreamConfig,
    UnifiedModel,
)

# Advanced Training Modules / Utilities
from .distillation import DistillationLightningModule, KnowledgeDistillationLoss
from .utils import ConfidenceThreshold, FusionValidator

# GAN components are an optional dependency
try:
    from .gan import EnhancedDiscriminator, EnhancedGenerator, Pix2PixGAN
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False
    # Create dummy classes for type checking if GAN dependencies are not installed
    class EnhancedDiscriminator: pass
    class EnhancedGenerator: pass
    class Pix2PixGAN: pass

# Define the public API of the 'models' package.
__all__ = [
    # Base model
    'BaseModel',
    
    # Core models
    'UnifiedModel',
    'HybridModel',
    
    # Configuration
    'ModelConfig',
    'StreamConfig',
    'FusionConfig',
    
    # Model utilities & trainers
    'KnowledgeDistillationLoss',
    'DistillationLightningModule',
    'ConfidenceThreshold',
    'FusionValidator',
]

# Conditionally extend the public API with GAN components if they are available
if GAN_AVAILABLE:
    __all__.extend([
        'EnhancedDiscriminator',
        'EnhancedGenerator',
        'Pix2PixGAN',
    ])