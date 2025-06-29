# src/krishi_sahayak/models/__init__.py
"""
Initializes the models package, defining its public API.

This file exports the main model classes and their primary configuration schemas,
making them easily accessible to other parts of the application. It handles
optional dependencies like GANs gracefully.
"""

# --- Core Model Exports ---
# REFACTORED: Import BaseModel and its config from the 'base' sub-package.
from .base import BaseModel, BaseModelConfig
from .core import HybridModel, UnifiedModel

# --- Core Schema Exports ---
from .schemas import (
    DistillationConfig,
    FusionConfig,
    ModelArchitecture,
    ModelConfig,
    StreamConfig,
    TaskType,
)

# --- Optional GAN Exports ---
try:
    from .gan import GANConfig, EnhancedGenerator, EnhancedDiscriminator
    GAN_AVAILABLE = True
except ImportError:
    # Allows the package to work even if GAN dependencies are not installed
    GAN_AVAILABLE = False
    # Create dummy classes for type hinting if needed
    class GANConfig: pass
    class EnhancedGenerator: pass
    class EnhancedDiscriminator: pass

# --- Public API Definition ---
__all__ = [
    # From .base
    "BaseModel",
    "BaseModelConfig",
    
    # From .core
    "UnifiedModel",
    "HybridModel",
    
    # From .schemas
    "DistillationConfig",
    "FusionConfig",
    "ModelArchitecture",
    "ModelConfig",
    "StreamConfig",
    "TaskType",
]

if GAN_AVAILABLE:
    __all__.extend(["GANConfig", "EnhancedGenerator", "EnhancedDiscriminator"])