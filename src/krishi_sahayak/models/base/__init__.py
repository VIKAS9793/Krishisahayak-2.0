# src/krishi_sahayak/models/base/__init__.py
"""
Initializes the base model package, exporting key components.

This module exports the single, canonical BaseModel and its configuration BaseModelConfig
which serve as the foundation for all models in the system.
"""

from .base import BaseModel, BaseModelConfig

__all__ = [
    "BaseModel",
    "BaseModelConfig",
]
