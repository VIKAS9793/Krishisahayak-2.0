# src/krishisahayak/__init__.py
"""
KrishiSahayak: AI-Powered Crop Health Assistant

A comprehensive AI solution for plant disease classification and agricultural advisory.
NOTE: This package uses lazy loading to improve import performance.
"""

from importlib import metadata
from typing import Any

# --- Author and Contact Information ---
__author__ = "Vikas Sahani"
__email__ = "vikassahani17@gmail.com"
__github__ = "https://github.com/VIKAS9793/Krishisahayak-2.0"

# --- Versioning ---
# REFACTORED: The version is now dynamically read from the installed package's
# metadata. This ensures a single source of truth from pyproject.toml.
try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    # This fallback is useful for development environments where the package
    # might not be formally installed yet.
    __version__ = "0.0.0-dev"

# --- Data structure for lazy loading ---
# This dictionary defines the public API and where to find it.
_LAZY_IMPORTS = {
    "PlantDiseaseDataModule": ("krishi_sahayak.data.data_module", "PlantDiseaseDataModule"),
    "UnifiedPlantDataset": ("krishi_sahayak.data.dataset", "UnifiedPlantDataset"),
    "PlantDiseaseDataset": ("krishi_sahayak.data.dataset", "UnifiedPlantDataset"), # Alias
    "BaseModel": ("krishi_sahayak.models.base", "BaseModel"),
    "HybridModel": ("krishi_sahayak.models.core.hybrid_model", "HybridModel"),
    "UnifiedModel": ("krishi_sahayak.models.core.unified_model", "UnifiedModel"),
    "PlantDiseaseModel": ("krishi_sahayak.models.core.unified_model", "UnifiedModel"), # Alias
    "setup_logging": ("krishi_sahayak.utils.logger", "setup_logging"),
    "set_seed": ("krishi_sahayak.utils.seed", "set_seed"),
}

def __getattr__(name: str) -> Any:
    """
    Implements lazy loading for the package's public API using PEP 562.
    This avoids importing heavy dependencies like PyTorch until they are needed.
    """
    if name in _LAZY_IMPORTS:
        module_path, attribute_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attribute_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Public API Export List ---
# This list defines what `from krishisahayak import *` will expose.
__all__ = [
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__github__",
    # Data Components
    'PlantDiseaseDataModule',
    'UnifiedPlantDataset',
    'PlantDiseaseDataset',  # Alias
    # Model Components
    'BaseModel',
    'UnifiedModel',
    'HybridModel',
    'PlantDiseaseModel',  # Alias
    # Utility Functions
    'setup_logging',
    'set_seed',
]