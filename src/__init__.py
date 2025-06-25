"""
KrishiSahayak: AI-Powered Crop Health Assistant

A comprehensive AI solution for plant disease classification and agricultural advisory.

NOTE: This package provides logging utilities, but it does not automatically
configure logging. The end-user application is responsible for calling
`setup_logging()` once at its entry point to initialize the logging system.
"""

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Vikas Sahani"
__email__ = "vikassahani17@gmail.com"
__github__ = "https://github.com/VIKAS9793/KrishiSahayak"

# --- Data structure for lazy loading ---
# Maps a public name to the module path and the attribute name within that module.
# This approach handles both direct imports and aliases cleanly.
_LAZY_IMPORTS = {
    "PlantDiseaseDataModule": ("krishisahayak.data.data_module", "PlantDiseaseDataModule"),
    "UnifiedPlantDataset": ("krishisahayak.data.dataset", "UnifiedPlantDataset"),
    "PlantDiseaseDataset": ("krishisahayak.data.dataset", "UnifiedPlantDataset"), # Alias
    "BaseModel": ("krishisahayak.models", "BaseModel"),
    "HybridModel": ("krishisahayak.models", "HybridModel"),
    "UnifiedModel": ("krishisahayak.models", "UnifiedModel"),
    "PlantDiseaseModel": ("krishisahayak.models", "UnifiedModel"), # Alias
    "setup_logging": ("krishisahayak.utils", "setup_logging"),
    "set_seed": ("krishisahayak.utils", "set_seed"),
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
# Defines the names that are part of the public API for tools like linters.
__all__ = [
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
