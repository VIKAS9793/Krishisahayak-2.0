# src/krishisahayak/__init__.py
"""KrishiSahayak: AI-Powered Crop Health Assistant."""

from importlib import metadata

# --- Author and Contact Information ---
__author__ = "Vikas Sahani"
__email__ = "vikassahani17@gmail.com"
__github__ = "https://github.com/VIKAS9793/Krishisahayak-2.0"

# --- Versioning ---
# REFACTORED: The version is now dynamically read from the installed package's
# metadata. This ensures a single source of truth from pyproject.toml and
# prevents version mismatches.
try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    # This fallback is useful for development environments where the package
    # might not be formally installed yet.
    __version__ = "0.0.0-dev"

# Defines the top-level public API of the package.
# Other components should be imported from their respective sub-packages.
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__github__",
]