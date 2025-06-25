"""
Utility functions for the KrishiSahayak project.

This package provides common, project-wide helper functions for tasks
such as logging configuration and random seed management.
"""

from .logger import setup_logging
from .seed import set_seed

# Define the public API for the 'utils' package.
# The 'get_logger' function was intentionally removed in favor of the standard
# `logging.getLogger(__name__)` pattern.
__all__ = [
    "setup_logging",
    "set_seed",
]
