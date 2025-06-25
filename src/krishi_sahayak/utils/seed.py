"""
Utility function for setting random seeds for reproducibility in ML experiments.

This module provides a single, comprehensive function to seed all major libraries
(random, numpy, torch) and configure PyTorch's CUDA backend for deterministic
behavior.
"""

import logging
import os
import random

import numpy as np
import torch

# It is a best practice for any module to get its own logger instance.
# This avoids interfering with the root logger configuration set by the main application.
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_ops: bool = True) -> None:
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.

    This function seeds `random`, `os.environ['PYTHONHASHSEED']`, `numpy`,
    and `torch` (for CPU and CUDA). It can also configure PyTorch to use
    deterministic algorithms, which is crucial for reproducible results but
    can have a minor performance impact.

    Args:
        seed (int): The integer value to use as the random seed.
        deterministic_ops (bool): If True, configures PyTorch to use deterministic
                                  CUDA algorithms. Set to False if you need
                                  maximal performance and can tolerate slight
                                  non-determinism.
    """
    try:
        # Standard library
        random.seed(seed)
        
        # Set PYTHONHASHSEED environment variable
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        
        if deterministic_ops:
            # Configure PyTorch to use deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        logger.info(f"Global random seed set to {seed}")

    except Exception as e:
        # This function is critical, so any error should be highly visible.
        logger.error(f"Failed to set random seed: {e}", exc_info=True)
        raise

# Note: In a PyTorch Lightning project, it is often preferred to use the built-in
# `pytorch_lightning.seed_everything(seed)` function, which provides an even
# more comprehensive approach by handling dataloader worker seeding.
# This standalone `set_seed` function is excellent for non-Lightning projects
# or for ensuring a base level of reproducibility.