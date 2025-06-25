"""
KrishiSahayak - Hardware Detection Utility

This module provides functions to intelligently detect and select the best
available hardware accelerator for PyTorch operations.
"""
import logging
import torch

logger = logging.getLogger(__name__)

def auto_detect_accelerator() -> str:
    """
    Automatically detects and returns the best available hardware accelerator.

    The priority is: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.

    Returns:
        str: A string representing the best available accelerator ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        logger.info("CUDA (NVIDIA GPU) detected. Using 'cuda' as the accelerator.")
        return "cuda"
    
    # Apple's Metal Performance Shaders (MPS) backend for Apple Silicon GPUs
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            logger.info("MPS (Apple Silicon GPU) detected. Using 'mps' as the accelerator.")
            return "mps"
        else:
            logger.warning("MPS is available but not built. Falling back to CPU.")
            return "cpu"
    
    logger.info("No dedicated GPU detected. Using 'cpu' as the accelerator.")
    return "cpu"
