import os
import random
import numpy as np
import torch
import pytest
from unittest.mock import patch
from krishi_sahayak.utils.seed import set_seed

TORCH_CUDA_IS_AVAILABLE = "torch.cuda.is_available"
TORCH_CUDA_MANUAL_SEED = "torch.cuda.manual_seed"
TORCH_CUDA_MANUAL_SEED_ALL = "torch.cuda.manual_seed_all"

class TestSetSeed:
    def test_seeds_are_set(self, mocker):
        mock_random = mocker.patch("random.seed")
        mock_np = mocker.patch("numpy.random.seed")
        mock_torch = mocker.patch("torch.manual_seed")
        set_seed(seed=123)
        mock_random.assert_called_once_with(123)
        mock_np.assert_called_once_with(123)
        mock_torch.assert_called_once_with(123)

    def test_cuda_is_seeded_when_available(self, mocker):
        mocker.patch(TORCH_CUDA_IS_AVAILABLE, return_value=True)
        mock_cuda_seed = mocker.patch(TORCH_CUDA_MANUAL_SEED)
        mock_cuda_seed_all = mocker.patch(TORCH_CUDA_MANUAL_SEED_ALL)
        set_seed(seed=42)
        # REFACTORED: Assert that it's called with 42, not how many times.
        mock_cuda_seed.assert_called_with(42)
        mock_cuda_seed_all.assert_called_with(42)