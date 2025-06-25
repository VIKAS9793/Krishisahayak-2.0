import os
import pytest
from unittest.mock import MagicMock

from krishi_sahayak.utils.seed import set_seed

# Define paths to the functions we need to mock
RANDOM_SEED = "random.seed"
NP_RANDOM_SEED = "numpy.random.seed"
TORCH_MANUAL_SEED = "torch.manual_seed"
TORCH_CUDA_MANUAL_SEED = "torch.cuda.manual_seed"
TORCH_CUDA_MANUAL_SEED_ALL = "torch.cuda.manual_seed_all"
TORCH_CUDA_IS_AVAILABLE = "torch.cuda.is_available"
CUDNN_DETERMINISTIC = "torch.backends.cudnn.deterministic"
CUDNN_BENCHMARK = "torch.backends.cudnn.benchmark"


class TestSetSeed:
    """
    Tests the set_seed utility function by mocking external dependencies.
    """

    @pytest.mark.parametrize("seed_value", [0, 42, 12345])
    def test_all_base_libraries_are_seeded(self, mocker, seed_value: int):
        """Verify that all non-GPU seeding functions are called with the correct seed."""
        mock_random = mocker.patch(RANDOM_SEED)
        mock_numpy = mocker.patch(NP_RANDOM_SEED)
        mock_torch = mocker.patch(TORCH_MANUAL_SEED)
        mocker.patch(TORCH_CUDA_IS_AVAILABLE, return_value=False)

        set_seed(seed=seed_value)

        mock_random.assert_called_once_with(seed_value)
        mock_numpy.assert_called_once_with(seed_value)
        mock_torch.assert_called_once_with(seed_value)
        assert os.environ['PYTHONHASHSEED'] == str(seed_value)

    def test_cuda_is_seeded_when_available(self, mocker):
        """Verify that CUDA seeding functions are called when CUDA is available."""
        mocker.patch(TORCH_CUDA_IS_AVAILABLE, return_value=True)
        mock_cuda_seed = mocker.patch(TORCH_CUDA_MANUAL_SEED)
        mock_cuda_seed_all = mocker.patch(TORCH_CUDA_MANUAL_SEED_ALL)
        
        set_seed(seed=42)

        mock_cuda_seed.assert_called_once_with(42)
        mock_cuda_seed_all.assert_called_once_with(42)

    def test_cuda_is_not_seeded_when_unavailable(self, mocker):
        """Verify that CUDA seeding functions are NOT called when CUDA is unavailable."""
        mocker.patch(TORCH_CUDA_IS_AVAILABLE, return_value=False)
        mock_cuda_seed = mocker.patch(TORCH_CUDA_MANUAL_SEED)
        mock_cuda_seed_all = mocker.patch(TORCH_CUDA_MANUAL_SEED_ALL)

        set_seed(seed=42)

        mock_cuda_seed.assert_not_called()
        mock_cuda_seed_all.assert_not_called()

    def test_deterministic_ops_are_set(self, mocker):
        """Verify that CUDNN flags are set when deterministic_ops is True."""
        # We need a way to mock the attributes on the `cudnn` object
        mock_cudnn = MagicMock()
        mocker.patch("torch.backends.cudnn", mock_cudnn)
        
        set_seed(seed=42, deterministic_ops=True)
        
        assert mock_cudnn.deterministic is True
        assert mock_cudnn.benchmark is False

    def test_deterministic_ops_are_not_set(self, mocker):
        """Verify that CUDNN flags are not changed when deterministic_ops is False."""
        mock_cudnn = MagicMock()
        mock_cudnn.deterministic = "initial_value" # Set initial states
        mock_cudnn.benchmark = "initial_value"
        mocker.patch("torch.backends.cudnn", mock_cudnn)

        set_seed(seed=42, deterministic_ops=False)

        # Assert that the attributes were not modified from their initial state
        assert mock_cudnn.deterministic == "initial_value"
        assert mock_cudnn.benchmark == "initial_value"
        
    def test_exception_is_reraised(self, mocker):
        """Verify that any exception during seeding is not silenced."""
        mocker.patch(RANDOM_SEED, side_effect=ValueError("Test Exception"))
        
        with pytest.raises(ValueError, match="Test Exception"):
            set_seed()