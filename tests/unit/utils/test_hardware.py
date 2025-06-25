import pytest
from unittest.mock import MagicMock

from krishi_sahayak.utils.hardware import auto_detect_accelerator

class TestAutoDetectAccelerator:
    """
    Tests the auto_detect_accelerator function by mocking the environment.
    """

    def test_cuda_is_detected(self, mocker):
        """
        Verify that 'cuda' is returned when torch.cuda.is_available() is True.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        # We don't need to mock MPS because the CUDA check comes first.

        result = auto_detect_accelerator()
        assert result == "cuda"

    def test_mps_is_detected(self, mocker):
        """
        Verify that 'mps' is returned when CUDA is unavailable but MPS is
        available and built.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=True)
        mocker.patch("torch.backends.mps.is_built", return_value=True)
        
        result = auto_detect_accelerator()
        assert result == "mps"

    def test_cpu_is_used_when_mps_not_built(self, mocker):
        """
        Verify that 'cpu' is returned when MPS is available but not built.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=True)
        mocker.patch("torch.backends.mps.is_built", return_value=False)
        
        result = auto_detect_accelerator()
        assert result == "cpu"

    def test_cpu_fallback_when_no_gpu_available(self, mocker):
        """
        Verify that 'cpu' is returned when neither CUDA nor MPS are available.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)
        
        result = auto_detect_accelerator()
        assert result == "cpu"
