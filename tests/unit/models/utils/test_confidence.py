import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from unittest.mock import MagicMock

from krishi_sahayak.models.utils.confidence import ConfidenceThreshold

class MockModel(nn.Module):
    def __init__(self, output_logits: torch.Tensor):
        super().__init__()
        self.output_logits = output_logits
    
    def forward(self, x: Any) -> torch.Tensor:
        # Handle both dict and tensor inputs
        if isinstance(x, dict):
            batch_size = next(iter(x.values())).shape[0]
        else:
            batch_size = x.shape[0]
        return self.output_logits.repeat(batch_size, 1)

@pytest.fixture
def models() -> Tuple[nn.Module, nn.Module]:
    primary_model = MockModel(torch.tensor([[10.0, 0.0], [0.5, 0.4]]))
    secondary_model = MockModel(torch.tensor([[-10.0, 10.0]]))
    return primary_model, secondary_model


class TestConfidenceThreshold:
    def test_init_invalid_threshold(self):
        with pytest.raises(ValueError): 
            ConfidenceThreshold(MagicMock(), MagicMock(), threshold=1.1)

    def test_forward_no_fallback(self, models: tuple):
        primary_model, secondary_model = models
        fallback_module = ConfidenceThreshold(primary_model, secondary_model, threshold=0.9)
        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)
        secondary_model.forward = MagicMock()
        final_logits, metadata = fallback_module(primary_input, fallback_input)
        secondary_model.forward.assert_not_called()
        assert metadata['fallback_count'] == 0

    def test_forward_full_fallback(self, models: tuple):
        """Test the case where all samples fall back to the secondary model."""
        primary_model, secondary_model = models
        # Set a high threshold that all samples will fail
        fallback_module = ConfidenceThreshold(primary_model, secondary_model, threshold=0.99)
        
        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)
        
        final_logits, metadata = fallback_module(primary_input, fallback_input)

        assert metadata['fallback_count'] == 2
        # The final logits should be the secondary model's output for all samples
        assert torch.allclose(final_logits, secondary_model.output_logits.repeat(2, 1))
        
    def test_forward_partial_fallback(self, models: tuple, mocker):
        primary_model, secondary_model = models
        fallback_module = ConfidenceThreshold(primary_model, secondary_model, threshold=0.7)
        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)
        mocker.spy(secondary_model, "forward")
        final_logits, metadata = fallback_module(primary_input, fallback_input)
        assert metadata['fallback_count'] == 1
        assert secondary_model.forward.call_count == 1
        assert torch.allclose(final_logits[1], secondary_model.output_logits[0])

    def test_stats_tracking_and_reset(self, models: tuple):
        """Verify that statistics are tracked correctly and can be reset."""
        fallback_module = ConfidenceThreshold(models[0], models[1], threshold=0.7)
        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)

        fallback_module(primary_input, fallback_input)
        stats = fallback_module.get_fallback_stats()
        assert stats['total_samples'] == 2.0
        assert stats['fallback_count'] == 1.0
        assert stats['fallback_ratio'] == 0.5
        
        fallback_module.reset_stats()
        stats_after_reset = fallback_module.get_fallback_stats()
        assert stats_after_reset['total_samples'] == 0.0
        assert stats_after_reset['fallback_count'] == 0.0
        assert stats_after_reset['fallback_ratio'] == 0.0
