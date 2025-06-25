import pytest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from krishi_sahayak.models.confidence import ConfidenceThreshold

class MockModel(nn.Module):
    """A mock model that returns predictable logits."""
    def __init__(self, output_logits: torch.Tensor):
        super().__init__()
        self.output_logits = output_logits

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Return the predefined logits, repeated for batch size
        batch_size = next(iter(kwargs.values())).shape[0] if kwargs else args[0].shape[0]
        return self.output_logits.repeat(batch_size, 1)

@pytest.fixture
def models() -> Tuple[nn.Module, nn.Module]:
    """Provides mock primary and secondary models."""
    # Primary model will output high confidence for class 0, low for class 1
    primary_logits = torch.tensor([[10.0, 0.0], [0.5, 0.4]]) # High confidence, Low confidence
    primary_model = MockModel(primary_logits)

    # Secondary model returns a very confident prediction for class 1
    secondary_logits = torch.tensor([[-10.0, 10.0]])
    secondary_model = MockModel(secondary_logits)
    return primary_model, secondary_model


class TestConfidenceThreshold:
    """Tests the ConfidenceThreshold module."""

    def test_init_invalid_threshold(self):
        """Verify it raises errors for thresholds outside the (0, 1] range."""
        with pytest.raises(ValueError):
            ConfidenceThreshold(MagicMock(), MagicMock(), threshold=1.1)
        with pytest.raises(ValueError):
            ConfidenceThreshold(MagicMock(), MagicMock(), threshold=0.0)

    def test_forward_no_fallback(self, models: tuple):
        """Test the case where all samples are above the confidence threshold."""
        primary_model, secondary_model = models
        fallback_module = ConfidenceThreshold(primary_model, secondary_model, threshold=0.9)
        
        # Create inputs where the primary model will be confident on both samples
        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)

        # Mock the secondary model to ensure it's not called
        secondary_model.forward = MagicMock()
        
        final_logits, metadata = fallback_module(primary_input, fallback_input)

        secondary_model.forward.assert_not_called()
        assert metadata['fallback_count'] == 0
        assert torch.allclose(final_logits, primary_model.output_logits)

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
        """Test the case where only some samples fall back."""
        primary_model, secondary_model = models
        fallback_module = ConfidenceThreshold(primary_model, secondary_model, threshold=0.7)

        primary_input = {'input': torch.randn(2, 3)}
        fallback_input = torch.randn(2, 3)
        
        # Spy on the secondary model's forward pass
        mocker.spy(secondary_model, "forward")

        final_logits, metadata = fallback_module(primary_input, fallback_input)

        assert metadata['fallback_count'] == 1
        
        # Verify secondary model was called with only the low-confidence sample
        assert secondary_model.forward.call_count == 1
        call_args = secondary_model.forward.call_args[0][0]
        assert call_args.shape[0] == 1 # Batch size of the call should be 1
        
        # The first sample's logits should be from the primary model
        assert torch.allclose(final_logits[0], primary_model.output_logits[0])
        # The second sample's logits should be from the secondary model
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
