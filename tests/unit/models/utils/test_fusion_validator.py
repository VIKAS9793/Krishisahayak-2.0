# tests/unit/models/utils/test_fusion_validator.py
import pytest
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple # <-- Add Dict

from krishi_sahayak.models.utils.fusion_validator import FusionValidator

class MockModel(nn.Module):
    """A mock model that returns predictable logits based on input shape."""
    def forward(self, x: Any) -> torch.Tensor:
        # Determine batch size
        if isinstance(x, dict):
            batch_size = next(iter(x.values())).shape[0]
        else:
            batch_size = x.shape[0]
        
        # Return fixed logits
        return torch.tensor([[0.1, 0.9]] * batch_size) # Always predicts class 1 with 90% conf

@pytest.fixture
def models() -> Dict[str, nn.Module]:
    """Provides a dictionary of mock models."""
    return {
        "rgb_model": MockModel(),
        "nir_model": MockModel(),
        "fusion_model": MockModel(),
        "nir_simulator": MockModel(),
    }

class TestFusionValidator:
    def test_init(self, models):
        """Verify successful initialization and stat reset."""
        validator = FusionValidator(models['rgb_model'], models['fusion_model'])
        assert validator.stats['total_samples'] == 0
        assert validator.stats['rgb_correct'] == 0

    def test_validate_batch_all_models(self, models):
        """Test statistics aggregation when all models are present."""
        validator = FusionValidator(**models)
        
        batch_size = 4
        rgb = torch.randn(batch_size, 3, 32, 32)
        nir = torch.randn(batch_size, 1, 32, 32)
        target = torch.ones(batch_size).long() # MockModel always predicts class 1
        
        validator._validate_batch(rgb, nir, target)
        
        assert validator.stats['total_samples'] == batch_size
        assert validator.stats['rgb_correct'] == batch_size
        assert validator.stats['nir_correct'] == batch_size
        assert validator.stats['fusion_correct'] == batch_size
        assert validator.stats['nir_samples'] == batch_size
        assert validator.stats['fusion_samples'] == batch_size
        assert validator.stats['rgb_confidence_sum'] == pytest.approx(0.9 * batch_size)

    def test_validate_batch_with_nir_simulation(self, models):
        """Test logic when real NIR is missing but a simulator is used."""
        validator = FusionValidator(
            rgb_model=models['rgb_model'],
            fusion_model=models['fusion_model'],
            nir_simulator=models['nir_simulator']
        )
        batch_size = 2
        rgb = torch.randn(batch_size, 3, 32, 32)
        target = torch.ones(batch_size).long()
        
        validator._validate_batch(rgb, nir=None, target=target)
        
        assert validator.stats['total_samples'] == batch_size
        assert validator.stats['nir_samples'] == 0 # No real NIR model/data
        assert validator.stats['fusion_samples'] == batch_size # Fusion happens with simulated NIR
        assert validator.stats['fusion_correct'] == batch_size

    def test_summary_calculation(self, models):
        """Verify accuracy and confidence calculations in the summary."""
        validator = FusionValidator(**models)
        validator.stats = {
            'rgb_correct': 8.0, 'nir_correct': 7.0, 'fusion_correct': 9.0,
            'rgb_confidence_sum': 9.5, 'nir_confidence_sum': 6.5, 'fusion_confidence_sum': 9.8,
            'nir_samples': 10.0, 'fusion_samples': 10.0, 'total_samples': 10.0
        }
        
        summary = validator._get_summary()
        
        assert summary['rgb_accuracy'] == 0.8
        assert summary['nir_accuracy'] == 0.7
        assert summary['fusion_accuracy'] == 0.9
        assert summary['rgb_avg_confidence'] == 0.95
        assert summary['nir_avg_confidence'] == 0.65
        assert summary['fusion_avg_confidence'] == 0.98

    def test_run_loop(self, models, mocker):
        """Test the main 'run' orchestrator method."""
        device = "cpu"
        # Mock the dataloader to return two batches
        mock_dataloader = [
            {'rgb': torch.randn(2, 3), 'target': torch.ones(2).long()},
            {'rgb': torch.randn(2, 3), 'target': torch.zeros(2).long()},
        ]
        
        # The new batch_processor is simple and testable
        def test_batch_processor(batch):
            return batch['rgb'], None, batch['target']

        validator = FusionValidator(rgb_model=models['rgb_model'], fusion_model=models['fusion_model'])
        mocker.spy(validator, "_validate_batch")
        
        results = validator.run(mock_dataloader, device, test_batch_processor)
        
        assert validator._validate_batch.call_count == 2
        assert results['raw_stats']['total_samples'] == 4
        # MockModel predicts class 1. Target is class 1 in 1st batch, class 0 in 2nd.
        assert results['raw_stats']['rgb_correct'] == 2
        assert results['summary']['rgb_accuracy'] == 0.5
