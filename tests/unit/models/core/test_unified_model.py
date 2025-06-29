# tests/unit/models/core/test_unified_model.py
import pytest
import torch
import torch.nn as nn
from typing import Any

from krishi_sahayak.models.core.unified_model import _UnifiedModelCore, UnifiedModel

# REFACTORED: Imports are now separated to reflect the final, correct schema locations.
from krishi_sahayak.models.schemas import ModelConfig, StreamConfig, FusionConfig
from krishi_sahayak.models.base import BaseModelConfig


@pytest.fixture
def single_stream_config() -> ModelConfig:
    """Provides a valid config for a single-stream model."""
    return ModelConfig(backbone_name='resnet18', streams={'rgb': StreamConfig(channels=3)})

@pytest.fixture
def fusion_config() -> ModelConfig:
    """Provides a valid config for a multi-stream fusion model."""
    return ModelConfig(
        backbone_name='resnet18',
        streams={'rgb': StreamConfig(channels=3), 'nir': StreamConfig(channels=1)},
        fusion=FusionConfig()
    )

@pytest.fixture
def base_config() -> BaseModelConfig:
    """Provides a default Base Model configuration for testing."""
    return BaseModelConfig()


class TestUnifiedModelCore:
    """Unit tests for the internal _UnifiedModelCore nn.Module."""

    def test_single_stream_forward(self, single_stream_config: ModelConfig):
        """Verify the forward pass for a single-stream model."""
        model = _UnifiedModelCore(model_config=single_stream_config, num_classes=10)
        output = model(torch.randn(2, 3, 64, 64))
        assert output.shape == (2, 10)

    def test_fusion_forward(self, fusion_config: ModelConfig):
        """Verify the forward pass for a multi-stream fusion model."""
        model = _UnifiedModelCore(model_config=fusion_config, num_classes=10)
        input_dict = {
            "rgb": torch.randn(2, 3, 64, 64),
            "nir": torch.randn(2, 1, 64, 64)
        }
        output = model(input_dict)
        assert output.shape == (2, 10)


class TestUnifiedModel:
    """Unit tests for the UnifiedModel LightningModule wrapper."""

    def test_build_model(self, single_stream_config: ModelConfig, base_config: BaseModelConfig, mocker: Any):
        """Verify that the LightningModule correctly builds the core model."""
        mocker.patch("krishi_sahayak.models.base.base.BaseModel.__init__")
        
        pl_model = UnifiedModel(
            model_config=single_stream_config,
            base_config=base_config,
            num_classes=10
        )
        assert isinstance(pl_model.model, _UnifiedModelCore)
        # Verify the parent init was called correctly
        pl_model.__init__.assert_called_once()

    def test_get_feature_maps(self, single_stream_config: ModelConfig, base_config: BaseModelConfig, mocker: Any):
        """Verify that the feature map extraction hook raises an error if no valid layers are found."""
        mocker.patch("krishi_sahayak.models.base.base.BaseModel.__init__")
        
        pl_model = UnifiedModel(
            model_config=single_stream_config,
            base_config=base_config,
            num_classes=10
        )
        
        # Test that it raises an error with a non-existent layer
        with pytest.raises(ValueError, match="No valid layers were provided"):
            pl_model.get_feature_maps({}, {'rgb': 'non_existent_layer'})