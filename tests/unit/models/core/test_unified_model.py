import pytest
import torch
from unittest.mock import MagicMock

from krishi_sahayak.models.core.unified_model import (
    ModelConfig, StreamConfig, FusionConfig, UnifiedModel, _UnifiedModelCore
)

@pytest.fixture
def single_stream_config() -> ModelConfig:
    """Config for a single-stream (e.g., RGB only) model."""
    return ModelConfig(
        backbone_name="resnet18",
        streams={"rgb": StreamConfig(channels=3, pretrained=False)}
    )

@pytest.fixture
def fusion_config() -> ModelConfig:
    """Config for a multi-stream fusion model."""
    return ModelConfig(
        backbone_name="resnet18",
        streams={
            "rgb": StreamConfig(channels=3),
            "nir": StreamConfig(channels=1)
        },
        fusion=FusionConfig(method="concat")
    )

class TestUnifiedModelCore:
    def test_single_stream_forward(self, single_stream_config):
        """Verify the forward pass for a single-stream model."""
        model = _UnifiedModelCore(model_config=single_stream_config, num_classes=10)
        input_tensor = torch.randn(2, 3, 64, 64) # (B, C, H, W)
        output = model(input_tensor)
        assert output.shape == (2, 10)

    def test_fusion_forward(self, fusion_config):
        """Verify the forward pass for a multi-stream fusion model."""
        model = _UnifiedModelCore(model_config=fusion_config, num_classes=10)
        input_dict = {
            "rgb": torch.randn(2, 3, 64, 64),
            "nir": torch.randn(2, 1, 64, 64)
        }
        output = model(input_dict)
        assert output.shape == (2, 10)

class TestUnifiedModel:
    def test_build_model(self, single_stream_config, mocker):
        """Verify that the LightningModule correctly builds the core model."""
        # Mock the parent class init to isolate testing
        mocker.patch("krishi_sahayak.models.base.base.BaseModel.__init__")
        
        pl_model = UnifiedModel(model_config=single_stream_config, num_classes=10)
        
        # _build_model is called inside the parent's init, which we can't easily
        # check after mocking, so we call it directly to test its output.
        core_model = pl_model._build_model()
        assert isinstance(core_model, _UnifiedModelCore)

    def test_get_feature_maps(self, single_stream_config, mocker):
        """Verify that feature maps can be extracted using hooks."""
        mocker.patch("krishi_sahayak.models.base.base.BaseModel.__init__")
        pl_model = UnifiedModel(model_config=single_stream_config, num_classes=10)
        
        # Replace a layer with a mock to control its output
        mock_layer = MagicMock(spec=torch.nn.Module)
        mock_layer.register_forward_hook = torch.nn.Module.register_forward_hook # Use real hook method
        pl_model.model.backbones['rgb'].layer1 = mock_layer
        
        input_dict = {"rgb": torch.randn(2, 3, 64, 64)}
        target_layers = {"rgb": "layer1"}
        
        features = pl_model.get_feature_maps(input_dict, target_layers)
        
        assert "rgb_layer1" in features
        mock_layer.assert_called_once() # Check that the forward pass went through the layer
