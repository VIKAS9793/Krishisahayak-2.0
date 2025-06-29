# tests/unit/test_explainability.py
"""
Unit tests for the explainability module.
"""

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from krishi_sahayak.utils.explainability import CAMMethod, create_cam_explainer
from krishi_sahayak.utils.visualization import overlay_cam_on_image

class SimpleCNN(nn.Module):
    """A simple CNN model for testing CAM methods."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # Target layer for CAM
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, num_classes)
    
    def forward(self, x):
        features = self.features(x)
        flat_features = torch.flatten(features, 1)
        return self.classifier(flat_features)

@pytest.fixture
def simple_model() -> SimpleCNN:
    """Fixture that returns an instance of the simple CNN model."""
    return SimpleCNN(num_classes=10)

@pytest.fixture
def input_tensor() -> torch.Tensor:
    """Fixture that returns a test input tensor."""
    return torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image

class ClassifierOutputTarget:
    """A helper callable class to select a specific logit from model output."""
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if model_output.ndim > 1:
            return model_output[:, self.category]
        return model_output[self.category]

@pytest.mark.parametrize("method_name", ["gradcam", "gradcam++"])
def test_cam_methods(simple_model, input_tensor, method_name):
    """
    Test that CAM can be created and executed for implemented methods.
    This test is parameterized to run for both GradCAM and GradCAM++.
    """
    # Define the target layer for CAM extraction
    target_layer = [simple_model.features[2]]
    
    cam_explainer = create_cam_explainer(
        method=method_name,
        model=simple_model,
        target_layers=target_layer,
        device='cpu'
    )
    
    # REFACTORED: Create a proper callable target, as expected by the CAM module.
    targets = [ClassifierOutputTarget(1)]  # Target class index 1
    
    # Generate the CAM output
    cam_output = cam_explainer(input_tensor=input_tensor, targets=targets)
    
    # REFACTORED: Assert the correct return type and properties.
    assert isinstance(cam_output, np.ndarray)
    assert cam_output.ndim == 2  # Should be a 2D heatmap
    assert cam_output.shape == (32, 32) # Should match input spatial dimensions
    assert 0 <= np.min(cam_output) <= 1, "Min value should be >= 0"
    assert 0 <= np.max(cam_output) <= 1, "Max value should be <= 1"

def test_cam_method_enum():
    """Test that CAMMethod enum contains the currently implemented values."""
    implemented_methods = {'gradcam', 'gradcam++'}
    enum_values = {e.value for e in CAMMethod}
    assert implemented_methods.issubset(enum_values)

def test_overlay_cam_on_image():
    """Test that CAM visualization works with a sample image."""
    # Create a sample image (RGB, 0-255)
    img = np.ones((64, 64, 3), dtype=np.uint8) * 128
    
    # Create a sample heatmap mask
    mask = np.zeros((32, 32), dtype=np.float32) # Mask can be a different size
    mask[8:24, 8:24] = 1.0
    
    # Generate visualization
    # REFACTORED: Use the actual cv2 constant for readability.
    result = overlay_cam_on_image(
        image=img,
        mask=mask,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.5
    )
    
    # Check output properties
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 64, 3)  # Same size as input image
    assert result.dtype == np.uint8
    assert 0 <= result.min() and result.max() <= 255 # Valid pixel values
