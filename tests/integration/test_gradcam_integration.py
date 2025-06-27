# tests/integration/test_gradcam_integration.py
"""
Integration tests for Grad-CAM implementation with the predictor.
"""

import os
import pytest
import numpy as np
import torch
from PIL import Image

from krishi_sahayak.inference.predictor import Predictor
from krishi_sahayak.utils.visualization import visualize_prediction


class TestGradCAMIntegration:
    """Test suite for Grad-CAM integration with the predictor."""
    
    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """Create a sample PIL image for testing."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    @pytest.fixture
    def sample_model(self) -> torch.nn.Module:
        """Create a simple convolutional model for testing."""
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1), # A potential target layer
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 5)  # 5 classes
        )
        return model
    
    def test_predictor_with_gradcam(self, sample_model):
        """Test that the predictor works with Grad-CAM and auto-detects layers."""
        # Create predictor, relying on auto-detection for target_layers
        predictor = Predictor(
            model=sample_model,
            class_names=[f'class_{i}' for i in range(5)],
            device='cpu'
        )
        
        # REFACTORED: Assert that the auto-detection found a valid layer.
        assert predictor.target_layers is not None, "Target layer auto-detection failed."
        assert len(predictor.target_layers) == 1, "Expected one target layer to be found."
        assert isinstance(predictor.target_layers[0], torch.nn.Module), "Target layer is not a torch Module."

        # REFACTORED: Use a standard random tensor for the input.
        input_tensor = torch.randn(1, 3, 224, 224)
        
        # Make prediction with explanation
        results = predictor.predict_batch(
            inputs=input_tensor,
            top_k=3,
            get_explanations=True,
            xai_config={'method': 'gradcam'}
        )
        
        # Check results structure
        assert len(results) == 1
        result = results[0]
        
        # Check predictions
        assert 'predictions' in result
        assert len(result['predictions']) == 3
        
        # Check explanation
        assert 'explanation' in result
        explanation = result['explanation']
        assert explanation['method'] == 'gradcam'
        assert 'heatmap' in explanation
        assert 'target_class' in explanation
        assert 'class_name' in explanation
        
        # Check heatmap properties
        heatmap = explanation['heatmap']
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2  # 2D heatmap
        assert 0 <= np.min(heatmap) and np.max(heatmap) <= 1, "Heatmap values should be normalized to [0, 1]"
    
    def test_visualization(self, sample_image, tmp_path):
        """Test visualization of prediction and explanation results."""
        # Create sample result dictionary
        result_data = {
            'image_array': np.array(sample_image),
            'predictions': [
                {'class': 'class_1', 'probability': 0.9, 'class_id': 1},
                {'class': 'class_2', 'probability': 0.05, 'class_id': 2},
            ]
        }
        
        # Create sample explanation dictionary
        explanation_data = {
            'heatmap': np.random.rand(56, 56),  # Simulate a smaller heatmap
            'method': 'gradcam',
            'target_class': 1,
            'class_name': 'class_1'
        }
        
        # Test saving the visualization to a file
        output_file = tmp_path / 'visualization.png'
        visualize_prediction(
            result=result_data,
            explanation=explanation_data,
            output_path=output_file,
            show=False # Do not attempt to show plot in a non-interactive environment
        )
        
        # Check that the output file was created and is not empty
        assert output_file.exists()
        assert output_file.stat().st_size > 0