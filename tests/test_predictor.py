"""
Unit tests for the Inference Predictor component.

This test suite verifies the correctness of the Predictor class's logic,
including output structure and data handling, using mocks and fixtures.
"""
import pytest
import torch
from typing import List

# Assume the Predictor class is importable from its canonical location
from krishi_sahayak.inference.predictor import Predictor

@pytest.fixture
def mock_model() -> torch.nn.Module:
    """Creates a mock PyTorch model that returns predictable logits."""
    class MockModel(torch.nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.num_classes = num_classes
        
        def forward(self, inputs: dict) -> torch.Tensor:
            # Return fixed logits based on batch size, ignoring input content
            batch_size = next(iter(inputs.values())).shape[0]
            # Logits for 3 classes: [[0.1, 2.0, 0.5], [0.1, 2.0, 0.5], ...]
            return torch.tensor([[0.1, 2.0, 0.5]] * batch_size)

    return MockModel(num_classes=3)

@pytest.fixture
def class_names() -> List[str]:
    """Provides a sample list of class names."""
    return ["healthy", "rust", "scab"]

@pytest.fixture
def predictor(mock_model: torch.nn.Module, class_names: List[str]) -> Predictor:
    """Provides a configured Predictor instance for testing."""
    return Predictor(model=mock_model, class_names=class_names, device=torch.device("cpu"))

def test_predictor_initialization(predictor: Predictor):
    """Tests that the Predictor initializes correctly."""
    assert predictor is not None
    assert predictor.model is not None
    assert predictor.class_names == ["healthy", "rust", "scab"]

def test_predict_batch_output_structure(predictor: Predictor):
    """Tests the structure and types of the prediction output for a batch."""
    batch_size = 4
    top_k = 2
    dummy_input = {'rgb': torch.randn(batch_size, 3, 224, 224)}
    
    results = predictor.predict_batch(dummy_input, top_k=top_k)

    assert isinstance(results, list)
    assert len(results) == batch_size, "Should return one result list per item in the batch."
    
    # Check the structure of the first prediction
    first_prediction_list = results[0]
    assert isinstance(first_prediction_list, list)
    assert len(first_prediction_list) == top_k, f"Should return top_k={top_k} predictions."
    
    top_prediction = first_prediction_list[0]
    assert isinstance(top_prediction, dict)
    assert 'class' in top_prediction
    assert 'probability' in top_prediction
    assert isinstance(top_prediction['class'], str)
    assert isinstance(top_prediction['probability'], float)

def test_predict_batch_correct_prediction(predictor: Predictor, class_names: List[str]):
    """Tests that the prediction results match the mock model's fixed output."""
    dummy_input = {'rgb': torch.randn(2, 3, 224, 224)}
    results = predictor.predict_batch(dummy_input, top_k=3)
    
    # The mock model always outputs logits [0.1, 2.0, 0.5].
    # The softmax of this will always have the highest probability at index 1.
    top_prediction = results[0][0]
    assert top_prediction['class'] == class_names[1] # 'rust'
    
    # Check that probabilities sum to approximately 1.0
    total_prob = sum(p['probability'] for p in results[0])
    assert total_prob == pytest.approx(1.0)
