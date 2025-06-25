"""
Tests for the FastAPI application endpoints.

These tests verify the functionality of the API endpoints using a test client
and mock dependencies to isolate the API layer from the model inference.
"""

import pytest
from httpx import AsyncClient
from pathlib import Path
from unittest.mock import MagicMock

# Import the app object and the dependency function to override it
from src.krishi_sahayak.api.main import app, get_inference_handler

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_inference_handler() -> MagicMock:
    """Provides a mock of the InferenceHandler."""
    handler = MagicMock()
    # Configure the return value for the method we will call
    handler.run_single.return_value = {
        "model_checkpoint": "mock_model.ckpt",
        "image_path": "test.jpg",
        "predictions": [{"class": "MockHealthy", "probability": 0.99}]
    }
    return handler

async def test_health_check(mock_inference_handler):
    """Tests the /health endpoint, ensuring it reflects model status."""
    app.dependency_overrides[get_inference_handler] = lambda: mock_inference_handler
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
        
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    
    app.dependency_overrides = {} # Clean up

async def test_predict_endpoint(mock_inference_handler, tmp_path: Path):
    """Tests the /predict endpoint with a mock handler and dummy file."""
    app.dependency_overrides[get_inference_handler] = lambda: mock_inference_handler
    
    dummy_file_path = tmp_path / "test.jpg"
    dummy_file_path.write_bytes(b"fake image data")

    async with AsyncClient(app=app, base_url="http://test") as ac:
        with open(dummy_file_path, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            response = await ac.post("/predict", files=files)
            
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.jpg"
    assert data["predictions"][0]["class_name"] == "MockHealthy"
    
    app.dependency_overrides = {} # Clean up

async def test_predict_invalid_file_type(mock_inference_handler):
    """Tests that a non-image file type returns a 400 Bad Request error."""
    app.dependency_overrides[get_inference_handler] = lambda: mock_inference_handler
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = await ac.post("/predict", files=files)
            
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    
    app.dependency_overrides = {} # Clean up
