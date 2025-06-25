import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from krishi_sahayak.inference.model_loader import ModelLoader

@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """Creates a fake checkpoint file for testing."""
    checkpoint_path = tmp_path / "test.ckpt"
    checkpoint_data = {
        "model_config": {"streams": {"rgb": {}}},
        "class_names": ["class_a", "class_b"],
        "state_dict": {"conv1.weight": torch.randn(3, 3, 3, 3)},
        "preprocessing_stats": {"mean": [0.5], "std": [0.5]},
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path

@pytest.fixture
def corrupt_checkpoint(tmp_path: Path) -> Path:
    """Creates a checkpoint file with a missing key."""
    checkpoint_path = tmp_path / "corrupt.ckpt"
    checkpoint_data = {
        "class_names": ["class_a", "class_b"],
        "state_dict": {},
    }
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path

class TestModelLoader:
    def test_init_success(self, fake_checkpoint: Path):
        """Verify successful initialization."""
        loader = ModelLoader(checkpoint_path=fake_checkpoint, device=torch.device("cpu"))
        assert loader.checkpoint is not None
        assert "class_names" in loader.checkpoint

    def test_init_file_not_found(self):
        """Verify FileNotFoundError is raised for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            ModelLoader(checkpoint_path=Path("non_existent.ckpt"), device=torch.device("cpu"))

    def test_init_missing_key(self, corrupt_checkpoint: Path):
        """Verify KeyError is raised for a corrupt checkpoint."""
        with pytest.raises(KeyError, match="Checkpoint is missing required key: 'model_config'"):
            ModelLoader(checkpoint_path=corrupt_checkpoint, device=torch.device("cpu"))

    def test_get_model(self, fake_checkpoint: Path, mocker):
        """Verify that the model is instantiated and state_dict is loaded."""
        # Mock the UnifiedModel to avoid dependency on its internal structure
        mock_model_class = mocker.patch("krishi_sahayak.inference.model_loader.UnifiedModel", autospec=True)
        
        loader = ModelLoader(checkpoint_path=fake_checkpoint, device=torch.device("cpu"))
        model = loader.get_model()

        # Assert that UnifiedModel was called with the correct number of classes
        mock_model_class.assert_called_once_with(num_classes=2, model_config={"streams": {"rgb": {}}})
        
        # Assert that load_state_dict was called on the model instance
        mock_model_instance = mock_model_class.return_value
        mock_model_instance.load_state_dict.assert_called_once()
        mock_model_instance.to.assert_called_with(torch.device("cpu"))
        mock_model_instance.eval.assert_called_once()
        assert model is not None