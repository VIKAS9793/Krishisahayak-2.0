import pytest
import torch
from PIL import Image
from pathlib import Path
from unittest.mock import MagicMock

from krishi_sahayak.inference.data_loader import InferenceDataset, create_transforms

@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with fake image files."""
    d = tmp_path / "images"
    d.mkdir()
    # Create valid images
    Image.new('RGB', (10, 10)).save(d / "img1.png")
    Image.new('RGB', (10, 10)).save(d / "img1_nir.tif")
    Image.new('RGB', (10, 10)).save(d / "img2.jpg")
    # Create a corrupt file
    (d / "corrupt.jpg").write_text("not an image")
    # Create an image without a matching NIR file
    Image.new('RGB', (10, 10)).save(d / "img3.png")
    return d

@pytest.fixture
def mock_transforms():
    """Returns mock transform functions."""
    return MagicMock(return_value=torch.randn(3, 8, 8)), MagicMock(return_value=torch.randn(1, 8, 8))


class TestInferenceDataset:
    def test_from_directory(self, image_dir, mock_transforms):
        """Test the factory method finds the correct number of images."""
        rgb_transform, nir_transform = mock_transforms
        dataset = InferenceDataset.from_directory(
            image_dir, rgb_transform, is_hybrid=False, nir_transform=nir_transform
        )
        # Finds img1.png, img2.jpg, img3.png (corrupt.jpg is not an image)
        # Although corrupt.jpg is created, PIL will fail on it, but it is found by glob
        assert len(dataset.file_paths) == 4

    def test_getitem_rgb_only(self, image_dir, mock_transforms):
        """Test loading a standard RGB image."""
        rgb_transform, _ = mock_transforms
        dataset = InferenceDataset([image_dir / "img2.jpg"], rgb_transform, is_hybrid=False)
        item = dataset[0]
        
        assert item is not None
        assert 'rgb' in item
        assert 'path' in item
        assert 'ms' not in item
        assert item['path'] == str(image_dir / "img2.jpg")
        rgb_transform.assert_called_once()
        
    def test_getitem_hybrid_success(self, image_dir, mock_transforms):
        """Test loading an RGB image with its corresponding NIR image."""
        rgb_transform, nir_transform = mock_transforms
        dataset = InferenceDataset(
            [image_dir / "img1.png"], rgb_transform, is_hybrid=True, nir_transform=nir_transform
        )
        item = dataset[0]

        assert item is not None
        assert 'rgb' in item
        assert 'ms' in item
        rgb_transform.assert_called_once()
        nir_transform.assert_called_once()

    def test_getitem_missing_nir(self, image_dir, mock_transforms, caplog):
        """Test that an item is skipped if its NIR file is missing in hybrid mode."""
        rgb_transform, nir_transform = mock_transforms
        dataset = InferenceDataset(
            [image_dir / "img3.png"], rgb_transform, is_hybrid=True, nir_transform=nir_transform
        )
        item = dataset[0]
        
        assert item is None
        assert "Cannot find matching NIR file" in caplog.text

    def test_getitem_corrupt_image(self, image_dir, mock_transforms, caplog):
        """Test that a corrupt image is gracefully skipped."""
        rgb_transform, _ = mock_transforms
        dataset = InferenceDataset([image_dir / "corrupt.jpg"], rgb_transform, is_hybrid=False)
        item = dataset[0]
        
        assert item is None
        assert "Skipping corrupt or unreadable image" in caplog.text
        
    def test_collate_fn(self):
        """Verify the collate_fn filters out None values."""
        batch = [
            {"path": "a", "rgb": torch.tensor(1)},
            None,
            {"path": "c", "rgb": torch.tensor(3)},
            None,
        ]
        collated = InferenceDataset.collate_fn(batch)
        
        assert len(collated) == 2
        assert all(isinstance(p, str) for p in collated['path'])
        assert collated['rgb'].shape == (2,)

    def test_collate_fn_all_none(self):
        """Verify it returns an empty dict if all items in the batch are None."""
        batch = [None, None, None]
        collated = InferenceDataset.collate_fn(batch)
        assert collated == {}
