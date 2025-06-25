import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from krishi_sahayak.utils.visualization import visualize_prediction

@pytest.fixture
def mock_matplotlib(mocker):
    """Mocks the entire matplotlib.pyplot module."""
    return mocker.patch("krishi_sahayak.utils.visualization.plt", autospec=True)

@pytest.fixture
def sample_result(tmp_path: Path) -> dict:
    """Creates a sample prediction result dictionary."""
    # Create a dummy image file for PIL.Image.open to work
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (50, 50)).save(image_path)
    
    return {
        "image_path": str(image_path),
        "predictions": [
            {"class": "Healthy", "probability": 0.95},
            {"class": "Rust", "probability": 0.04},
            {"class": "Powdery Mildew", "probability": 0.01},
        ],
    }

class TestVisualization:
    def test_saves_file_when_path_is_provided(self, mock_matplotlib: MagicMock, sample_result: dict, tmp_path: Path):
        """Verify that plt.savefig is called when an output_path is given."""
        output_path = tmp_path / "viz" / "output.png"
        
        visualize_prediction(sample_result, output_path=output_path)
        
        mock_matplotlib.subplots.assert_called_once()
        mock_matplotlib.savefig.assert_called_once_with(output_path, bbox_inches='tight')
        mock_matplotlib.show.assert_not_called()
        mock_matplotlib.close.assert_called_once()
        assert output_path.parent.exists() # Check if parent directory was created

    def test_closes_plot_on_exception(self, mock_matplotlib: MagicMock, sample_result: dict):
        """Verify that plt.close is called even if an error occurs during plotting."""
        # Simulate an error during plotting
        mock_matplotlib.subplots.side_effect = ValueError("Plotting error")
        
        visualize_prediction(sample_result)
        
        # Should still try to close the figure if it was created
        mock_matplotlib.close.assert_called_once()
