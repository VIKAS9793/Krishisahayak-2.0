"""
Smoke tests to ensure all public project modules and key components can be imported.

This test suite verifies that the project's __init__.py files are correctly
configured to expose the intended public API, preventing accidental breaking changes
to the library's import structure.
"""
import pytest
import importlib

# Use a fixture to check for optional dependencies in a more pytest-idiomatic way.
@pytest.fixture(scope="session")
def gan_dependencies_available() -> bool:
    """Checks if optional GAN dependencies can be imported."""
    try:
        importlib.import_module("krishi_sahayak.models.pix2pix")
        return True
    except ImportError:
        return False


class TestPublicAPI:
    """Groups all tests related to the public API surface."""

    def test_import_toplevel_package(self) -> None:
        """Test that the top-level package can be imported and has a version."""
        import krishi_sahayak
        assert hasattr(krishi_sahayak, "__version__")
        assert krishi_sahayak.__version__ is not None

    @pytest.mark.parametrize(
        "subpackage",
        [
            "data",
            "models",
            "utils",
            "pipelines",
            "synthesis",
        ],
    )
    def test_import_subpackages(self, subpackage: str) -> None:
        """Test that all main sub-packages can be imported from the top level."""
        module = importlib.import_module(f"krishi_sahayak.{subpackage}")
        assert module is not None

    def test_import_core_models(self) -> None:
        """Test that core model classes and configs can be imported."""
        from krishi_sahayak.models import (
            BaseModel,
            HybridModel,
            ModelConfig,
            StreamConfig,
            UnifiedModel,
        )

        # A lightweight instantiation test to ensure Pydantic models are valid
        config = ModelConfig(
            backbone_name="efficientnet_b0",
            streams={"rgb": StreamConfig(channels=3, pretrained=True)},
        )
        assert config is not None
        assert issubclass(UnifiedModel, BaseModel)
        assert HybridModel is not None

    def test_import_data_components(self) -> None:
        """Test that public data components can be imported."""
        from krishi_sahayak.data import (
            PlantDiseaseDataModule,
            UnifiedPlantDataset,
            collate_and_filter_none,
        )
        assert PlantDiseaseDataModule is not None
        assert UnifiedPlantDataset is not None
        assert callable(collate_and_filter_none)

    def test_import_pipeline_runners(self) -> None:
        """Test that public pipeline runners can be imported."""
        from krishi_sahayak.pipelines.runners import (
            BaseRunner,
            ClassificationRunner,
            GanRunner,
        )
        assert issubclass(ClassificationRunner, BaseRunner)
        assert issubclass(GanRunner, BaseRunner)

    @pytest.mark.skipif(
        not gan_dependencies_available(),
        reason="GAN optional dependencies not installed",
    )
    def test_import_gan_components(self) -> None:
        """Conditionally test that optional GAN components can be imported."""
        from krishi_sahayak.models.gan import (
            EnhancedDiscriminator,
            EnhancedGenerator,
        )
        from krishi_sahayak.models.pix2pix import Pix2PixGAN

        assert EnhancedGenerator is not None
        assert EnhancedDiscriminator is not None
        assert Pix2PixGAN is not None
