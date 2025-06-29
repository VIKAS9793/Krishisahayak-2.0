# tests/smoke/test_public_api_imports.py
"""
Smoke tests to ensure all public project modules and key components can be imported.
"""
import pytest
import importlib

# REFACTORED: This is now a regular helper function, not a pytest fixture.
# This allows it to be called directly inside the @pytest.mark.skipif marker.
def gan_dependencies_available() -> bool:
    """Checks if optional GAN dependencies can be imported."""
    try:
        # Check for a specific, uniquely named module from the GAN package
        importlib.import_module("krishi_sahayak.models.gan.pix2pix")
        return True
    except ImportError:
        return False


class TestPublicAPI:
    """Groups all tests related to the public API surface."""

    def test_import_toplevel_package(self) -> None:
        """Test that the top-level package can be imported and has a version."""
        import krishi_sahayak
        assert hasattr(krishi_sahayak, "__version__")

    @pytest.mark.parametrize(
        "subpackage",
        [
            "data",
            "models",
            "utils",
            "pipelines",
            "api",
            "config",
            "inference",
        ],
    )
    def test_import_subpackages(self, subpackage: str) -> None:
        """Test that all main sub-packages can be imported."""
        module = importlib.import_module(f"krishi_sahayak.{subpackage}")
        assert module is not None

    def test_import_core_models(self) -> None:
        """Test that core model classes can be imported."""
        from krishi_sahayak.models import BaseModel, UnifiedModel, HybridModel
        assert BaseModel is not None
        assert UnifiedModel is not None
        assert HybridModel is not None

    # REFACTORED: The skipif marker now correctly calls the helper function.
    @pytest.mark.skipif(not gan_dependencies_available(), reason="GAN optional dependencies not installed")
    def test_import_gan_components(self) -> None:
        """Conditionally test that optional GAN components can be imported."""
        from krishi_sahayak.models.gan import EnhancedDiscriminator, EnhancedGenerator
        assert EnhancedGenerator is not None
        assert EnhancedDiscriminator is not None