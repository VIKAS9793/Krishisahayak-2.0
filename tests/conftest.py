"""
Pytest configuration and shared fixtures for the KrishiSahayak test suite.

This file provides reusable fixtures for creating temporary data directories,
loading configurations, and preparing model instances for testing. Fixtures defined
here are automatically available to all tests in the suite.
"""
from __future__ import annotations
from pathlib import Path
import pytest
import yaml

# REFACTORED: Replaced the placeholder 'AppConfig' with the correct, canonical
# master configuration schema from the project.
from krishi_sahayak.config.schemas import MasterConfig

# --- Generic Fixtures ---

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the project root directory, calculated once per test session."""
    # __file__ is this conftest.py, .parent is tests/, .parent.parent is root.
    return Path(__file__).parent.parent


# --- Project-Specific Fixtures ---

@pytest.fixture(scope="session")
def master_config(project_root: Path) -> MasterConfig:
    """
    Loads, validates, and returns the main project configuration as a typed
    Pydantic object.

    This ensures that tests run against a configuration that is guaranteed
    to be valid. The fixture is session-scoped for efficiency.

    Args:
        project_root: The root directory of the project.

    Returns:
        A validated MasterConfig object.
    """
    config_path = project_root / "configs" / "master_config.yaml" # Point to the canonical config
    if not config_path.is_file():
        pytest.skip(f"Master config file not found at {config_path}, skipping tests that require it.")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    try:
        # Validate the raw dictionary against the correct Pydantic model
        return MasterConfig(**config_dict)
    except Exception as e:
        # If validation fails, it's a critical error in the config.
        pytest.fail(f"Failed to parse or validate {config_path}: {e}")


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary data directory structure for a single test.
    This provides test isolation for I/O operations. `tmp_path` is a
    built-in pytest fixture that provides a unique temporary directory.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    return tmp_path


@pytest.fixture
def plantvillage_raw_data_dir(temp_data_dir: Path) -> Path:
    """
    Creates a dummy raw PlantVillage directory structure within a temporary
    directory. This fixture composes `temp_data_dir`.
    """
    pv_dir = temp_data_dir / "raw" / "plantvillage"
    (pv_dir / "Apple___Apple_scab").mkdir(parents=True, exist_ok=True)
    (pv_dir / "Corn___Common_rust").mkdir(parents=True, exist_ok=True)
    
    # Create dummy files
    (pv_dir / "Apple___Apple_scab" / "img1.JPG").touch()
    (pv_dir / "Corn___Common_rust" / "img2.JPG").touch()
    
    return pv_dir