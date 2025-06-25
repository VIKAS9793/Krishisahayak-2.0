import pytest
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

from krishi_sahayak.config.loader import (
    load_yaml_file,
    parse_cli_overrides,
    merge_configs,
    load_config,
    MAX_CONFIG_SIZE_BYTES
)

# --- Define a simple Pydantic schema for testing ---
class NestedModel(BaseModel):
    d: int

class TestSchema(BaseModel):
    a: str
    b: NestedModel
    c: bool = True


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """Creates a temporary YAML file for testing."""
    config_content = {"a": "hello", "b": {"d": 10}}
    p = tmp_path / "test_config.yaml"
    p.write_text(yaml.dump(config_content))
    return p


class TestLoadYamlFile:
    def test_load_success(self, yaml_file: Path):
        """Verify successful loading and parsing of a valid YAML file."""
        data = load_yaml_file(yaml_file)
        assert data["a"] == "hello"
        assert data["b"]["d"] == 10

    def test_file_not_found(self):
        """Verify FileNotFoundError is raised for a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_file(Path("non_existent_file.yaml"))

    def test_file_too_large(self, yaml_file: Path, mocker):
        """Verify ValueError is raised if the file exceeds the size limit."""
        mocker.patch.object(yaml_file, 'stat', return_value=mocker.Mock(st_size=MAX_CONFIG_SIZE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeds max size"):
            load_yaml_file(yaml_file)
            
    def test_invalid_extension(self, tmp_path: Path):
        """Verify ValueError for files without .yaml or .yml extension."""
        p = tmp_path / "config.txt"
        p.touch()
        with pytest.raises(ValueError, match="must be a .yaml or .yml file"):
            load_yaml_file(p)


class TestParseCliOverrides:
    def test_simple_and_nested_keys(self):
        """Test parsing of both flat and dot-notation nested keys."""
        overrides = ["a.b.c=world", "d=123", "a.e=true"]
        expected = {
            "a": {
                "b": {"c": "world"},
                "e": "true"
            },
            "d": "123"
        }
        assert parse_cli_overrides(overrides) == expected

    def test_empty_list(self):
        """Test that an empty list of overrides produces an empty dict."""
        assert parse_cli_overrides([]) == {}

    def test_invalid_format(self):
        """Test that malformed override strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid override format"):
            parse_cli_overrides(["a.b="])
        with pytest.raises(ValueError, match="Invalid override format"):
            parse_cli_overrides(["key_without_value"])


class TestMergeConfigs:
    def test_simple_merge(self):
        """Test basic merging and overriding."""
        base = {'a': 1, 'b': 2}
        overrides = {'b': 3, 'c': 4}
        expected = {'a': 1, 'b': 3, 'c': 4}
        assert merge_configs(base, overrides) == expected

    def test_nested_merge(self):
        """Test deep merging of nested dictionaries."""
        base = {'a': {'x': 1, 'y': 2}, 'b': 10}
        overrides = {'a': {'y': 3, 'z': 4}, 'c': 20}
        expected = {'a': {'x': 1, 'y': 3, 'z': 4}, 'b': 10, 'c': 20}
        assert merge_configs(base, overrides) == expected


class TestLoadConfig:
    def test_end_to_end_success(self, yaml_file: Path):
        """Test the full orchestration of loading, no overrides."""
        validated_config = load_config(schema=TestSchema, config_path=yaml_file)
        assert isinstance(validated_config, TestSchema)
        assert validated_config.a == "hello"
        assert validated_config.b.d == 10
        assert validated_config.c is True

    def test_end_to_end_with_overrides(self, yaml_file: Path):
        """Test the full orchestration including merging CLI overrides."""
        # Pydantic will coerce the string "false" to a boolean False.
        overrides = ["a=goodbye", "c=false"]
        validated_config = load_config(
            schema=TestSchema, config_path=yaml_file, overrides=overrides
        )
        assert validated_config.a == "goodbye"
        assert validated_config.c is False

    def test_end_to_end_validation_error(self, yaml_file: Path):
        """Test that a Pydantic ValidationError is correctly raised."""
        # This override will fail validation because 'd' expects an int.
        overrides = ["b.d=not_an_int"]
        with pytest.raises(ValidationError):
            load_config(schema=TestSchema, config_path=yaml_file, overrides=overrides)
