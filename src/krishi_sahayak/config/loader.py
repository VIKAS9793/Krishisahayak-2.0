"""
Secure, Pydantic-driven configuration loader for the KrishiSahayak project.

This module provides robust, standalone functions for loading YAML files,
parsing command-line overrides, merging configurations, and validating them
against Pydantic schemas.
"""
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
MAX_CONFIG_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

def load_config(
    schema: Type[BaseModel],
    config_path: Path,
    overrides: Optional[List[str]] = None,
) -> BaseModel:
    """
    Main entry point to securely load, merge, and validate configurations.

    Args:
        schema: The Pydantic model to validate the final configuration against.
        config_path: Path to the base YAML configuration file.
        overrides: An optional list of CLI overrides (e.g., ["key.a=val"]).

    Returns:
        A validated Pydantic model instance representing the configuration.
    """
    logger.info(f"Loading base configuration from '{config_path}'...")
    base_config = load_yaml_file(config_path)

    cli_overrides = parse_cli_overrides(overrides or [])
    if cli_overrides:
        logger.info(f"Applying CLI overrides: {cli_overrides}")

    final_config = merge_configs(base_config, cli_overrides)

    try:
        validated_config = schema(**final_config)
        logger.info(f"Configuration successfully validated against {schema.__name__}.")
        return validated_config
    except ValidationError as e:
        logger.error(f"Configuration validation failed for {config_path}:\n{e}")
        raise

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Securely loads a YAML file into a dictionary.

    Args:
        path: The path to the YAML file.

    Returns:
        The content of the YAML file as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is too large or not a .yaml/.yml file.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    if path.suffix.lower() not in [".yaml", ".yml"]:
        raise ValueError(f"Configuration file must be a .yaml or .yml file: {path}")

    if path.stat().st_size > MAX_CONFIG_SIZE_BYTES:
        raise ValueError(f"Configuration file exceeds max size of {MAX_CONFIG_SIZE_BYTES // 1024**2}MB.")

    with open(path, 'r') as f:
        # Use safe_load to prevent arbitrary code execution vulnerabilities.
        return yaml.safe_load(f)

def parse_cli_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    Parses a list of dot-notation CLI overrides into a nested dictionary.

    Example:
        ["a.b.c=val", "a.d=10"] -> {"a": {"b": {"c": "val"}, "d": "10"}}

    Args:
        overrides: A list of 'key=value' strings.

    Returns:
        A nested dictionary representing the overrides.
    """
    config = {}
    for override in overrides:
        try:
            key_str, value = override.split('=', 1)
            keys = key_str.split('.')
        except ValueError:
            raise ValueError(f"Invalid override format: '{override}'. Must be 'key=value'.")

        temp_dict = config
        for key in keys[:-1]:
            temp_dict = temp_dict.setdefault(key, {})
        temp_dict[keys[-1]] = value
    return config

def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. The `overrides` dict takes precedence.

    Args:
        base: The base configuration dictionary.
        overrides: The dictionary with override values.

    Returns:
        A new dictionary containing the merged configuration.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if isinstance(value, MutableMapping) and isinstance(merged.get(key), MutableMapping):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged