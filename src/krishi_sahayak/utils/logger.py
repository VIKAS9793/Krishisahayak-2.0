# src/krishi_sahayak/utils/logger.py
"""
KrishiSahayak - Advanced, Configurable Logging Utility (Refactored)

This module provides a robust and encapsulated setup for logging across the
project. It uses Python's standard `dictConfig` and validates the configuration
with Pydantic for maximum robustness. It gracefully handles optional
dependencies like `colorlog`.
"""
from __future__ import annotations
import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

# Gracefully handle the optional colorlog dependency
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Define valid logging levels for stricter Pydantic validation
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# =============================================================================
# PART 1: Pydantic Configuration Models
# =============================================================================
class FormatterConfig(BaseModel):
    """Schema for a logging formatter."""
    format: str
    datefmt: Optional[str] = None
    # '()' is a special key used by dictConfig to specify a factory/class.
    class_: str = Field(alias="()")


class HandlerConfig(BaseModel):
    """Schema for a logging handler."""
    # 'class' is a standard key in dictConfig for the handler class.
    class_: str = Field(alias="class")
    level: LogLevel
    formatter: str
    stream: Optional[str] = None
    filename: Optional[str] = None
    maxBytes: Optional[int] = None
    backupCount: Optional[int] = None
    encoding: Optional[str] = None


class LoggerConfig(BaseModel):
    """Schema for a specific logger instance."""
    level: LogLevel
    handlers: List[str]
    propagate: bool


class LoggingConfig(BaseModel):
    """Master schema for the entire logging configuration."""
    version: Literal[1]
    disable_existing_loggers: bool
    formatters: Dict[str, FormatterConfig]
    handlers: Dict[str, HandlerConfig]
    loggers: Dict[str, LoggerConfig]
    root: Optional[LoggerConfig] = None


# =============================================================================
# PART 2: Default Configuration and Helper Functions
# =============================================================================
def _generate_default_config(project_name: str, console_level: LogLevel) -> Dict[str, Any]:
    """Generates a robust default logging configuration dictionary dynamically."""
    console_formatter = "console_color" if COLORLOG_AVAILABLE else "console_simple"
    
    # If colorlog is desired but not available, log a warning early.
    if not COLORLOG_AVAILABLE and console_formatter == "console_color":
        logging.warning("`colorlog` package not found. Console output will not be colored.")
        console_formatter = "console_simple"

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console_simple": {
                "()": "logging.Formatter",
                "format": "%(asctime)s - [%(levelname)s] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "file_detailed": {
                "()": "logging.Formatter",
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": console_formatter,
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            project_name: {
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    }

    if COLORLOG_AVAILABLE:
        config["formatters"]["console_color"] = {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(asctime)s - [%(levelname)s] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    
    return config

def _add_file_handler_to_config(config: Dict, log_file: Path, file_level: LogLevel) -> None:
    """Dynamically adds a rotating file handler to the config dictionary."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    config["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": file_level,
        "formatter": "file_detailed",
        "filename": str(log_file),
        "maxBytes": 10 * 1024 * 1024,  # 10 MB
        "backupCount": 5,
        "encoding": "utf8",
    }
    # Add the file handler to all defined loggers
    for logger_name in config.get("loggers", {}):
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
    if config.get("root") and "file" not in config["root"]["handlers"]:
        config["root"]["handlers"].append("file")

# =============================================================================
# PART 3: Main Setup Function
# =============================================================================
def setup_logging(
    project_name: str = "krishisahayak",
    config_path: Optional[Path] = None,
    log_file: Optional[Path] = None,
    console_level: LogLevel = "INFO",
    file_level: LogLevel = "DEBUG",
) -> None:
    """
    Sets up logging for the project.

    It loads configuration from a YAML file if provided; otherwise, it uses a
    robust default. It can dynamically add a file logger. If any part of this
    setup fails, it falls back to a basic configuration to ensure that
    error messages are not lost.
    """
    try:
        if config_path and config_path.is_file():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            log_source_msg = f"from config file: {config_path}"
        else:
            # Generate the default config dynamically. This is more robust.
            config_dict = _generate_default_config(project_name, console_level)
            log_source_msg = "from internal default"

        if log_file:
            _add_file_handler_to_config(config_dict, log_file, file_level)

        # Validate the final configuration dictionary against our Pydantic schema
        validated_config = LoggingConfig(**config_dict)
        
        # Apply the validated config. Use model_dump to handle Pydantic-specific fields.
        logging.config.dictConfig(validated_config.model_dump(by_alias=True))
        
        logger = logging.getLogger(project_name)
        logger.info(f"Logging configured for '{project_name}' (loaded {log_source_msg}).")

    except Exception as e:
        # Fallback configuration in case of any error during the setup process
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - [%(levelname)s] - FATAL: Failed to configure logger: %(message)s"
        )
        logging.critical(f"Error during logging setup: {e}. Falling back to basic config.", exc_info=True)