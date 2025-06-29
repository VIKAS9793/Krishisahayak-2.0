import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from krishi_sahayak.utils.logger import setup_logging, LoggingConfig

# Use a global mock to avoid interfering with pytest's own logging
@pytest.fixture(autouse=True)
def mock_dict_config(mocker):
    """Mocks logging.config.dictConfig to inspect the config passed to it."""
    return mocker.patch("logging.config.dictConfig", autospec=True)

class TestSetupLogging:
    def test_default_setup(self, mock_dict_config: MagicMock):
        """Verify that the default configuration is applied correctly."""
        setup_logging()
        
        mock_dict_config.assert_called_once()
        config_arg = mock_dict_config.call_args[0][0]
        
        assert config_arg["version"] == 1
        assert "console" in config_arg["handlers"]
        assert "file" not in config_arg["handlers"]
        assert config_arg["handlers"]["console"]["level"] == "INFO"
        assert "krishisahayak" in config_arg["loggers"]

    def test_setup_with_console_level_override(self, mock_dict_config: MagicMock):
        """Verify that the console_level argument overrides the default."""
        setup_logging(console_level="DEBUG")
        
        config_arg = mock_dict_config.call_args[0][0]
        assert config_arg["handlers"]["console"]["level"] == "DEBUG"

    def test_setup_with_log_file(self, tmp_path: Path, mock_dict_config: MagicMock):
        """Verify that a file handler is correctly added to the config."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, file_level="WARNING")
        
        config_arg = mock_dict_config.call_args[0][0]
        
        assert "file" in config_arg["handlers"]
        assert config_arg["handlers"]["file"]["class"] == "logging.handlers.RotatingFileHandler"
        assert config_arg["handlers"]["file"]["level"] == "WARNING"
        assert config_arg["handlers"]["file"]["filename"] == str(log_file)
        
        # Check that the file handler was added to the logger and root handlers lists
        assert "file" in config_arg["loggers"]["krishisahayak"]["handlers"]
        assert "file" in config_arg["root"]["handlers"]

    def test_setup_with_yaml_config(self, tmp_path: Path, mock_dict_config: MagicMock):
        """Verify that a provided YAML configuration is loaded and used."""
        config_content = """
        version: 1
        disable_existing_loggers: false
        formatters:
          simple:
            format: '%(message)s'
        handlers:
          custom_console:
            class: 'logging.StreamHandler'
            level: 'DEBUG'
            formatter: 'simple'
        root:
          level: 'DEBUG'
          handlers: ['custom_console']
        """
        config_file = tmp_path / "custom_log.yaml"
        config_file.write_text(config_content)
        
        setup_logging(config_path=config_file)
        
        config_arg = mock_dict_config.call_args[0][0]
        assert "custom_console" in config_arg["handlers"]
        assert "console" not in config_arg["handlers"]

    @patch("krishi_sahayak.utils.logger.LoggingConfig", side_effect=ValueError("Pydantic validation failed"))
    @patch("logging.basicConfig")
    def test_setup_fallback_on_validation_error(
        self, mock_basic_config: MagicMock, mock_pydantic: MagicMock, mock_dict_config: MagicMock
    ):
        """Verify that it falls back to basicConfig on Pydantic validation error."""
        setup_logging()
        
        mock_dict_config.assert_not_called()
        mock_basic_config.assert_called_once()
