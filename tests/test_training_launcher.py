import argparse
import pytest
from unittest.mock import MagicMock, patch

from krishi_sahayak.launchers.training_launcher import (
    TrainingLauncher,
    _parse_cli_args,
    main,
)

# Mock the Pydantic model and other dependencies for isolated testing
@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies of the launcher."""
    mocker.patch(
        "krishi_sahayak.launchers.training_launcher.SecureConfigParser",
        autospec=True,
    )
    mocker.patch(
        "krishi_sahayak.launchers.training_launcher.TrainingJobManager",
        autospec=True,
    )
    mocker.patch(
        "krishi_sahayak.launchers.training_launcher.TrainingConfig",
        autospec=True,
    )
    mocker.patch(
        "krishi_sahayak.launchers.training_launcher.setup_logging", autospec=True
    )


@pytest.fixture
def mock_args():
    """Provides a default mock of parsed CLI arguments."""
    return argparse.Namespace(
        config="config.yaml",
        job="test_job",
        dry_run=False,
        overrides=["model.param=100"],
    )


class TestTrainingLauncher:
    """Tests the TrainingLauncher class."""

    def test_prepare_config_success(self, mock_dependencies, mock_args, mocker):
        """Verify successful config preparation."""
        launcher = TrainingLauncher(mock_args)
        
        # Setup mocks
        mock_parser_instance = launcher.config_parser
        mock_parser_instance.load_config.return_value = {
            "project_name": "TestProject",
            "paths": {"data": "/data"},
            "training_pipelines": {"test_job": {"model": {"name": "old"}}},
        }
        mock_parser_instance.parse_cli_overrides.return_value = {"model": {"param": 100}}
        mock_parser_instance.merge_configs.return_value = {
            "model": {"name": "old", "param": 100}
        }
        
        # Call the method
        validated_config = launcher._prepare_config()

        # Assertions
        mock_parser_instance.load_config.assert_called_once_with("config.yaml")
        mock_parser_instance.parse_cli_overrides.assert_called_once_with(
            ["model.param=100"]
        )
        mock_parser_instance.merge_configs.assert_called_once()
        
        # Check that the final validated object was created
        mocker.patch.object(launcher, 'TrainingConfig').assert_called_once()
        assert validated_config is not None


    def test_prepare_config_job_not_found(self, mock_dependencies, mock_args):
        """Verify it raises ValueError for a missing job."""
        launcher = TrainingLauncher(mock_args)
        launcher.config_parser.load_config.return_value = {
            "training_pipelines": {"another_job": {}}
        }
        
        with pytest.raises(ValueError, match="Job 'test_job' not found"):
            launcher._prepare_config()

    def test_run_executes_job_manager(self, mock_dependencies, mock_args, mocker):
        """Verify that TrainingJobManager.run() is called on a normal run."""
        launcher = TrainingLauncher(mock_args)
        # Mock the prepare_config to return a dummy config object
        mocker.patch.object(launcher, '_prepare_config', return_value=MagicMock())
        
        launcher.run()

        # Assert that the job manager was created and run
        mock_job_manager_class = mocker.patch.object(launcher, 'TrainingJobManager')
        mock_job_manager_class.assert_called_once()
        mock_job_manager_instance = mock_job_manager_class.return_value
        mock_job_manager_instance.run.assert_called_once()

    def test_run_dry_run_mode(self, mock_dependencies, mock_args, mocker):
        """Verify that JobManager is NOT run in dry-run mode."""
        mock_args.dry_run = True
        launcher = TrainingLauncher(mock_args)
        
        mock_config = MagicMock()
        mocker.patch.object(launcher, '_prepare_config', return_value=mock_config)

        launcher.run()

        # Assert the job manager was NOT called
        mock_job_manager_class = mocker.patch.object(launcher, 'TrainingJobManager')
        mock_job_manager_class.assert_not_called()
        mock_config.model_dump_json.assert_called_once()


class TestMainFunction:
    """Tests the main entrypoint and CLI parsing."""

    def test_parse_cli_args(self):
        """Verify CLI arguments are parsed correctly."""
        argv = ["-c", "path/to/config.yaml", "-j", "my_job", "--dry-run"]
        args = _parse_cli_args(argv)
        assert args.config == "path/to/config.yaml"
        assert args.job == "my_job"
        assert args.dry_run is True
        assert args.overrides == []

    @patch("krishi_sahayak.launchers.training_launcher.TrainingLauncher", autospec=True)
    def test_main_success_path(self, mock_launcher_class, mock_dependencies):
        """Test the main function's successful execution path."""
        with patch("krishi_sahayak.launchers.training_launcher._parse_cli_args") as mock_parse:
            mock_parse.return_value = MagicMock()
            main()
            mock_launcher_class.assert_called_once()
            mock_launcher_instance = mock_launcher_class.return_value
            mock_launcher_instance.run.assert_called_once()

    @patch("krishi_sahayak.launchers.training_launcher.TrainingLauncher", side_effect=Exception("Boom!"))
    @patch("sys.exit")
    def test_main_exception_path(self, mock_exit, mock_launcher_class, mock_dependencies):
        """Test that main catches exceptions and exits with status 1."""
        with patch("krishi_sahayak.launchers.training_launcher._parse_cli_args"):
            main()
            # Assert that due to the exception, sys.exit(1) was called.
            mock_exit.assert_called_once_with(1)
