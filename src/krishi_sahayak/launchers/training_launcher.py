"""
KrishiSahayak - Enterprise Training Launcher Script

This script serves as the primary entry point for initiating training pipelines.
It is responsible for parsing configurations, applying overrides, validating settings,
and handing off execution to the TrainingJobManager.
"""

import argparse
import logging
import sys
from typing import List, Optional

# It's assumed these components exist and have their own robust logic.
from krishi_sahayak.config.loader import load_config
from krishi_sahayak.pipelines.job_manager import (
    TrainingJobManager,
    TrainingConfig,
)
from krishi_sahayak.utils.logger import setup_logging

# A logger for this specific module.
logger = logging.getLogger(__name__)


class TrainingLauncher:
    """
    Orchestrates the loading, validation, and execution of a training job.

    This class encapsulates the logic for parsing configurations from files and
    CLI overrides, ensuring a validated configuration object is passed to the
    job manager.
    """

    def __init__(self, cli_args: argparse.Namespace) -> None:
        """
        Initializes the launcher with parsed command-line arguments.

        Args:
            cli_args: An object containing parsed CLI arguments. Expected attributes
                      include 'config', 'job', 'overrides', and 'dry_run'.
        """
        self.args = cli_args

    def _prepare_config(self) -> TrainingConfig:
        """
        Loads, merges, and validates the configuration for the specified job.

        This is the core logic that combines the base YAML config with CLI
        overrides to produce the final, validated Pydantic model.

        Returns:
            A validated TrainingConfig Pydantic model instance.

        Raises:
            ValueError: If the specified job is not found in the config file.
        """
        from krishi_sahayak.pipelines.schemas import TrainingConfig
        full_config = load_config(TrainingConfig, Path(self.args.config), self.args.overrides)

        job_config = full_config.get("training_pipelines", {}).get(self.args.job)
        if not job_config:
            raise ValueError(f"Job '{self.args.job}' not found in the config file.")
        logger.info(f"Found configuration for job: '{self.args.job}'")

        overrides = self.config_parser.parse_cli_overrides(self.args.overrides)
        if overrides:
            logger.info(f"Applying CLI overrides: {overrides}")

        # Combine base job config with CLI overrides.
        final_job_config = self.config_parser.merge_configs(job_config, overrides)

        # Inject global settings required by the job.
        final_job_config.setdefault("paths", full_config.get("paths", {}))
        final_job_config.setdefault(
            "project_name", full_config.get("project_name")
        )

        logger.info("Validating final configuration against the schema...")
        # SECURITY NOTE: Ensure `TrainingConfig` uses `pydantic.SecretStr`
        # for any sensitive fields (e.g., api_keys, passwords) to prevent
        # them from being exposed in logs or dry runs.
        validated_config = TrainingConfig(**final_job_config)
        logger.info("Configuration successfully validated.")

        return validated_config

    def run(self) -> None:
        """
        Prepares the configuration and executes the training job.
        Handles the --dry-run flag to prevent actual execution.
        """
        validated_config = self._prepare_config()

        if self.args.dry_run:
            # Using model_dump_json for safe serialization of SecretStr fields.
            config_json = validated_config.model_dump_json(indent=4)
            logger.info(
                f"--- DRY RUN MODE ---"
                f"\nJob '{self.args.job}' configuration is valid."
                f"\nFinal validated config:\n{config_json}"
            )
            return

        logger.info(f"Initializing TrainingJobManager for job '{self.args.job}'...")
        job_manager = TrainingJobManager(validated_config)
        job_manager.run()
        logger.info(f"Training job '{self.args.job}' completed successfully.")


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parses command-line arguments.

    Args:
        argv: A list of strings representing the command line arguments.
              If None, `sys.argv[1:]` is used.

    Returns:
        An argparse.Namespace object with the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="KrishiSahayak Enterprise Training Launcher"
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the main YAML config file."
    )
    parser.add_argument(
        "-j", "--job", type=str, required=True, help="The name of the training job to run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and show final settings without running the job.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Key-value pairs to override config, e.g., 'model.name=new_model'",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Main entry point for the training launcher."""
    setup_logging(project_name="krishi_sahayak_training")

    try:
        cli_args = _parse_cli_args()
        launcher = TrainingLauncher(cli_args)
        launcher.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()