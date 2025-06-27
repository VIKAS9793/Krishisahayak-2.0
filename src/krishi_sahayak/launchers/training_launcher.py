# src/krishi_sahayak/launchers/training_launcher.py
"""
KrishiSahayak - Enterprise Training Launcher Script (Refactored)

This script serves as the primary entry point for initiating training pipelines.
It correctly uses the project's canonical configuration loader to parse,
validate, and hand off execution to the TrainingJobManager.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# REFACTORED: Corrected import paths to align with package structure
from krishisahayak.config.loader import load_config
from krishisahayak.config.schemas import MasterConfig
from krishisahayak.pipelines.job_manager import TrainingJobManager
from krishisahayak.utils.logger import setup_logging

# A logger for this specific module
logger = logging.getLogger(__name__)


class TrainingLauncher:
    """
    Orchestrates the loading, validation, and execution of a training job.
    """

    def __init__(self, cli_args: argparse.Namespace) -> None:
        """Initializes the launcher with parsed command-line arguments."""
        self.args = cli_args

    def launch(self) -> None:
        """
        Prepares the configuration and executes the training job.
        Handles the --dry-run flag to prevent actual execution.
        """
        logger.info("--- KrishiSahayak Training Launcher Initialized ---")
        
        # The entire config loading process is handled by the project's
        # robust `load_config` utility, ensuring a single source of truth.
        logger.info(f"Loading master configuration from: {self.args.config_path}")
        master_config = load_config(
            schema=MasterConfig,
            config_path=self.args.config_path,
            overrides=self.args.overrides
        )
        
        # Extract the specific job configuration from the validated master config
        job_config = master_config.training_pipelines.get(self.args.job)
        if not job_config:
            raise ValueError(f"Job '{self.args.job}' not found in the 'training_pipelines' section of the config file.")
        logger.info(f"Found configuration for job: '{self.args.job}'")

        if self.args.dry_run:
            config_json = job_config.model_dump_json(indent=4)
            logger.info(
                f"--- DRY RUN MODE ---\n"
                f"Job '{self.args.job}' configuration is valid.\n"
                f"Final validated config for job:\n{config_json}"
            )
            return

        logger.info(f"Initializing TrainingJobManager for job '{self.args.job}'...")
        # The TrainingJobManager is instantiated with the validated Pydantic objects
        job_manager = TrainingJobManager(
            config=job_config,
            paths=master_config.paths,
            job_name=self.args.job
        )
        job_manager.run()
        logger.info(f"Training job '{self.args.job}' completed successfully.")


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Defines and parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="KrishiSahayak Enterprise Training Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config-path", type=Path, required=True,
        help="Path to the main YAML config file."
    )
    parser.add_argument(
        "--job", type=str, required=True,
        help="The name of the training job to run (must be a key under 'training_pipelines')."
    )
    parser.add_argument(
        "--dry-run", action='store_true',
        help="Validate config and show final settings without running the job."
    )
    parser.add_argument(
        "overrides", nargs="*", default=[],
        help="Key-value pairs to override config, e.g., 'training_params.learning_rate=0.0005'"
    )
    return parser.parse_args(argv)


def main() -> None:
    """Main entry point for the training launcher."""
    setup_logging(project_name="krishisahayak_training")

    try:
        cli_args = _parse_cli_args()
        launcher = TrainingLauncher(cli_args)
        launcher.launch()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()