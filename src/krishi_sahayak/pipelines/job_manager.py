"""
KrishiSahayak - Training Job Manager (Refactored)

This module provides the TrainingJobManager, a high-level supervisor class that
manages the entire lifecycle of a single training job. It handles resource
monitoring, status tracking, signal handling, and orchestration of the training
runner.
"""
from __future__ import annotations
import json
import logging
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Protocol

import psutil

# --- Canonical Project Imports ---
from krishisahayak.config.schemas import JobType, TrainingConfig, PathsConfig
from krishisahayak.training.runners import BaseRunner, ClassificationRunner, GanRunner

logger = logging.getLogger(__name__)

# --- Enums and Dataclasses ---

class TrainingStatus(str, Protocol):
    """Defines the status of a training job."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingCallback(Protocol):
    """Defines a formal protocol for callbacks."""
    def on_training_start(self, manager: "TrainingJobManager") -> None: ...
    def on_training_end(self, manager: "TrainingJobManager", status: TrainingStatus, error: Exception | None) -> None: ...
    def on_checkpoint_save(self, manager: "TrainingJobManager", checkpoint_path: str) -> None: ...

@dataclass
class ResourceUsage:
    """Tracks resource usage during training."""
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    checkpoint_count: int = 0

    @property
    def duration_seconds(self) -> float:
        return (self.end_time or time.time()) - self.start_time
    
    @property
    def duration_formatted(self) -> str:
        duration = self.duration_seconds
        hours, rem = divmod(int(duration), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"


# --- Job Manager Class ---

class TrainingJobManager:
    """
    Supervises a training job, managing its lifecycle, resources, and state.
    This is the top tier of the training pipeline architecture.
    """
    
    def __init__(self, config: TrainingConfig, paths: PathsConfig, job_name: str) -> None:
        self.config = config
        self.paths = paths
        self.job_name = job_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.status = TrainingStatus.INITIALIZING
        self.resources = ResourceUsage()
        self.runner: BaseRunner | None = None
        self._shutdown_requested = False
        self.callbacks: list[TrainingCallback] = []

        # Bind signals for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        # CORRECTED: Fixed typo from self_signal_handler to self._signal_handler
        signal.signal(signal.SIGTERM, self._signal_handler)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Registers a callback object to receive training events."""
        self.callbacks.append(callback)
        self.logger.info(f"Registered callback: {callback.__class__.__name__}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handles shutdown signals gracefully."""
        if self._shutdown_requested:
            self.logger.warning("Multiple shutdown signals received, ignoring.")
            return
            
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.status = TrainingStatus.CANCELLED
        self._shutdown_requested = True
        
        # The main `run` loop's finally block will handle cleanup.
        # This approach is generally more robust than calling sys.exit() here.

    @contextmanager
    def resource_monitoring(self) -> Any:
        """Context manager for comprehensive resource monitoring."""
        # Check if monitoring is enabled in the config (assuming it's nested)
        monitoring_enabled = self.config.model_dump().get("monitoring", {}).get("enable_resource_monitoring", True)
        if not monitoring_enabled:
            yield
            return
            
        process = psutil.Process()
        self.resources.initial_memory_mb = process.memory_info().rss / (1024**2)
        self.logger.info(f"Starting resource monitoring. Initial memory: {self.resources.initial_memory_mb:.2f}MB")
        try:
            yield
        finally:
            self.resources.peak_memory_mb = max(self.resources.peak_memory_mb, process.memory_info().rss / (1024**2))
            self.resources.end_time = time.time()

    def _validate_and_create_dirs(self) -> None:
        """Validates resources and creates directories before starting training."""
        self.logger.info("Creating output directories...")
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Assuming an output_dir for job summaries exists in the paths config
        if hasattr(self.paths, 'output_dir'):
            self.paths.output_dir.mkdir(parents=True, exist_ok=True)


    def _create_runner(self) -> BaseRunner:
        """Factory method to get the appropriate runner for the job type."""
        runner_map = {
            JobType.CLASSIFICATION: ClassificationRunner,
            JobType.GAN: GanRunner,
            # Add other runner types here as they are implemented
        }
        runner_class = runner_map.get(self.config.type)
        if not runner_class:
            raise ValueError(f"Unknown job type: '{self.config.type}'. Available: {list(runner_map.keys())}")
        
        # REFACTORED: Pass the validated Pydantic objects directly to the runner.
        return runner_class(config=self.config, paths=self.paths, job_name=self.job_name)

    def run(self) -> None:
        """Executes the training job with comprehensive monitoring and control."""
        final_result: Dict[str, Any] | None = None
        error: Exception | None = None
        try:
            self.logger.info(f"Initializing training job: '{self.job_name}'")
            self._validate_and_create_dirs()
            self.runner = self._create_runner()
            # self._execute_callbacks("on_training_start")
            
            with self.resource_monitoring():
                self.status = TrainingStatus.RUNNING
                final_result = self.runner.run() # The runner now returns test metrics
                if not self._shutdown_requested:
                    self.status = TrainingStatus.COMPLETED
                    self.logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            self.status = TrainingStatus.CANCELLED
            self.logger.warning("Training cancelled by user (KeyboardInterrupt).")
        except Exception as e:
            self.status = TrainingStatus.FAILED
            error = e
            self.logger.error(f"Training job failed: {e}", exc_info=True)
        finally:
            # self._execute_callbacks("on_training_end", status=self.status, error=error)
            self._cleanup(result=final_result)

    def _cleanup(self, result: Dict[str, Any] | None) -> None:
        """Performs cleanup actions, including saving final job state."""
        self.logger.info(f"Job '{self.job_name}' finished with status: '{self.status.value}'. Cleaning up.")
        self._log_resource_summary()
        
        output_dir = getattr(self.paths, 'output_dir', self.paths.output_root / "jobs")
        output_dir.mkdir(parents=True, exist_ok=True)
        state_file = output_dir / f"job_{self.job_name}_final_state.json"
        
        try:
            state_data = {
                "job_name": self.job_name,
                "final_status": self.status.value,
                "experiment_name": self.config.experiment_name,
                "resources": self.resources.__dict__,
                "result_metrics": result,
                "end_time_utc": datetime.utcnow().isoformat(),
            }
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=4)
            self.logger.info(f"Final state saved to: {state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save final state: {e}", exc_info=True)
    
    def _log_resource_summary(self) -> None:
        """Logs a comprehensive resource usage summary."""
        self.logger.info(
            f"--- Resource Usage Summary for Job '{self.job_name}' ---\n"
            f"  Status: {self.status.value}\n"
            f"  Duration: {self.resources.duration_formatted}\n"
            f"  Peak Memory: {self.resources.peak_memory_mb:.2f}MB\n"
            f"  Checkpoints Saved: {self.resources.checkpoint_count}"
        )