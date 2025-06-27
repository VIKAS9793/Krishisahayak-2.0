# src/krishi_sahayak/pipelines/job_manager.py
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
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

from krishi_sahayak.config.schemas import JobType, PathsConfig, TrainingConfig
from .runners import BaseRunner, ClassificationRunner, GanRunner

logger = logging.getLogger(__name__)

class TrainingStatus(str, Enum):
    """Defines the status of a training job."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ResourceUsage:
    """Tracks resource usage during training."""
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return (self.end_time or time.time()) - self.start_time
    
    @property
    def duration_formatted(self) -> str:
        duration = self.duration_seconds
        hours, rem = divmod(int(duration), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

class TrainingJobManager:
    """Supervises a training job, managing its lifecycle, resources, and state."""
    
    def __init__(self, config: TrainingConfig, paths: PathsConfig, job_name: str) -> None:
        self.config = config
        self.paths = paths
        self.job_name = job_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.status: TrainingStatus = TrainingStatus.INITIALIZING
        self.resources = ResourceUsage()
        self.runner: BaseRunner | None = None
        self._shutdown_requested = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        if self._shutdown_requested: return
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.status = TrainingStatus.CANCELLED
        self._shutdown_requested = True

    @contextmanager
    def resource_monitoring(self) -> Any:
        process = psutil.Process()
        self.resources.initial_memory_mb = process.memory_info().rss / (1024**2)
        self.logger.info(f"Starting resource monitoring. Initial memory: {self.resources.initial_memory_mb:.2f}MB")
        try:
            yield
        finally:
            self.resources.peak_memory_mb = max(self.resources.peak_memory_mb, process.memory_info().rss / (1024**2))
            self.resources.end_time = time.time()

    def _validate_and_create_dirs(self) -> None:
        self.logger.info("Creating output directories...")
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_runner(self) -> BaseRunner:
        runner_map = {
            JobType.CLASSIFICATION: ClassificationRunner,
            JobType.GAN: GanRunner,
        }
        runner_class = runner_map.get(self.config.type)
        if not runner_class:
            raise ValueError(f"Unknown job type: '{self.config.type}'.")
        return runner_class(config=self.config, paths=self.paths, job_name=self.job_name)

    def run(self) -> None:
        final_result: Dict[str, Any] | None = None
        try:
            self.logger.info(f"Initializing training job: '{self.job_name}'")
            self._validate_and_create_dirs()
            self.runner = self._create_runner()
            
            with self.resource_monitoring():
                self.status = TrainingStatus.RUNNING
                final_result = self.runner.run()
                if not self._shutdown_requested:
                    self.status = TrainingStatus.COMPLETED
                    self.logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            self.status = TrainingStatus.CANCELLED
            self.logger.warning("Training cancelled by user (KeyboardInterrupt).")
        except Exception as e:
            self.status = TrainingStatus.FAILED
            self.logger.error(f"Training job failed: {e}", exc_info=True)
        finally:
            self._cleanup(result=final_result)

    def _cleanup(self, result: Dict[str, Any] | None) -> None:
        self.logger.info(f"Job '{self.job_name}' finished with status: '{self.status.value}'. Cleaning up.")
        self._log_resource_summary()
        
        state_file = self.paths.output_dir / f"job_{self.job_name}_final_state.json"
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
        self.logger.info(
            f"--- Resource Usage Summary for Job '{self.job_name}' ---\n"
            f"  Status: {self.status.value}\n"
            f"  Duration: {self.resources.duration_formatted}\n"
            f"  Peak Memory: {self.resources.peak_memory_mb:.2f}MB"
        )