"""
Manages the lifecycle of a training job, including resource monitoring,
status tracking, and graceful shutdown handling.
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
from typing import Any, Callable, Protocol

import psutil

# The manager assumes a contract with the runner, including a stop() method
# and a run() method that returns a dictionary of results.
from .runners import BaseRunner, ClassificationRunner, GanRunner
from .schemas import JobType, TrainingConfig, TrainingStatus

logger = logging.getLogger(__name__)

# --- Enhanced Callback Protocol ---
class TrainingCallback(Protocol):
    """
    Defines a formal protocol for callbacks, enabling more structured and type-safe extensions.
    """
    def on_training_start(self, manager: "TrainingJobManager") -> None: ...
    def on_training_end(self, manager: "TrainingJobManager", status: TrainingStatus, error: Exception | None) -> None: ...
    def on_checkpoint_save(self, manager: "TrainingJobManager", checkpoint_path: str) -> None: ...


# --- Resource Tracking ---
@dataclass
class ResourceUsage:
    """Track resource usage during training with enhanced metrics."""
    initial_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    checkpoint_count: int = 0

    @property
    def duration_seconds(self) -> float:
        """Calculate training duration."""
        return (self.end_time or time.time()) - self.start_time
    
    @property
    def duration_formatted(self) -> str:
        """Return human-readable duration."""
        duration = self.duration_seconds
        hours, rem = divmod(int(duration), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

# --- Job Manager ---
class TrainingJobManager:
    """Enhanced training job manager with improved monitoring and control."""
    
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.status = TrainingStatus.INITIALIZING
        self.resources = ResourceUsage()
        self.runner: BaseRunner | None = None
        self._shutdown_requested = False
        self.callbacks: list[TrainingCallback] = []

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self_signal_handler)

    def register_callback(self, callback: TrainingCallback) -> None:
        """Registers a callback object to receive training events."""
        self.callbacks.append(callback)
        self.logger.info(f"Registered callback: {callback.__class__.__name__}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        if self._shutdown_requested:
            self.logger.warning("Multiple shutdown signals received, ignoring.")
            return
            
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self._shutdown_requested = True
        
        # Safely attempt to stop the runner, assuming it adheres to the contract.
        # The runner's stop() method should handle the graceful shutdown of the trainer.
        if self.runner and hasattr(self.runner, 'stop'):
            try:
                self.runner.stop()
                self.logger.info("Runner stop signal sent successfully.")
            except Exception as e:
                self.logger.error(f"Error while calling runner.stop(): {e}", exc_info=True)
        
        # Update status and exit. The main `run` loop's finally block will handle cleanup.
        self.status = TrainingStatus.CANCELLED
        # It's often better to let the main exception handling flow deal with exiting.
        # sys.exit(0) # Or could raise a specific exception here.

    @contextmanager
    def resource_monitoring(self) -> Any:
        """Context manager for comprehensive resource monitoring."""
        if not self.config.monitoring.enable_resource_monitoring:
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
            # The summary is logged at the end of the cleanup phase for a complete picture.

    def _validate_setup(self) -> None:
        """Validate resources and create directories before starting training."""
        self.logger.info("Validating setup and creating directories...")
        for path in [self.config.paths.log_dir, self.config.paths.checkpoint_dir, self.config.paths.output_dir]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

    def _create_runner(self) -> BaseRunner:
        """Create and configure the appropriate runner based on job type."""
        runner_map = {
            JobType.CLASSIFICATION: ClassificationRunner,
            JobType.GAN: GanRunner,
        }
        runner_class = runner_map.get(self.config.type)
        if not runner_class:
            raise ValueError(f"Unknown job type: {self.config.type}. Available: {list(runner_map.keys())}")
        
        # Assumes TrainingConfig is compatible with the runner's expected ExperimentConfig
        return runner_class(config=self.config.model_dump())

    def _execute_callbacks(self, event_name: str, **kwargs: Any) -> None:
        """Dynamically execute registered callback methods for a given event."""
        for callback in self.callbacks:
            if hasattr(callback, event_name):
                try:
                    method_to_call = getattr(callback, event_name)
                    method_to_call(self, **kwargs)
                except Exception as e:
                    self.logger.error(f"Callback error during '{event_name}': {e}", exc_info=True)

    def run(self) -> dict[str, Any]:
        """Execute the training job with comprehensive monitoring and control."""
        final_result = None
        try:
            self.logger.info(f"Initializing training job: {self.config.name} ({self.config.type.value})")
            self._validate_setup()
            self.runner = self._create_runner()
            self._execute_callbacks("on_training_start")
            
            with self.resource_monitoring():
                self.status = TrainingStatus.RUNNING
                # The runner's run method is expected to return a dictionary of results
                final_result = self.runner.run()
                if not self._shutdown_requested:
                    self.status = TrainingStatus.COMPLETED
                    self.logger.info("Training completed successfully!")
            
            return {"status": self.status.value, "resources": self.resources, "result": final_result}
        
        except KeyboardInterrupt:
            # This is a fallback if the signal handler doesn't cause a clean exit
            self.status = TrainingStatus.CANCELLED
            self.logger.warning("Training cancelled by user (KeyboardInterrupt).")
            raise
        except Exception as e:
            self.status = TrainingStatus.FAILED
            self.logger.error(f"Training job failed: {e}", exc_info=True)
            self._execute_callbacks("on_training_end", status=self.status, error=e)
            raise
        finally:
            self._execute_callbacks("on_training_end", status=self.status, error=sys.exc_info()[1])
            self._cleanup(result=final_result)
        
        # This path should ideally not be reached
        return {"status": self.status.value, "resources": self.resources, "result": final_result}

    def _cleanup(self, result: dict[str, Any] | None) -> None:
        """Perform cleanup actions, including saving final job state and resource summary."""
        self.logger.info(f"Job '{self.config.name}' finished with status: {self.status.value}. Cleaning up.")
        
        # Log the final resource summary
        self._log_resource_summary()
        
        if self.config.paths.output_dir:
            state_file = self.config.paths.output_dir / f"job_{self.config.name}_final_state.json"
            try:
                state_data = {
                    "job_name": self.config.name,
                    "final_status": self.status.value,
                    "resources": {
                        "duration_seconds": self.resources.duration_seconds,
                        "duration_formatted": self.resources.duration_formatted,
                        "peak_memory_mb": self.resources.peak_memory_mb,
                        "checkpoints_saved": self.resources.checkpoint_count,
                    },
                    "result_metrics": result,
                    "end_time_utc": datetime.utcnow().isoformat(),
                }
                with open(state_file, 'w') as f:
                    json.dump(state_data, f, indent=4)
                self.logger.info(f"Final state saved to: {state_file}")
            except Exception as e:
                self.logger.error(f"Failed to save final state: {e}", exc_info=True)
    
    def _log_resource_summary(self) -> None:
        """Log comprehensive resource usage summary."""
        self.logger.info(
            f"--- Resource Usage Summary for Job '{self.config.name}' ---\n"
            f"  Status: {self.status.value}\n"
            f"  Duration: {self.resources.duration_formatted}\n"
            f"  Peak Memory: {self.resources.peak_memory_mb:.2f}MB\n"
            f"  Checkpoints Saved: {self.resources.checkpoint_count}"
        )