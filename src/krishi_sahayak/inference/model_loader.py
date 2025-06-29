# src/krishi_sahayak/inference/model_loader.py
"""Handles the secure loading and validation of model checkpoints."""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple

from krishi_sahayak.models import UnifiedModel, BaseModelConfig, ModelConfig

logger = logging.getLogger(__name__)

def _default_inference_batch_processor(batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], None]:
    """A default batch processor for inference where targets are not needed."""
    return batch, None

class ModelLoader:
    """
    Safely loads a model checkpoint from disk, validates its contents,
    and instantiates the model according to its required contract.
    """
    def __init__(self, checkpoint_path: Path, device: torch.device):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._validate_checkpoint()

    def _validate_checkpoint(self) -> None:
        """Ensures the checkpoint contains all necessary keys."""
        required_keys = ['hyper_parameters', 'class_names', 'state_dict']
        for key in required_keys:
            if key not in self.checkpoint:
                raise KeyError(f"Checkpoint is missing required key: '{key}'.")
        if "model_config" not in self.checkpoint['hyper_parameters']:
            raise KeyError("Checkpoint 'hyper_parameters' are missing 'model_config'.")

    def get_model(self) -> UnifiedModel:
        """Instantiates the model and loads the state dictionary."""
        hparams = self.checkpoint['hyper_parameters']
        num_classes = len(self.get_class_names())
        
        model_config = ModelConfig(**hparams['model_config'])
        base_config = BaseModelConfig(**hparams.get('base_config', {}))

        model = UnifiedModel(
            model_config=model_config,
            base_config=base_config,
            num_classes=num_classes,
            batch_processor=_default_inference_batch_processor
        )
        
        # Accommodate different checkpoint saving strategies
        state_dict = self.checkpoint['state_dict']
        # Remove "model." prefix if it exists from Lightning saving
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        logger.info("Model state dictionary loaded successfully.")
        return model.to(self.device).eval()

    def get_class_names(self) -> List[str]:
        return self.checkpoint['class_names']

    def get_model_config(self) -> Dict[str, Any]:
        return self.checkpoint['hyper_parameters']['model_config']

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        return self.checkpoint.get('hyper_parameters', {}).get('preprocessing_stats', {})

    def get_xai_config(self) -> Dict[str, Any]:
        """Extracts explainability configuration from the checkpoint."""
        return self.checkpoint.get('hyper_parameters', {}).get('xai_config', {})