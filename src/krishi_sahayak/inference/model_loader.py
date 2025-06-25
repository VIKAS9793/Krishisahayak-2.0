"""Handles the secure loading and validation of model checkpoints."""

import torch
from pathlib import Path
from typing import Dict, Any, List

# Assuming this is the project's core model definition
from krishi_sahayak.models.core import UnifiedModel

class ModelLoader:
    """
    Safely loads a model checkpoint from disk, validates its contents,
    and instantiates the model.
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
        required_keys = ['model_config', 'class_names', 'state_dict']
        for key in required_keys:
            if key not in self.checkpoint:
                raise KeyError(f"Checkpoint is missing required key: '{key}'.")

    def get_model(self) -> UnifiedModel:
        """
        Instantiates the model and loads the state dictionary.
        This is safer than unpickling a whole model object.
        """
        num_classes = len(self.get_class_names())
        model = UnifiedModel(
            num_classes=num_classes,
            model_config=self.get_model_config()
        )
        model.load_state_dict(self.checkpoint['state_dict'])
        return model.to(self.device).eval()

    def get_class_names(self) -> List[str]:
        return self.checkpoint['class_names']

    def get_model_config(self) -> Dict[str, Any]:
        return self.checkpoint['model_config']

    def get_preprocessing_stats(self) -> Dict[str, Any]:
        return self.checkpoint.get('preprocessing_stats', {})