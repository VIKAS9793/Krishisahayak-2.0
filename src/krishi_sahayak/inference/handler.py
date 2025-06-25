"""High-level handler for orchestrating the inference workflow."""

import logging
from pathlib import Path
from typing import Dict, Any, Iterator, List
import torch
from torch.utils.data import DataLoader

from .model_loader import ModelLoader
from .predictor import Predictor
from .data_loader import InferenceDataset, create_transforms

logger = logging.getLogger(__name__)

class InferenceHandler:
    """Orchestrates the full inference pipeline."""
    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        logger.info("Initializing model loader...")
        loader = ModelLoader(checkpoint_path, device)
        model = loader.get_model()
        self.class_names = loader.get_class_names()
        self.model_config = loader.get_model_config()
        self.stats = loader.get_preprocessing_stats()

        logger.info("Initializing predictor...")
        self.predictor = Predictor(model, self.class_names, self.device)
        self.is_hybrid = 'ms' in self.model_config.get('streams', {})

    def run_single(self, image_path: Path, nir_image_path: Path | None, top_k: int) -> Dict[str, Any]:
        """Runs inference on a single image."""
        rgb_transform, nir_transform = create_transforms(self.stats)
        
        dataset = InferenceDataset(
            file_paths=[image_path],
            rgb_transform=rgb_transform,
            is_hybrid=self.is_hybrid,
            nir_transform=nir_transform,
            nir_lookup_paths=[nir_image_path] if nir_image_path else None
        )
        
        item = dataset[0]
        if item is None:
            raise ValueError(f"Failed to process image: {image_path}")

        # Create a batch of 1
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in item.items() if k != 'path'}

        predictions = self.predictor.predict_batch(inputs, top_k)[0]
        
        return {
            "model_checkpoint": str(self.checkpoint_path),
            "image_path": str(image_path),
            "predictions": predictions,
        }

    def run_batch(self, input_dir: Path, batch_size: int, num_workers: int, nir_suffix: str, top_k: int) -> Iterator[Dict[str, Any]]:
        """Runs inference on a directory of images."""
        rgb_transform, nir_transform = create_transforms(self.stats)
        
        dataset = InferenceDataset.from_directory(
            input_dir, rgb_transform, self.is_hybrid, nir_transform, nir_suffix
        )
        if not dataset:
            logger.warning(f"No valid images found in {input_dir}.")
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=InferenceDataset.collate_fn)

        for batch in dataloader:
            if not batch:
                continue
            
            paths = batch.pop('path')
            inputs = {key: tensor.to(self.device) for key, tensor in batch.items()}
            
            batch_predictions = self.predictor.predict_batch(inputs, top_k)

            for i, path_str in enumerate(paths):
                yield {
                    "model_checkpoint": str(self.checkpoint_path),
                    "image_path": path_str,
                    "predictions": batch_predictions[i],
                }
