# src/krishi_sahayak/inference/data_loader.py
"""
Data loading and preprocessing components for inference.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms as T

# REFACTORED: Import the canonical collate function from the central utils package.
from krishi_sahayak.utils.data_utils import collate_and_filter_none

logger = logging.getLogger(__name__)

def create_transforms(
    stats: Dict[str, Any]
) -> Tuple[Callable, Optional[Callable]]:
    """
    Creates the image transformation pipelines from saved model stats.
    """
    if not stats:
        stats = {'image_size': [256, 256], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        logger.warning("Preprocessing stats not found in checkpoint. Using default transforms.")

    size = tuple(stats['image_size'])
    rgb_transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=stats['mean'], std=stats['std'])
    ])
    nir_transform = T.Compose([T.Resize(size), T.ToTensor()])
    return rgb_transform, nir_transform


class InferenceDataset(Dataset):
    """
    A flexible PyTorch Dataset for loading inference data from file paths.
    """
    collate_fn = collate_and_filter_none

    def __init__(
        self,
        file_paths: List[Path],
        rgb_transform: Callable,
        is_hybrid: bool,
        nir_transform: Optional[Callable] = None,
        nir_suffix: str = '_nir.tif',
        nir_lookup_paths: Optional[List[Optional[Path]]] = None,
    ):
        self.file_paths = file_paths
        self.rgb_transform = rgb_transform
        self.is_hybrid = is_hybrid
        self.nir_transform = nir_transform
        self.nir_suffix = nir_suffix
        self.nir_lookup_paths = nir_lookup_paths or [None] * len(file_paths)

        if is_hybrid and not nir_transform:
            raise ValueError("nir_transform must be provided for hybrid models.")

    @classmethod
    def from_directory(
        cls,
        input_dir: Path,
        rgb_transform: Callable,
        is_hybrid: bool,
        nir_transform: Optional[Callable],
        nir_suffix: str = '_nir.tif',
    ) -> 'InferenceDataset':
        """Factory method to create a dataset from a directory of images."""
        image_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])
        return cls(image_paths, rgb_transform, is_hybrid, nir_transform, nir_suffix)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        rgb_path = self.file_paths[idx]
        try:
            rgb_image = Image.open(rgb_path).convert('RGB')
            item: Dict[str, Any] = {
                'rgb': self.rgb_transform(rgb_image),
                'path': str(rgb_path),
                'original_image': T.ToTensor()(rgb_image), # Keep original for visualization
            }

            if self.is_hybrid:
                nir_path = self.nir_lookup_paths[idx] or rgb_path.with_name(f"{rgb_path.stem}{self.nir_suffix}")
                if not nir_path.exists():
                    logger.warning(f"Skipping {rgb_path.name}: Cannot find matching NIR file at {nir_path}")
                    return None
                
                nir_image = Image.open(nir_path).convert('L')
                if self.nir_transform:
                    item['ms'] = self.nir_transform(nir_image)
            
            return item

        except (IOError, UnidentifiedImageError, SyntaxError) as e:
            logger.warning(f"Skipping corrupt or unreadable image at {rgb_path}: {e}")
            return None