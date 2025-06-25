"""
Data loading and preprocessing components for inference.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)

def create_transforms(
    stats: Dict[str, Any]
) -> Tuple[Callable, Optional[Callable]]:
    """
    Creates the image transformation pipelines.

    Args:
        stats: A dictionary containing preprocessing stats like image_size,
               mean, and std.

    Returns:
        A tuple containing the RGB transform and an optional NIR transform.
    """
    if not stats:
        # Fallback to reasonable defaults if stats are missing
        stats = {'image_size': [256, 256], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        logger.warning(f"Preprocessing stats not found in checkpoint. Using default transforms.")

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
    Handles both RGB and hybrid (RGB + NIR) models and gracefully skips
    corrupt files or items with missing data.
    """
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
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
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
            }

            if self.is_hybrid:
                # Determine NIR path from lookup list or by suffix convention
                nir_path = self.nir_lookup_paths[idx] or rgb_path.with_name(f"{rgb_path.stem}{self.nir_suffix}")
                
                if not nir_path.exists():
                    logger.warning(f"Skipping {rgb_path.name}: Cannot find matching NIR file at {nir_path}")
                    return None
                
                nir_image = Image.open(nir_path)
                item['ms'] = self.nir_transform(nir_image)
            
            return item

        except (IOError, UnidentifiedImageError, SyntaxError) as e:
            logger.warning(f"Skipping corrupt or unreadable image at {rgb_path}: {e}")
            return None

    @staticmethod
    def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Custom collate_fn to filter out None values from a batch.
        This safely handles items that failed to load in __getitem__.
        """
        # Filter out None items from the batch
        valid_batch = [item for item in batch if item is not None]
        
        # If the whole batch was invalid, return an empty dict
        if not valid_batch:
            return {}
        
        # Use the default collate function on the cleaned batch
        # Assuming torch is available in the environment
        import torch
        return torch.utils.data.dataloader.default_collate(valid_batch)