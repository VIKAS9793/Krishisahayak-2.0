# src/krishi_sahayak/utils/data_utils.py
"""
Shared utility functions for data handling across the KrishiSahayak project.
"""
from typing import Any, Dict, List, Optional

import torch


def collate_and_filter_none(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    A custom collate function that filters out None values from a batch.
    This is required to handle cases where a dataset returns None for a
    corrupt or missing image, preventing training crashes.
    """
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return {}  # Return an empty dict if the entire batch is bad
    return torch.utils.data.dataloader.default_collate(valid_batch)