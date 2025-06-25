"""
Confidence-based Model Fallback System (Refactored)

Provides a robust and efficient implementation for managing fallback between
a primary and secondary model based on prediction confidence. This module is
decoupled from data structures by using a configurable batch processor.
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Defines the expected signature for a function that processes a raw dataloader batch
BatchProcessorCallable = Callable[[Any], Tuple[Dict[str, torch.Tensor], torch.Tensor]]


class ConfidenceThreshold(nn.Module):
    """
    Implements an efficient, per-sample confidence-based model fallback.

    If a sample's prediction confidence from the primary model is below a
    threshold, this module runs that specific sample through a secondary
    (fallback) model and uses its result instead.

    This module is stateful for statistics tracking and is intended for evaluation,
    not for end-to-end training.
    """

    def __init__(
        self,
        primary_model: nn.Module,
        secondary_model: nn.Module,
        threshold: float = 0.7,
        temperature: float = 1.0
    ) -> None:
        super().__init__()
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Confidence threshold must be in the range (0.0, 1.0].")

        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.threshold = threshold
        self.temperature = temperature
        self.reset_stats()

    def forward(
        self,
        primary_input: Dict[str, torch.Tensor],
        fallback_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs a forward pass with fallback logic.

        Args:
            primary_input: The input for the primary model (e.g., a dict for multi-modal).
            fallback_input: The input for the secondary model (e.g., an RGB tensor).

        Returns:
            A tuple containing the final logits tensor and metadata about the fallback operation.
        """
        batch_size = fallback_input.shape[0]
        self._total_count += batch_size

        # Get confidence scores from the primary model
        primary_logits = self.primary_model(primary_input)
        probs = F.softmax(primary_logits / self.temperature, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)

        # Identify indices of samples that fall below the confidence threshold
        fallback_indices = (max_probs < self.threshold).nonzero(as_tuple=True)[0]

        # If all samples are confident, return primary model's results directly
        if fallback_indices.numel() == 0:
            return primary_logits, {'used_fallback': False, 'fallback_count': 0}

        # Run the secondary model ONLY on the low-confidence subset
        low_confidence_fallback_input = fallback_input[fallback_indices]
        secondary_logits_subset = self.secondary_model(low_confidence_fallback_input)

        # Combine results: start with primary logits, then overwrite the low-confidence ones
        final_logits = primary_logits.clone()
        final_logits[fallback_indices] = secondary_logits_subset

        self._fallback_count += fallback_indices.numel()
        metadata = {'used_fallback': True, 'fallback_count': fallback_indices.numel()}
        return final_logits, metadata

    def get_fallback_stats(self) -> Dict[str, float]:
        """Returns the cumulative fallback statistics since the last reset."""
        ratio = self._fallback_count / self._total_count if self._total_count > 0 else 0.0
        return {
            'fallback_ratio': ratio,
            'fallback_count': float(self.fallback_count),
            'total_samples': float(self.total_count)
        }

    def reset_stats(self) -> None:
        """Resets the internal fallback counters."""
        self._fallback_count = 0
        self._total_count = 0

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, device: str, batch_processor: BatchProcessorCallable) -> Dict[str, float]:
        """
        Runs a full evaluation loop on a dataloader, managing state automatically.

        Args:
            dataloader: The dataloader to evaluate on.
            device: The torch device to run the models on.
            batch_processor: A function that takes a raw batch from the dataloader
                             and returns a tuple of (primary_input, fallback_input).
                             This decouples the module from the dataset's structure.
        """
        self.to(torch.device(device))
        self.eval()
        self.reset_stats()

        for batch in tqdm(dataloader, desc="Running Confidence Evaluation"):
            try:
                # Use the provided function to unpack the batch
                primary_input_cpu, fallback_input_cpu = batch_processor(batch)
                
                # Move tensors to the target device
                primary_input = {k: v.to(device) for k, v in primary_input_cpu.items() if isinstance(v, torch.Tensor)}
                fallback_input = fallback_input_cpu.to(device)

                self.forward(primary_input, fallback_input)
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Skipping a batch due to a preparation error: {e}")
                continue

        return self.get_fallback_stats()