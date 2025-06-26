"""
KrishiSahayak - Multi-Modal Fusion Validation Framework (Refactored)

This module provides a robust framework to empirically validate that a multi-modal
(RGB+NIR) model performs better than its single-modality counterparts.
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Defines the expected signature for a function that processes a raw dataloader batch
FusionBatchProcessor = Callable[[Any], Optional[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]]


class FusionValidator:
    """
    Validates the effectiveness of multi-modal fusion by comparing the performance
    of RGB-only, NIR-only, and fused models.
    """

    def __init__(
        self,
        rgb_model: nn.Module,
        fusion_model: nn.Module,
        nir_model: Optional[nn.Module] = None,
        nir_simulator: Optional[nn.Module] = None
    ) -> None:
        """Initializes the FusionValidator with the models to be compared."""
        self.rgb_model = rgb_model
        self.nir_model = nir_model
        self.fusion_model = fusion_model
        self.nir_simulator = nir_simulator
        self.stats: Dict[str, float] = {}
        self._reset()

    def _reset(self) -> None:
        """Resets all internal statistics to start a fresh validation run."""
        self.stats = {
            'rgb_correct': 0.0, 'nir_correct': 0.0, 'fusion_correct': 0.0,
            'rgb_confidence_sum': 0.0, 'nir_confidence_sum': 0.0, 'fusion_confidence_sum': 0.0,
            'nir_samples': 0.0, 'fusion_samples': 0.0, 'total_samples': 0.0
        }

    def _validate_batch(self, rgb: torch.Tensor, nir: Optional[torch.Tensor], target: torch.Tensor) -> None:
        """Validates a single batch of data and accumulates raw statistics."""
        batch_size = float(target.size(0))
        self.stats['total_samples'] += batch_size

        # --- RGB Model Evaluation ---
        rgb_logits = self.rgb_model(rgb)
        rgb_probs = F.softmax(rgb_logits, dim=-1)
        rgb_conf, rgb_pred = torch.max(rgb_probs, dim=-1)
        self.stats['rgb_correct'] += (rgb_pred == target).sum().item()
        self.stats['rgb_confidence_sum'] += rgb_conf.sum().item()

        # --- NIR Model Evaluation (if applicable) ---
        if self.nir_model and nir is not None:
            self.stats['nir_samples'] += batch_size
            nir_logits = self.nir_model(nir)
            nir_probs = F.softmax(nir_logits, dim=-1)
            nir_conf, nir_pred = torch.max(nir_probs, dim=-1)
            self.stats['nir_correct'] += (nir_pred == target).sum().item()
            self.stats['nir_confidence_sum'] += nir_conf.sum().item()

        # --- Fusion Model Evaluation (if applicable) ---
        # If real NIR is missing, try to simulate it
        if nir is None and self.nir_simulator is not None:
            nir = self.nir_simulator(rgb)
        
        if nir is not None:
            self.stats['fusion_samples'] += batch_size
            fusion_input = {'rgb': rgb, 'ms': nir}
            fusion_logits = self.fusion_model(fusion_input)
            fusion_probs = F.softmax(fusion_logits, dim=-1)
            fusion_conf, fusion_pred = torch.max(fusion_probs, dim=-1)
            self.stats['fusion_correct'] += (fusion_pred == target).sum().item()
            self.stats['fusion_confidence_sum'] += fusion_conf.sum().item()
    
    def run(self, dataloader: DataLoader, device: str, batch_processor: FusionBatchProcessor) -> Dict[str, Any]:
        """
        Runs a full validation loop on the provided dataloader.
        """
        self._reset()
        self.rgb_model.eval().to(device)
        if self.nir_model: self.nir_model.eval().to(device)
        self.fusion_model.eval().to(device)
        if self.nir_simulator: self.nir_simulator.eval().to(device)
        
        for batch in tqdm(dataloader, desc="Running Fusion Validation"):
            prepared_batch = batch_processor(batch)
            if prepared_batch is None:
                logger.warning("Skipping a batch due to a preparation error.")
                continue
            
            rgb, nir, target = prepared_batch
            rgb, target = rgb.to(device), target.to(device)
            if nir is not None:
                nir = nir.to(device)
            
            with torch.no_grad():
                self._validate_batch(rgb, nir, target)
        
        summary = self._get_summary()
        importance = self._get_modality_importance(summary)

        return {"summary": summary, "modality_importance": importance, "raw_stats": self.stats}
    
    def _get_summary(self) -> Dict[str, float]:
        """Calculates summary statistics from aggregated raw counts."""
        s = self.stats
        total = s['total_samples']
        if total == 0:
            return {}
        
        summary: Dict[str, float] = {
            'rgb_accuracy': s['rgb_correct'] / total,
            'rgb_avg_confidence': s['rgb_confidence_sum'] / total
        }
        if s['nir_samples'] > 0:
            summary['nir_accuracy'] = s['nir_correct'] / s['nir_samples']
            summary['nir_avg_confidence'] = s['nir_confidence_sum'] / s['nir_samples']
        if s['fusion_samples'] > 0:
            summary['fusion_accuracy'] = s['fusion_correct'] / s['fusion_samples']
            summary['fusion_avg_confidence'] = s['fusion_confidence_sum'] / s['fusion_samples']
            
        return summary
    
    def _get_modality_importance(self, summary: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the importance of fusion by measuring the performance
        uplift over the best single modality.
        """
        rgb_acc = summary.get('rgb_accuracy', 0.0)
        nir_acc = summary.get('nir_accuracy', 0.0)
        fusion_acc = summary.get('fusion_accuracy', 0.0)

        best_single_modality_acc = max(rgb_acc, nir_acc)
        
        if fusion_acc > 0 and best_single_modality_acc > 0:
            uplift = fusion_acc - best_single_modality_acc
            relative_uplift = (uplift / best_single_modality_acc) * 100 if best_single_modality_acc > 0 else 0.0
        else:
            uplift = 0.0
            relative_uplift = 0.0

        return {
            "best_single_modality_accuracy": best_single_modality_acc,
            "fusion_accuracy": fusion_acc,
            "absolute_uplift": uplift,
            "relative_uplift_percent": relative_uplift
        }