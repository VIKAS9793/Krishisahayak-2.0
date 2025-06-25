"""
KrishiSahayak - Advanced Hybrid Model Wrapper (Production-Ready)

This module provides a unified, high-level interface for a hybrid RGB+MS model.
It orchestrates on-the-fly NIR generation, confidence-based fallbacks, and
fusion validation into a single, robust nn.Module.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make GAN imports optional for flexibility
try:
    from krishi_sahayak.models.pix2pix import Pix2PixGAN
except ImportError:
    Pix2PixGAN = None  # Define as None if not available

from krishi_sahayak.models.utils import ConfidenceThreshold, FusionValidator

logger = logging.getLogger(__name__)

class HybridModel(nn.Module):
    """
    A unified hybrid model that orchestrates advanced inference logic by composing
    other models. It follows a consistent Dependency Injection pattern.
    """
    def __init__(
        self,
        rgb_model: nn.Module,
        fusion_model: Optional[nn.Module],
        gan_model: Optional[nn.Module] = None, # Changed from checkpoint path to model instance
        confidence_threshold: float = 0.7,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.rgb_model = rgb_model
        self.fusion_model = fusion_model
        self.gan = gan_model

        if self.gan:
            self.gan.eval()
            logger.info("GAN model provided for NIR generation.")
        
        # The ConfidenceThreshold and FusionValidator are internal components
        # composed of the injected models.
        if self.fusion_model:
            self.confidence_model: Optional[ConfidenceThreshold] = ConfidenceThreshold(
                primary_model=self.fusion_model,
                secondary_model=self.rgb_model,
                threshold=confidence_threshold
            )
            self.validator: Optional[FusionValidator] = FusionValidator(
                rgb_model=self.rgb_model,
                fusion_model=self.fusion_model,
                nir_simulator=self.generate_nir if self.gan else None
            )
        else:
            self.confidence_model = None
            self.validator = None
            logger.warning("No fusion_model provided. HybridModel will operate in RGB-only mode.")
        self.to(self.device)

    @torch.no_grad()
    def generate_nir(self, rgb: torch.Tensor) -> torch.Tensor:
        """Generates a synthetic NIR channel from an RGB image using the GAN."""
        if self.gan is None:
            raise RuntimeError("Cannot generate NIR: GAN model is not available.")
        # Assumes GAN expects input normalized to [-1, 1] and outputs in the same range
        rgb_norm = (rgb * 2) - 1
        generated_nir = self.gan.generator(rgb_norm) # GAN is already on the correct device
        # Denormalize output back to [0, 1]
        return (generated_nir + 1) / 2

    def forward(
        self,
        rgb: torch.Tensor,
        nir: Optional[torch.Tensor] = None,
        return_metadata: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs a forward pass with optional NIR generation and confidence fallback.
        """
        rgb = rgb.to(self.device)
        metadata: Dict[str, Any] = {'used_generated_nir': False, 'used_fallback': False}

        if self.fusion_model and self.confidence_model:
            # If NIR is not provided, attempt to generate it with the GAN
            if nir is None:
                if not self.gan:
                    raise ValueError("Fusion model requires NIR input, but real NIR was not provided and no GAN is available.")
                nir = self.generate_nir(rgb)
                metadata['used_generated_nir'] = True
            
            primary_input = {'rgb': rgb, 'ms': nir.to(self.device)}
            logits, confidence_meta = self.confidence_model(
                primary_input=primary_input,
                fallback_input=rgb
            )
            metadata.update(confidence_meta)
        else:
            # If no fusion model, operate in simple RGB-only mode
            logits = self.rgb_model(rgb)
        
        return (logits, metadata) if return_metadata else logits
    
    def validate_fusion(self, dataloader: DataLoader, batch_processor: Callable) -> Optional[Dict[str, Any]]:
        if not self.validator:
            logger.warning("FusionValidator not available. Skipping validation.")
            return None
        logger.info("Running fusion validation...")
        return self.validator.run(dataloader, device=str(self.device), batch_processor=batch_processor)
        
    def get_fallback_stats(self) -> Optional[Dict[str, float]]:
        if not self.confidence_model: return None
        return self.confidence_model.get_fallback_stats()

    def to(self, *args: Any, **kwargs: Any) -> "HybridModel":
        """Overrides `.to()` to move all sub-models to the specified device."""
        super().to(*args, **kwargs)
        # Ensure all composed modules are moved to the correct device
        if hasattr(self, 'rgb_model'): self.rgb_model.to(*args, **kwargs)
        if hasattr(self, 'fusion_model') and self.fusion_model: self.fusion_model.to(*args, **kwargs)
        if hasattr(self, 'gan') and self.gan: self.gan.to(*args, **kwargs)
        if hasattr(self, 'confidence_model') and self.confidence_model: self.confidence_model.to(*args, **kwargs)
        # Update the device attribute after moving
        try: self.device = next(self.parameters()).device
        except StopIteration: pass # Handle case with no parameters
        return self
