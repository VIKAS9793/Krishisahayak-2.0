"""
KrishiSahayak - Advanced Unified Model Core (Refactored)

This module provides a state-of-the-art, unified model framework. This refactored
version resolves architectural flaws by encapsulating the model logic in a
dedicated inner module and using correct, conventional import paths.
"""
from __future__ import annotations
import logging
from collections.abc import Callable
from typing import Any, Dict, List, Literal

import timm
import torch
import torch.nn as nn
from pydantic import BaseModel as PydanticBaseModel, Field

# Import BaseModel from the parent directory's base module
from ..base import BaseModel, BaseModelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: Pydantic Configuration Models
# =============================================================================
class StreamConfig(PydanticBaseModel):
    channels: int = Field(..., gt=0)
    adapter_out: int | None = Field(None, gt=0)
    pretrained: bool = False

class FusionConfig(PydanticBaseModel):
    method: Literal["concat", "add", "attention", "cross_attention"] = "concat"
    num_heads: int = Field(8, gt=0)
    dropout_rate: float = Field(0.1, ge=0, le=1)

class ModelConfig(PydanticBaseModel):
    backbone_name: str
    streams: Dict[str, StreamConfig]
    fusion: FusionConfig | None = None
    classifier_hidden_dim: int | None = Field(None, gt=0)
    classifier_dropout: float = Field(0.0, ge=0, le=1)


# =============================================================================
# PART 2: Core Architectural Implementation (_UnifiedModelCore)
# =============================================================================

class _UnifiedModelCore(nn.Module):
    """Internal nn.Module encapsulating the complete architecture and forward logic."""
    def __init__(self, model_config: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.config = model_config
        self.feature_dims = self._get_feature_dims()
        
        self.backbones = nn.ModuleDict()
        self.adapters = nn.ModuleDict()

        for name, stream_cfg in self.config.streams.items():
            in_chans = stream_cfg.adapter_out or stream_cfg.channels
            self.backbones[name] = timm.create_model(
                self.config.backbone_name, pretrained=stream_cfg.pretrained,
                num_classes=0, in_chans=in_chans
            )
            if stream_cfg.adapter_out is not None:
                self.adapters[name] = nn.Conv2d(stream_cfg.channels, stream_cfg.adapter_out, kernel_size=1)
        
        self.fusion = self._create_fusion_layer()
        self.classifier = self._create_classifier(num_classes)

    def _get_feature_dims(self) -> Dict[str, int]:
        """Dynamically determines feature dimensions from the backbone."""
        try:
            temp_backbone = timm.create_model(self.config.backbone_name, num_classes=0)
            return {name: temp_backbone.num_features for name in self.config.streams}
        except Exception as e:
            raise RuntimeError(f"Failed to infer feature dims for '{self.config.backbone_name}': {e}") from e

    def _create_fusion_layer(self) -> nn.Module | None:
        """Factory for creating the feature fusion layer."""
        if len(self.config.streams) <= 1:
            return None
        
        fusion_cfg = self.config.fusion or FusionConfig()
        output_dim = list(self.feature_dims.values())[0]

        if fusion_cfg.method == 'cross_attention':
            from .fusion import CrossAttentionFusion # Lazy import for complex dependency
            return CrossAttentionFusion(self.feature_dims, output_dim, fusion_cfg.num_heads, fusion_cfg.dropout_rate)
        
        total_feature_dim = sum(self.feature_dims.values())
        return nn.Linear(total_feature_dim, output_dim)

    def _create_classifier(self, num_classes: int) -> nn.Module:
        """Creates the final classification head."""
        input_dim = list(self.feature_dims.values())[0]
        if self.config.classifier_hidden_dim:
            return nn.Sequential(
                nn.Linear(input_dim, self.config.classifier_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.classifier_dropout),
                nn.Linear(self.config.classifier_hidden_dim, num_classes)
            )
        return nn.Linear(input_dim, num_classes)

    def forward(self, batch: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs the forward pass for the multi-modal model."""
        if not isinstance(batch, dict):
            stream_name = next(iter(self.config.streams.keys()))
            batch = {stream_name: batch}

        features = {name: self.backbones[name](self.adapters.get(name, nn.Identity())(x)) for name, x in batch.items()}

        if self.fusion:
            if isinstance(self.fusion, nn.Linear):
                fused_features = self.fusion(torch.cat(list(features.values()), dim=1))
            else:
                fused_features = self.fusion(features)
        else:
            stream_name = next(iter(self.config.streams.keys()))
            fused_features = features[stream_name]

        return self.classifier(fused_features)


# =============================================================================
# PART 3: Main Unified LightningModule
# =============================================================================

class UnifiedModel(BaseModel):
    """
    The LightningModule wrapper for the unified model. This class is responsible
    for the training loop, optimization, and metrics, while delegating the
    architecture definition to the internal `_UnifiedModelCore`.
    """
    def __init__(self, model_config: ModelConfig, base_config: BaseModelConfig, num_classes: int, **kwargs: Any) -> None:
        # Create the underlying nn.Module architecture first.
        core_model = _UnifiedModelCore(model_config, num_classes)
        
        # Now, initialize the parent BaseModel with the core model and other required components.
        # This satisfies the contract of the BaseModel.
        super().__init__(
            model=core_model,
            num_classes=num_classes,
            config=base_config,
            **kwargs
        )
        self.model_config = model_config

    def get_feature_maps(self, batch: Dict[str, torch.Tensor], target_layers: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate feature maps using temporary forward hooks.
        This is useful for explainability methods like Grad-CAM.
        """
        feature_maps: Dict[str, torch.Tensor] = {}
        hooks: List[Callable] = []

        def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor, name: str) -> None:
            feature_maps[name] = output.detach()

        try:
            # self.model is the _UnifiedModelCore instance inherited from BaseModel
            for stream_name, layer_name in target_layers.items():
                try:
                    module = dict(self.model.backbones[stream_name].named_modules())[layer_name]
                    hook = module.register_forward_hook(
                        lambda m, i, o, name=f"{stream_name}_{layer_name}": hook_fn(m, i, o, name)
                    )
                    hooks.append(hook)
                except KeyError:
                    logger.warning(f"Layer '{layer_name}' not found in backbone '{stream_name}'.")

            if not hooks:
                raise ValueError("No valid layers were provided for hooking.")
            
            with torch.no_grad():
                self.forward(batch)
        finally:
            # CRITICAL: Always remove hooks after use to prevent memory leaks.
            for hook in hooks:
                hook.remove()

        return feature_maps