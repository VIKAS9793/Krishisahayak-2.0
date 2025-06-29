# src/krishi_sahayak/models/core/unified_model.py
"""KrishiSahayak - Advanced Unified Model Core"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Tuple

import timm
import torch
import torch.nn as nn

from ..base import BaseModel, BaseModelConfig
from ..schemas import ModelConfig

logger = logging.getLogger(__name__)

class _UnifiedModelCore(nn.Module):
    def __init__(self, model_config: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.config = model_config
        self.feature_dims = self._get_feature_dims()
        self.backbones = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        for name, stream_cfg in self.config.streams.items():
            in_chans = stream_cfg.adapter_out or stream_cfg.channels
            self.backbones[name] = timm.create_model(self.config.backbone_name, pretrained=stream_cfg.pretrained, num_classes=0, in_chans=in_chans)
            if stream_cfg.adapter_out is not None: self.adapters[name] = nn.Conv2d(stream_cfg.channels, stream_cfg.adapter_out, kernel_size=1)
        self.fusion = self._create_fusion_layer()
        self.classifier = self._create_classifier(num_classes)

    def _get_feature_dims(self) -> Dict[str, int]:
        try:
            first_stream_cfg = next(iter(self.config.streams.values()))
            in_chans = first_stream_cfg.adapter_out or first_stream_cfg.channels
            temp_backbone = timm.create_model(self.config.backbone_name, num_classes=0, in_chans=in_chans)
            return {name: temp_backbone.num_features for name in self.config.streams}
        except Exception as e:
            raise RuntimeError(f"Failed to infer feature dims for '{self.config.backbone_name}': {e}") from e

    def _create_fusion_layer(self) -> nn.Module | None:
        if len(self.config.streams) <= 1: return None
        fusion_cfg = self.config.fusion
        output_dim = list(self.feature_dims.values())[0]
        if fusion_cfg and fusion_cfg.method == 'cross_attention': raise NotImplementedError("CrossAttentionFusion is not yet implemented.")
        total_feature_dim = sum(self.feature_dims.values())
        return nn.Linear(total_feature_dim, output_dim)

    def _create_classifier(self, num_classes: int) -> nn.Module:
        input_dim = list(self.feature_dims.values())[0]
        if self.config.classifier_hidden_dim:
            return nn.Sequential(nn.Linear(input_dim, self.config.classifier_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(self.config.classifier_dropout), nn.Linear(self.config.classifier_hidden_dim, num_classes))
        return nn.Linear(input_dim, num_classes)

    def forward(self, batch: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(batch, dict):
            if len(self.config.streams) != 1: raise ValueError("Input must be a dict for multi-stream models.")
            stream_name = next(iter(self.config.streams.keys()))
            batch = {stream_name: batch}
        features = {}
        for name, x in batch.items():
            adapter = self.adapters[name] if name in self.adapters else nn.Identity()
            backbone = self.backbones[name]
            features[name] = backbone(adapter(x))
        if self.fusion:
            if isinstance(self.fusion, nn.Linear): fused_features = self.fusion(torch.cat(list(features.values()), dim=1))
            else: fused_features = self.fusion(features)
        else:
            fused_features = next(iter(features.values()))
        return self.classifier(fused_features)

class UnifiedModel(BaseModel):
    def __init__(self, model_config: ModelConfig, base_config: BaseModelConfig, num_classes: int, **kwargs: Any) -> None:
        core_model = _UnifiedModelCore(model_config, num_classes)
        def default_batch_processor(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]: return batch[0], batch[1]
        batch_processor = kwargs.pop('batch_processor', default_batch_processor)
        super().__init__(model=core_model, num_classes=num_classes, config=base_config, batch_processor=batch_processor, **kwargs)
        self.model_config = model_config

    def get_feature_maps(self, batch: Dict[str, torch.Tensor], target_layers: Dict[str, str]) -> Dict[str, torch.Tensor]:
        feature_maps: Dict[str, torch.Tensor] = {}
        hooks: List[Callable] = []

        def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor, name: str) -> None:
            feature_maps[name] = output.detach()

        try:
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
            for hook in hooks:
                hook.remove()

        return feature_maps