"""
KrishiSahayak - Advanced Knowledge Distillation Module (Refactored)

This module provides a complete Knowledge Distillation framework, fully integrated
with the project's PyTorch Lightning architecture. This version uses modern type
hints, Pydantic configuration, and Dependency Injection for a robust, decoupled design.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

# Assuming a project-level base module exists for PL models
from krishi_sahayak.models.base import BaseModel as ProjectBaseModel

logger = logging.getLogger(__name__)

# --- Pydantic Configuration Schemas ---

class DistillationConfig(BaseModel):
    """Configuration for the knowledge distillation process."""
    temperature: float = Field(4.0, gt=0, description="Temperature for softening logits.")
    alpha: float = Field(0.7, ge=0, le=1, description="Weight for soft-target (KL) loss vs. hard-target (CE) loss.")
    feature_loss_weight: float = Field(0.0, ge=0, description="Weight for intermediate feature matching loss.")
    feature_match_layers: List[str] = Field(default_factory=list, description="Names of layers to match features from.")


# --- Custom Loss Module ---

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss that combines soft-target (logits) loss,
    hard-target (labels) loss, and optional intermediate feature-matching loss.
    """
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        feature_loss_weight: float = 0.5,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
            
        self.temperature = temperature
        self.alpha = alpha
        self.feature_loss_weight = feature_loss_weight
        
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.feature_loss_fn = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        student_features: Dict[str, torch.Tensor] | None = None,
        teacher_features: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        
        # Soft-target loss (distilling knowledge from teacher's probabilities)
        kld_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)  # Scale by T^2 as per original paper
        
        # Hard-target loss (learning from the ground truth labels)
        ce_loss = self.ce_loss(student_logits, target)
        
        # Optional feature-matching loss
        feat_loss = torch.tensor(0.0, device=student_logits.device)
        if self.feature_loss_weight > 0 and student_features and teacher_features:
            total_feature_loss = torch.tensor(0.0, device=student_logits.device)
            num_layers = 0
            for s_name, s_feat in student_features.items():
                if s_name in teacher_features:
                    t_feat = teacher_features[s_name]
                    # Normalize features before comparing to focus on orientation, not magnitude
                    s_feat_norm = F.normalize(s_feat.view(s_feat.size(0), -1), p=2, dim=1)
                    t_feat_norm = F.normalize(t_feat.view(t_feat.size(0), -1), p=2, dim=1)
                    total_feature_loss += self.feature_loss_fn(s_feat_norm, t_feat_norm)
                    num_layers += 1
            if num_layers > 0:
                feat_loss = total_feature_loss / num_layers

        # Combine losses
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kld_loss
        total_loss += self.feature_loss_weight * feat_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss.detach(),
            'kld_loss': kld_loss.detach(),
            'feature_loss': feat_loss.detach()
        }


# --- Lightning Module ---

class DistillationLightningModule(ProjectBaseModel):
    """
    A PyTorch Lightning Module for Knowledge Distillation.

    This module expects pre-instantiated and configured student and teacher models,
    following the Dependency Injection pattern for maximum flexibility and testability.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: DistillationConfig,
        learning_rate: float,
        **kwargs: Any,
    ) -> None:
        # Call parent __init__ first, passing the student model as the primary model
        super().__init__(model=student_model, learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(ignore=['student_model', 'teacher_model'])

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config

        # Ensure teacher model is frozen, as this component should not train it.
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("Teacher model has been frozen for distillation.")

        self.criterion = KnowledgeDistillationLoss(
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            feature_loss_weight=self.config.feature_loss_weight
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """The forward pass for inference uses only the student model."""
        return self.student_model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Performs a single knowledge distillation training step."""
        x, y = self._prepare_batch(batch) # Assumes method from parent ProjectBaseModel

        # Teacher provides soft labels and feature targets
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
            teacher_features = None
            if self.config.feature_loss_weight > 0 and hasattr(self.teacher_model, 'get_feature_maps'):
                teacher_features = self.teacher_model.get_feature_maps(x, self.config.feature_match_layers)
        
        # Student makes predictions and provides features
        student_logits = self.student_model(x)
        student_features = None
        if self.config.feature_loss_weight > 0 and hasattr(self.student_model, 'get_feature_maps'):
            student_features = self.student_model.get_feature_maps(x, self.config.feature_match_layers)
        
        loss_dict = self.criterion(
            student_logits=student_logits, teacher_logits=teacher_logits, target=y,
            student_features=student_features, teacher_features=teacher_features
        )
        
        self.log("train/loss", loss_dict['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items() if k != 'loss'}, on_step=False, on_epoch=True)
        
        return loss_dict['loss']

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Performs validation on the student model. The inherited `_shared_step`
        from the parent `ProjectBaseModel` calculates all necessary student-only metrics.
        """
        # This delegates validation to the parent class, which presumably
        # calculates metrics like accuracy, f1-score, etc., on the student model.
        self._shared_step(batch, 'val')
