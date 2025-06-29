# src/krishi_sahayak/inference/predictor.py
"""Predictor class for running inference and generating explanations."""
import logging
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F

# REFACTORED: Corrected import from `create_cam` to `create_cam_explainer`
from krishi_sahayak.utils.explainability import create_cam_explainer, CAMMethod

logger = logging.getLogger(__name__)

class Predictor:
    """A wrapper around a PyTorch model for inference and explainability."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: torch.device,
        target_layers: Optional[List[torch.nn.Module]] = None,
    ):
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.target_layers = target_layers or self._find_target_layers()
        self.cam = None

    def _find_target_layers(self) -> List[torch.nn.Module]:
        model_to_search = getattr(self.model, 'model', self.model)
        model_to_search = getattr(model_to_search, 'backbone', model_to_search)
        for pattern in [['layer4'], ['features'], ['blocks', -1], ['layers', -1]]:
            try:
                layer = model_to_search
                for p in pattern:
                    layer = layer[p] if isinstance(p, int) else getattr(layer, p)
                for m in reversed(list(layer.modules())):
                     if isinstance(m, torch.nn.Conv2d):
                        logger.info(f"Auto-detected target CAM layer: {m.__class__.__name__}")
                        return [m]
            except (AttributeError, IndexError, TypeError):
                continue
        raise ValueError("Could not automatically determine target layers.")

    def _initialize_cam(self, method: Union[str, CAMMethod]) -> None:
        if self.cam is None:
            self.cam = create_cam_explainer(
                method=method,
                model=self.model,
                target_layers=self.target_layers
            )

    def _generate_explanation(self, input_tensor: torch.Tensor, target_class_index: int, xai_config: Dict[str, Any]) -> np.ndarray:
        method = xai_config.get('method', 'gradcam')
        self._initialize_cam(method)
        class ClassifierOutputTarget(object):
            def __init__(self, category):
                self.category = category
            def __call__(self, model_output):
                if model_output.ndim > 1: return model_output[:, self.category]
                return model_output[self.category]
        targets = [ClassifierOutputTarget(target_class_index)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets, aug_smooth=xai_config.get('aug_smooth', False))
        return grayscale_cam[0]

    def predict_batch(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], top_k: int = 5, get_explanations: bool = False, xai_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        input_tensor = inputs.get('rgb', next(iter(inputs.values()))) if isinstance(inputs, dict) else inputs
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
            probabilities = F.softmax(logits, dim=1)
            top_probs, top_indices = probabilities.topk(top_k, dim=1)
        batch_results = []
        for i in range(probabilities.shape[0]):
            predictions = [{'class': self.class_names[idx], 'probability': prob.item(), 'class_id': idx.item()} for prob, idx in zip(top_probs[i], top_indices[i])]
            result = {'predictions': predictions}
            if get_explanations:
                try:
                    with torch.enable_grad():
                        explanation_map = self._generate_explanation(input_tensor=input_tensor[i:i+1], target_class_index=top_indices[i, 0].item(), xai_config=xai_config or {})
                    result['explanation'] = {'heatmap': explanation_map, 'method': (xai_config or {}).get('method', 'gradcam'), 'target_class': top_indices[i, 0].item(), 'class_name': self.class_names[top_indices[i, 0].item()]}
                except Exception as e:
                    logger.error(f"Failed to generate explanation for item {i}: {e}", exc_info=True)
            batch_results.append(result)
        return batch_results