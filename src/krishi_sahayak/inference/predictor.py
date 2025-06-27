# src/krishi_sahayak/inference/predictor.py
"""
Predictor class for running inference and generating explanations using various CAM methods.
"""
import logging
from typing import Any, Dict, List, Optional, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from krishi_sahayak.utils.explainability import create_cam, CAMMethod
from krishi_sahayak.utils.visualization import visualize_prediction

logger = logging.getLogger(__name__)

class Predictor:
    """A wrapper around a PyTorch model for inference and explainability.
    
    Supports multiple CAM methods for model interpretability.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        class_names: List[str], 
        device: torch.device,
        target_layers: Optional[List[torch.nn.Module]] = None,
    ):
        """Initialize the predictor.
        
        Args:
            model: The PyTorch model for inference.
            class_names: List of class names.
            device: Device to run inference on.
            target_layers: List of target layers for CAM. If None, will try to auto-detect.
        """
        self.model = model.to(device).eval()
        self.class_names = class_names
        self.device = device
        
        # Set target layers for CAM
        self.target_layers = target_layers or self._find_target_layers()
        self.cam = None
        
    def _find_target_layers(self) -> List[torch.nn.Module]:
        """Attempt to automatically find suitable target layers for CAM."""
        # This heuristic attempts to find the last convolutional block in common architectures.
        model_to_search = getattr(self.model, 'model', self.model) # Handle Lightning wrapper
        model_to_search = getattr(model_to_search, 'backbone', model_to_search)

        # Common layer patterns
        for pattern in [['layer4'], ['features'], ['blocks', -1], ['layers', -1]]:
            try:
                layer = model_to_search
                for p in pattern:
                    layer = layer[p] if isinstance(p, int) else getattr(layer, p)
                
                # We want the last convolutional module in this block
                for m in reversed(list(layer.modules())):
                     if isinstance(m, torch.nn.Conv2d):
                        logger.info(f"Auto-detected target CAM layer: {m.__class__.__name__}")
                        return [m]
            except (AttributeError, IndexError, TypeError):
                continue
                
        raise ValueError("Could not automatically determine target layers. Please specify them manually.")

    def _initialize_cam(self, method: Union[str, CAMMethod]) -> None:
        """Initialize the CAM method if not already done."""
        if self.cam is None or not isinstance(self.cam, create_cam.__annotations__['return']):
            self.cam = create_cam(
                method=method,
                model=self.model,
                target_layers=self.target_layers,
                use_cuda=self.device.type == 'cuda',
            )
    
    def _generate_explanation(
        self, 
        input_tensor: torch.Tensor,
        target_class_index: int,
        xai_config: Dict[str, Any],
    ) -> np.ndarray:
        """Generate a CAM explanation for the given input and target class."""
        method = xai_config.get('method', 'gradcam')
        self._initialize_cam(method)
        
        class ClassifierOutputTarget(object):
            def __init__(self, category):
                self.category = category
            def __call__(self, model_output):
                if model_output.ndim > 1:
                    return model_output[:, self.category]
                return model_output[self.category]

        targets = [ClassifierOutputTarget(target_class_index)]
        
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=xai_config.get('eigen_smooth', False),
            aug_smooth=xai_config.get('aug_smooth', False),
        )
        
        return grayscale_cam[0]  # Return first (and only) item in batch

    def predict_batch(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        top_k: int = 5,
        get_explanations: bool = False,
        xai_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Performs inference on a batch of inputs."""
        input_tensor = inputs.get('rgb', next(iter(inputs.values()))) if isinstance(inputs, dict) else inputs
        
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Handle different common output formats from models
            logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                
            probabilities = F.softmax(logits, dim=1)
            top_probs, top_indices = probabilities.topk(top_k, dim=1)
        
        batch_results = []
        for i in range(probabilities.shape[0]):
            predictions = [
                {
                    'class': self.class_names[idx],
                    'probability': prob.item(),
                    'class_id': idx.item()
                } for prob, idx in zip(top_probs[i], top_indices[i])
            ]
            result = {'predictions': predictions}
            
            # Generate explanation if requested
            if get_explanations:
                try:
                    # Enable grad only for the explanation generation
                    with torch.enable_grad():
                        explanation_map = self._generate_explanation(
                            input_tensor=input_tensor[i:i+1], # Pass a single-item batch
                            target_class_index=top_indices[i, 0].item(),
                            xai_config=xai_config or {},
                        )
                    result['explanation'] = {
                        'heatmap': explanation_map,
                        'method': (xai_config or {}).get('method', 'gradcam'),
                        'target_class': top_indices[i, 0].item(),
                        'class_name': self.class_names[top_indices[i, 0].item()]
                    }
                except Exception as e:
                    logger.error(f"Failed to generate explanation for item {i}: {e}", exc_info=True)
            
            batch_results.append(result)
        
        return batch_results