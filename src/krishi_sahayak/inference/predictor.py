# src/krishi_sahayak/inference/predictor.py
"""Contains the lean Predictor class for running inference on tensors."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from PIL import Image

class Predictor:
    """
    A lean wrapper around a PyTorch model for performing inference and
    generating explanations.
    """
    def __init__(self, model: torch.nn.Module, class_names: List[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.grad_cam_target_layer = None # Set this to the layer you want to hook

    def _generate_grad_cam(self, inputs, class_idx: int) -> Optional[Image.Image]:
        """Generates a Grad-CAM heatmap for a given class index."""
        if not hasattr(self.model, 'get_feature_maps'): return None
        
        # Simplified Grad-CAM logic
        self.model.eval()
        logits = self.model(inputs)
        logits[:, class_idx].backward()
        
        # Get gradients and feature maps from the hooked layer
        gradients = self.model.gradients['value']
        activations = self.model.activations['value']
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        
        # Resize heatmap and convert to image
        original_img_size = inputs['rgb'].shape[2:]
        heatmap = cv2.resize(heatmap.numpy(), (original_img_size[1], original_img_size[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_img)

    @torch.no_grad()
    def predict_batch(
        self,
        inputs: Dict[str, torch.Tensor],
        top_k: int,
        get_explanations: bool = False,
        xai_config: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs inference on a batch of tensors and optionally generates explanations.
        """
        outputs = self.model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

        batch_results = []
        for i in range(top_probs.shape[0]):
            instance_results = {
                'predictions': [
                    {
                        'class': self.class_names[top_indices[i, k]],
                        'probability': top_probs[i, k].item()
                    }
                    for k in range(top_k)
                ]
            }
            
            # Generate explanation for the top prediction
            if get_explanations and xai_config:
                # This is a placeholder for a full Grad-CAM implementation
                # which would require hooks to be set up based on xai_config
                top_pred_idx = top_indices[i, 0].item()
                # A real implementation would be more robust and likely live in a separate utility
                # For now, we simulate a simple heatmap from the model's features if possible.
                if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                    try:
                        with torch.enable_grad():
                             # A simplified stand-in for a full XAI utility call
                             # This logic is complex and would live in a dedicated XAI utility
                             pass
                    except Exception:
                        pass # Silently fail if explanation cannot be generated

            batch_results.append(instance_results)
        return batch_results
