# src/krishi_sahayak/utils/explainability.py
"""
Advanced model explainability module with multiple CAM methods. (Refactored)
This module is now solely responsible for the computation of explanation maps.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class CAMMethod(str, Enum):
    """Available Class Activation Mapping methods."""
    GRADCAM = 'gradcam'
    GRADCAM_PLUSPLUS = 'gradcam++'

class ActivationsAndGradients:
    """Class for extracting activations and gradients from target layers via hooks."""
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], reshape_transform: Optional[Callable]):
        self.model = model
        self.reshape_transform = reshape_transform
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.handles = [
            layer.register_forward_hook(self.save_activation) for layer in target_layers
        ]
        self.handles.extend([
            layer.register_full_backward_hook(self.save_gradient) for layer in target_layers
        ])
    
    def save_activation(self, module: nn.Module, inputs: Any, output: torch.Tensor):
        activation = output
        if self.reshape_transform:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
    
    def save_gradient(self, module: nn.Module, grad_input: Any, grad_output: Any):
        grad = grad_output[0]
        if self.reshape_transform:
            grad = self.reshape_transform(grad)
        self.gradients.append(grad.cpu().detach())
    
    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.gradients = []
        self.activations = []
        return [self.model(x)]
    
    def release(self):
        for handle in self.handles:
            handle.remove()

class BaseCAM:
    """Abstract Base Class for CAM methods."""
    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        reshape_transform: Optional[Callable] = None,
    ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.model.to(self.device)
        
    def get_cam_weights(self, grads: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_cam_image(self, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        weights = self.get_cam_weights(grads, activations)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        cam = F.relu(cam)
        
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.detach().cpu().numpy().squeeze()
    
    def forward(self, input_tensor: torch.Tensor, targets: List[Callable], aug_smooth: bool = False) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)
        outputs = self.activations_and_grads(input_tensor)
        
        if aug_smooth:
            return self._aug_smooth_forward(input_tensor, targets)

        self.model.zero_grad()
        score = torch.cat([target(output) for target, output in zip(targets, outputs)]).sum()
        score.backward(retain_graph=True)

        cam_per_layer = []
        for activations, grads in zip(self.activations_and_grads.activations, self.activations_and_grads.gradients[::-1]):
            cam_per_layer.append(self.get_cam_image(activations, grads))
        
        return np.mean(cam_per_layer, axis=0)
    
    def _aug_smooth_forward(self, input_tensor: torch.Tensor, targets: List[Callable]) -> np.ndarray:
        aug_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(15)])
        cams = [self.forward(aug_transforms(input_tensor), targets, aug_smooth=False) for _ in range(5)]
        return np.mean(cams, axis=0)

    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def __del__(self): self.activations_and_grads.release()


class GradCAM(BaseCAM):
    def get_cam_weights(self, grads: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        return torch.mean(grads, dim=(2, 3), keepdim=True)


class GradCAMPlusPlus(BaseCAM):
    def get_cam_weights(self, grads: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        alpha = torch.where(grads != 0, alpha, 0)
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)
        return weights


def create_cam_explainer(method: Union[str, CAMMethod], **kwargs: Any) -> BaseCAM:
    """Factory function to create a CAM explainer instance."""
    method_map = { CAMMethod.GRADCAM: GradCAM, CAMMethod.GRADCAM_PLUSPLUS: GradCAMPlusPlus }
    cam_class = method_map.get(CAMMethod(method.lower()))
    if cam_class is None:
        raise ValueError(f"Unsupported CAM method: {method}")
    return cam_class(**kwargs)