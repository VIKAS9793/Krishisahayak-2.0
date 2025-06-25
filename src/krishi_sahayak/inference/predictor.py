"""Contains the lean Predictor class for running inference on tensors."""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List

class Predictor:
    """
    A lean wrapper around a PyTorch model for performing inference.
    It expects pre-processed tensors and returns structured predictions.
    """
    def __init__(self, model: torch.nn.Module, class_names: List[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device

    @torch.no_grad()
    def predict_batch(self, inputs: Dict[str, torch.Tensor], top_k: int) -> List[List[Dict[str, Any]]]:
        """
        Performs inference on a batch of tensors.

        Args:
            inputs: A dictionary of input tensors, already on the correct device.
            top_k: The number of top predictions to return.

        Returns:
            A list of prediction lists, where each inner list contains top_k
            prediction dictionaries for an image in the batch.
        """
        outputs = self.model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

        batch_results = []
        for i in range(top_probs.shape[0]):
            instance_results = [
                {
                    'class': self.class_names[top_indices[i, k]],
                    'probability': top_probs[i, k].item()
                }
                for k in range(top_k)
            ]
            batch_results.append(instance_results)
        return batch_results
