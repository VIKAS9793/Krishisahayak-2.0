# src/krishi_sahayak/utils/visualization.py
"""
Advanced visualization utilities for model predictions and explanations. (Refactored)
This module provides tools for visualizing model predictions and class activation maps.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def overlay_cam_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.6,
) -> np.ndarray:
    """
    Overlays a CAM heatmap on top of an input image. (Moved from explainability.py)
    
    Args:
        image: Input image as a numpy array (H, W, 3) in RGB format.
        mask: CAM heatmap as a 2D numpy array (H, W) with values in [0, 1].
        colormap: OpenCV colormap to use for the heatmap.
        image_weight: Weight for the original image in the final blend (0-1).
        
    Returns:
        Blended image with the heatmap overlay as a numpy array.
    """
    # Resize mask to match image dimensions
    h, w, _ = image.shape
    resized_mask = cv2.resize(mask, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * resized_mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    blended_image = cv2.addWeighted(np.uint8(image), image_weight, heatmap, 1 - image_weight, 0)
    return blended_image


def visualize_prediction(
    result: Dict[str, Any],
    explanation: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    dpi: int = 120,
    figsize: Tuple[int, int] = (10, 5),
) -> Optional[np.ndarray]:
    """Visualize model predictions with an optional explanation heatmap."""
    try:
        if 'image_array' in result and result['image_array'] is not None:
            image_source = result['image_array']
            image = Image.fromarray(image_source) if isinstance(image_source, np.ndarray) else image_source
        elif 'image_path' in result:
            image = Image.open(result['image_path'])
        else:
            raise ValueError("Either 'image_array' or 'image_path' must be provided in result")
        
        image = image.convert("RGB")
        image_np = np.array(image)

    except Exception as e:
        logger.error(f"Failed to load image for visualization: {e}")
        return None

    has_explanation = explanation is not None and 'heatmap' in explanation
    num_cols = 2 if has_explanation else 1
    fig, axes = plt.subplots(1, num_cols, figsize=figsize, dpi=dpi, squeeze=False)
    ax_img, ax_expl = axes.flatten()[0], (axes.flatten()[1] if has_explanation else None)

    # Plot Original Image and Predictions
    ax_img.imshow(image_np)
    ax_img.set_title('Input & Top Predictions')
    ax_img.axis('off')

    if 'predictions' in result and result['predictions']:
        pred_text = [f"{i+1}. {p.get('class', 'N/A')}: {p.get('probability', 0):.2%}" for i, p in enumerate(result['predictions'][:3])]
        ax_img.text(0.02, 0.98, "\n".join(pred_text), transform=ax_img.transAxes,
                    bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 4},
                    fontsize=9, color='white', verticalalignment='top')

    # Plot Explanation
    if has_explanation and ax_expl:
        overlay = overlay_cam_on_image(image_np, explanation['heatmap'])
        ax_expl.imshow(overlay)
        expl_title = f"Explanation: {explanation.get('method', 'CAM')}\nFor Class: {explanation.get('class_name', 'N/A')}"
        ax_expl.set_title(expl_title)
        ax_expl.axis('off')

    fig.tight_layout()
    
    if output_path is None and not show:
        fig.canvas.draw()
        final_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        final_image = final_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return final_image
        
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return None