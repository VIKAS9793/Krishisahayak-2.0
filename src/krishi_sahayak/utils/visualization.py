# src/krishi_sahayak/utils/visualization.py
"""
Utility functions for creating and saving visualizations.
This module includes advanced capabilities for rendering model explanations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def visualize_prediction(
    result: Dict[str, Any],
    explanation_map: Optional[np.ndarray] = None,
    output_path: Union[Path, str, None] = None
) -> None:
    """
    Generates and saves/shows a visualization of model predictions for an image,
    with an optional explanation map overlay.

    Args:
        result: A dictionary containing the predictions and image data.
                Expected keys:
                - 'predictions': A list of prediction dicts.
                - 'image_path': Path to the source image (used for title).
                - 'image_array' (Optional): Pre-loaded PIL Image or NumPy array.
        explanation_map (Optional[np.ndarray]): A 2D NumPy array representing the
                                                 heatmap (e.g., from Grad-CAM)
                                                 to be overlaid on the image.
        output_path: Path to save the visualization. If None, shows the plot.
    """
    fig = None  # Initialize for the finally block
    try:
        # Prioritize using a pre-loaded image array
        if 'image_array' in result and result['image_array'] is not None:
            image = result['image_array']
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        elif 'image_path' in result:
            image = Image.open(result['image_path']).convert('RGB')
        else:
            raise ValueError("Result dictionary must contain 'image_path' or 'image_array'.")

        predictions = result['predictions']
        top_pred = predictions[0]

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]}
        )
        
        # --- Image Panel (Axes 0) ---
        axes[0].imshow(image)
        # Overlay the explanation heatmap if provided
        if explanation_map is not None and isinstance(explanation_map, np.ndarray):
            axes[0].imshow(
                explanation_map,
                cmap='jet',
                alpha=0.5, # Make heatmap semi-transparent
                extent=(0, image.width, image.height, 0)
            )
        
        image_name = Path(result.get('image_path', 'In-memory Image')).name
        title_text = (
            f"Input: {image_name}\n"
            f"Top Prediction: {top_pred.get('class', 'N/A')} ({top_pred.get('probability', 0):.2%})"
        )
        axes[0].set_title(title_text)
        axes[0].axis('off')
        
        # --- Predictions Panel (Axes 1) ---
        class_names = [p.get('class', f'Class {i}') for i, p in enumerate(predictions)]
        probs = [p.get('probability', 0) for p in predictions]
        y_pos = np.arange(len(class_names))

        axes[1].barh(y_pos, probs, align='center', color='skyblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(class_names)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Top Predictions')
        axes[1].set_xlim(0, 1)

        fig.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        else:
            plt.show() # pragma: no cover (Difficult to test in automated environments)

    except Exception as e:
        logger.error(f"Failed to generate visualization for {result.get('image_path', 'N/A')}: {e}")
    finally:
        # Ensure the figure is closed to free up memory
        if fig is not None:
            plt.close(fig)