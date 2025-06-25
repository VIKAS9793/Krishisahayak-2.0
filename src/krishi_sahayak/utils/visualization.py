"""
Utility functions for creating and saving visualizations.
This module should only depend on libraries like Matplotlib, PIL, and NumPy.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def visualize_prediction(result: Dict[str, Any], output_path: Path | None = None) -> None:
    """
    Generates and saves/shows a visualization of model predictions for an image.

    Args:
        result: A dictionary containing the image_path and a list of predictions.
        output_path: Path to save the visualization. If None, shows the plot.
    """
    try:
        image = Image.open(result['image_path']).convert('RGB')
        predictions = result['predictions']
        top_pred = predictions[0]

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.2, 1]}
        )
        
        # Display the image and top prediction
        axes[0].imshow(image)
        title_text = (
            f"Input: {Path(result['image_path']).name}\n"
            f"Top Prediction: {top_pred['class']} ({top_pred['probability']:.2%})"
        )
        axes[0].set_title(title_text)
        axes[0].axis('off')
        
        # Display the bar chart of probabilities
        class_names = [p['class'] for p in predictions]
        probs = [p['probability'] for p in predictions]
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
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        else:
            plt.show() # pragma: no cover (Difficult to test in automated environments)

    except Exception as e:
        logger.error(f"Failed to generate visualization for {result.get('image_path', 'N/A')}: {e}")
    finally:
        # Ensure the figure is closed to free up memory
        if 'fig' in locals():
            plt.close(fig)
