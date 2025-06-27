# examples/gradcam_demo.py
"""
KrishiSahayak - Grad-CAM Demonstration Script (Best Practice)

This script demonstrates how to use the high-level `InferenceHandler` to get
both predictions and explainability heatmaps for a given image and model.
"""

import argparse
import logging
from pathlib import Path

import torch

# REFACTORED: Use absolute imports assuming the package is installed (`pip install -e .`)
# The demo now uses the canonical, high-level handler.
from krishi_sahayak.inference.handler import InferenceHandler
from krishi_sahayak.utils.hardware import auto_detect_accelerator
from krishi_sahayak.utils.logger import setup_logging
from krishi_sahayak.utils.visualization import visualize_prediction

logger = logging.getLogger(__name__)

def run_explanation_demo(args: argparse.Namespace) -> None:
    """
    Orchestrates the process of loading a model via the handler and generating
    an explained prediction.
    """
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {args.checkpoint}")
    if not Path(args.image_path).is_file():
        raise FileNotFoundError(f"Input image not found at: {args.image_path}")

    # --- 1. Initialize the Inference Handler ---
    # The handler manages model loading, device placement, and predictor setup.
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    device = torch.device(args.device)
    handler = InferenceHandler(checkpoint_path=Path(args.checkpoint), device=device)
    logger.info("Inference handler initialized successfully.")

    # --- 2. Run Inference to Get Predictions and Explanation ---
    # The handler's `run_single` method is the high-level entry point.
    # We assume the handler and predictor were refactored to return explanations.
    logger.info(f"Generating prediction and explanation for: {args.image_path}")
    result = handler.run_single(
        image_path=Path(args.image_path),
        top_k=args.top_k
    )

    if not result:
        logger.error("Inference failed to produce a result.")
        return

    # --- 3. Visualize the Output ---
    output_filename = f"explanation_{Path(args.image_path).stem}.png"
    output_path = Path(args.output_dir) / output_filename
    
    logger.info(f"Saving visualization to: {output_path}")
    visualize_prediction(
        result=result,
        explanation=result.get('explanation'), # Pass the explanation dict if it exists
        output_path=output_path,
        show=not args.no_show # Control display via CLI flag
    )

    # --- 4. Print Results to Console ---
    print("\n" + "="*50)
    print(f"✅ Analysis Complete for: {Path(args.image_path).name}")
    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(result.get('predictions', [])):
        print(f"  {i+1}. {pred.get('class', 'N/A')}: {pred.get('probability', 0):.4f}")
    
    if result.get('explanation'):
        expl = result['explanation']
        print(f"\nExplanation generated via '{expl.get('method', 'N/A')}' for class '{expl.get('class_name', 'N/A')}'.")
    
    print(f"\n🔗 Visualization saved to: {output_path}")
    print("="*50)


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Generate and visualize a Grad-CAM explanation for a model prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_path", type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.ckpt) file."
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/demos",
        help="Directory to save the output visualization."
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top predictions to show."
    )
    parser.add_argument(
        "--device", type=str, default=auto_detect_accelerator(),
        help="Device to run inference on (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not automatically open the visualization plot."
    )
    
    args = parser.parse_args()
    
    setup_logging(project_name="krishisahayak.demo")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        run_explanation_demo(args)
    except Exception as e:
        logger.critical(f"The demo script failed with a critical error: {e}", exc_info=True)
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()