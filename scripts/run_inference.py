"""
Unified Inference Script CLI for Plant Disease Classification Models.

This script provides a command-line interface to run predictions using a trained
model checkpoint on either a single image or a directory of images.
"""

import argparse
import json
import logging
import os  # CORRECTED: Added missing import for os.cpu_count()
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Assumes project is installed, making these modules available.
from src.krishisahayak.inference.handler import InferenceHandler
from src.krishisahayak.utils.hardware import auto_detect_accelerator
from src.krishisahayak.utils.logger import setup_logging
from src.krishisahayak.utils.visualization import visualize_prediction

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Defines and parses command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Production-Grade Inference Script for KrishiSahayak",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the trained model checkpoint (.ckpt) file."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image file for prediction.")
    group.add_argument("--input-dir", type=Path, help="Path to a directory of images for batch prediction.")
    
    parser.add_argument("--output-dir", type=Path, help="Directory to save visualizations and batch summary.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', 'mps'). Auto-detects if not set.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to return.")

    # Arguments specific to hybrid/multi-modal models
    parser.add_argument("--nir-image", type=Path, help="Path to a single corresponding NIR image (for use with --image).")
    parser.add_argument("--nir-suffix", type=str, default="_nir.tif", help="Suffix to find NIR images in batch mode.")

    # Arguments for batch processing
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for directory processing.")
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 4), help="Number of workers for data loading.")
    
    return parser.parse_args(argv)


def _run_single_inference(handler: InferenceHandler, args: argparse.Namespace) -> None:
    """Handles the logic for single image inference."""
    logger.info(f"Running single image inference on: {args.image}")
    result = handler.run_single(
        image_path=args.image,
        nir_image_path=args.nir_image,
        top_k=args.top_k
    )
    # Print JSON result to standard output for easy piping
    sys.stdout.write(json.dumps(result, indent=2) + '\n')

    if args.output_dir:
        output_path = args.output_dir / f"{args.image.stem}_prediction.png"
        visualize_prediction(result, output_path=output_path)


def _run_batch_inference(handler: InferenceHandler, args: argparse.Namespace) -> None:
    """Handles the logic for batch inference on a directory."""
    if not args.output_dir:
        raise ValueError("--output-dir is required for batch processing.")
    
    logger.info(f"Running batch inference on directory: {args.input_dir}")
    results_iterator = handler.run_batch(
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nir_suffix=args.nir_suffix,
        top_k=args.top_k
    )
    
    all_results = list(results_iterator)
    if not all_results:
        logger.warning("Batch processing finished with no results.")
        return

    for result in all_results:
        image_path = Path(result['image_path'])
        output_path = args.output_dir / f"{image_path.stem}_prediction.png"
        visualize_prediction(result, output_path=output_path)
    
    summary_path = args.output_dir / "predictions_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Batch processing complete. Summary saved to: {summary_path}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main orchestration function for the CLI."""
    setup_logging() # Use the project's standard logger setup
    args = _parse_args(argv)

    try:
        device = torch.device(args.device or auto_detect_accelerator())
        handler = InferenceHandler(checkpoint_path=args.checkpoint, device=device)

        if args.image:
            _run_single_inference(handler, args)
        elif args.input_dir:
            _run_batch_inference(handler, args)

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"A configuration or file error occurred: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()