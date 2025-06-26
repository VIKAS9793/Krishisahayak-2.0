"""
Unified Inference Script CLI for Plant Disease Classification Models.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch

# REFACTORED: Corrected import paths to be relative to the package root.
from krishisahayak.inference.handler import InferenceHandler
from krishisahayak.utils.hardware import auto_detect_accelerator
from krishisahayak.utils.logger import setup_logging
from krishisahayak.utils.visualization import visualize_prediction

logger = logging.getLogger(__name__)

# --- (The rest of the file's logic remains the same) ---

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    # ... function body is unchanged ...
    pass

def _run_single_inference(handler: InferenceHandler, args: argparse.Namespace) -> None:
    # ... function body is unchanged ...
    pass

def _run_batch_inference(handler: InferenceHandler, args: argparse.Namespace) -> None:
    # ... function body is unchanged ...
    pass

def main(argv: Optional[List[str]] = None) -> None:
    """Main orchestration function for the CLI."""
    # This setup now correctly uses the project's logging utility
    setup_logging(project_name="krishisahayak.inference_script")
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