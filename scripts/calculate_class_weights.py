# scripts/calculate_class_weights.py
"""
Calculate and save class weights for an imbalanced dataset from a metadata file.

This script reads a metadata file, calculates class weights using the inverse
frequency method, and saves them to a JSON file for use during training.
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def calculate_and_save_weights(
    metadata_path: Path,
    output_path: Path,
    label_col: str = 'label'
) -> None:
    """
    Calculates class weights from a metadata file and saves them to JSON.
    """
    try:
        logger.info(f"Loading metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)
        
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in metadata file.")
            
        class_counts = df[label_col].value_counts().sort_index()
        
        logger.info(f"Found {len(df)} total samples across {len(class_counts)} classes.")
        logger.info(f"Class distribution summary:\n{class_counts.describe()}")
        
        # Calculate weights using the inverse frequency method
        total_samples = len(df)
        num_classes = len(class_counts)
        
        weights = (total_samples / (num_classes * class_counts)).to_dict()
        
        # Normalize weights to have a mean of 1 for stability
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            normalized_weights = {k: (v * num_classes / weight_sum) for k, v in weights.items()}
        else:
            normalized_weights = weights

        logger.info(f"Calculated weight range: {min(normalized_weights.values()):.2f} - {max(normalized_weights.values()):.2f}")

        # Save weights to the specified output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(normalized_weights, f, indent=4)
        
        logger.info(f"Class weights successfully saved to: {output_path}")

    except FileNotFoundError:
        logger.error(f"Metadata file not found at: {metadata_path}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during class weight calculation: {e}", exc_info=True)
        raise


def main():
    """Main function to parse arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Calculate and save class weights for imbalanced datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metadata-path", type=Path, required=True,
        help="Path to the input metadata CSV file containing labels."
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Path to save the output class weights JSON file."
    )
    parser.add_argument(
        "--label-col", type=str, default='label',
        help="Name of the column containing class labels in the metadata file."
    )
    
    args = parser.parse_args()
    
    try:
        calculate_and_save_weights(args.metadata_path, args.output_path, args.label_col)
        logger.info("Process completed successfully!")
    except Exception:
        # The specific error will have already been logged by the function
        logger.critical("Process failed. Please see the error message above.")
        # Exiting with a non-zero status code signals failure to shell scripts/CI
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
