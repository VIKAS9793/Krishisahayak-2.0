"""
KrishiSahayak - Dataset Merging Utility

This script merges multiple pre-processed metadata CSV files into a single,
unified manifest. It includes logic to standardize class labels across
different datasets to ensure consistency for training.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def standardize_labels(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Standardizes labels in a DataFrame based on a provided mapping."""
    # Create a reverse mapping for faster lookups
    reverse_map = {v: k for k, values in mapping.items() for v in values}
    
    # Map old labels to new standard labels, keeping original if no mapping exists
    df['label'] = df['label'].map(reverse_map).fillna(df['label'])
    logger.info(f"Standardized labels. Found {df['label'].nunique()} unique labels after mapping.")
    return df

def merge_metadata_files(
    input_paths: list[Path],
    output_path: Path,
    label_mapping_path: Path
) -> None:
    """
    Loads multiple metadata CSVs, standardizes their labels, merges them,
    and saves the result.
    """
    # Load the label standardization mapping from the YAML file
    try:
        with open(label_mapping_path, 'r') as f:
            label_mapping = yaml.safe_load(f).get('label_standardization_map', {})
        logger.info(f"Loaded label mapping from: {label_mapping_path}")
    except FileNotFoundError:
        logger.error(f"Label mapping file not found at: {label_mapping_path}. Cannot standardize labels.")
        return

    all_dfs = []
    for path in input_paths:
        if not path.exists():
            logger.warning(f"Metadata file not found: {path}. Skipping.")
            continue
        
        logger.info(f"Loading metadata from: {path}")
        df = pd.read_csv(path)
        
        # Standardize labels before merging
        standardized_df = standardize_labels(df, label_mapping)
        all_dfs.append(standardized_df)

    if not all_dfs:
        logger.error("No valid metadata files were found to merge.")
        return

    # Merge all dataframes and shuffle the result for better training distribution
    merged_df = pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully merged {len(all_dfs)} datasets into a single file with {len(merged_df)} total samples.")
    logger.info(f"Final merged metadata saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and standardize multiple metadata CSV files.")
    parser.add_argument(
        "--inputs", type=Path, nargs='+', required=True,
        help="One or more paths to the input metadata CSV files."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to save the final merged metadata CSV file."
    )
    parser.add_argument(
        "--label-map", type=Path, required=True,
        help="Path to the YAML file containing the label standardization map."
    )
    
    args = parser.parse_args()
    merge_metadata_files(args.inputs, args.output, args.label_map)
