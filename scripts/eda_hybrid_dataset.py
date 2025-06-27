# scripts/eda_hybrid_dataset.py
"""
Performs Exploratory Data Analysis (EDA) on the merged hybrid dataset.

This script loads the unified metadata file, analyzes class and split
distributions, and computes statistics on a sample of the image files to
ensure data integrity before training.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def run_eda(metadata_path: Path, image_base_dir: Path, output_dir: Path, sample_size: int):
    """Main function to orchestrate the EDA process."""
    logger.info("--- Starting Exploratory Data Analysis for Hybrid Dataset ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Analyze Metadata
    try:
        logger.info(f"Loading merged metadata from: {metadata_path}")
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        logger.critical(f"Metadata file not found at {metadata_path}. Aborting.")
        return

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of unique classes: {df['label'].nunique()}")

    # 2. Analyze and Plot Class Distribution
    analyze_class_distribution(df, output_dir)

    # 3. Analyze and Plot Dataset Splits
    analyze_dataset_splits(df, output_dir)

    # 4. Analyze Image Statistics from a Sample
    analyze_image_stats(df, image_base_dir, output_dir, sample_size)
    
    logger.info(f"--- EDA Completed. Reports saved to: {output_dir} ---")

def analyze_class_distribution(df: pd.DataFrame, output_dir: Path):
    """Analyzes and plots the distribution of classes."""
    logger.info("Analyzing class distribution...")
    class_dist = df['label'].value_counts()
    
    plt.figure(figsize=(12, 10))
    sns.countplot(y='label', data=df, order=class_dist.index, palette="viridis")
    plt.title('Class Distribution in Merged Hybrid Dataset')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.xscale('log') # Use log scale for better visualization of imbalanced classes
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    logger.info(f"Saved class distribution plot to {output_dir / 'class_distribution.png'}")

def analyze_dataset_splits(df: pd.DataFrame, output_dir: Path):
    """Analyzes and plots the distribution of train/val/test splits."""
    if 'split' not in df.columns:
        logger.warning("No 'split' column found in metadata. Skipping split analysis.")
        return

    logger.info("Analyzing dataset splits...")
    split_counts = df['split'].value_counts()
    logger.info(f"Split distribution:\n{split_counts}")

    plt.figure(figsize=(8, 6))
    split_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66c2a5','#fc8d62','#8da0cb'])
    plt.title('Train/Validation/Test Split Distribution')
    plt.ylabel('') # Hide the y-label
    plt.savefig(output_dir / 'split_distribution.png')
    plt.close()
    logger.info(f"Saved split distribution plot to {output_dir / 'split_distribution.png'}")


def analyze_image_stats(df: pd.DataFrame, image_base_dir: Path, output_dir: Path, sample_size: int):
    """Analyzes image dimensions, aspect ratios, and channels on a sample of the data."""
    logger.info(f"Analyzing image statistics on a random sample of {sample_size} images...")
    
    # Take a random sample to avoid processing the entire dataset
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    dims, aspect_ratios = [], []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing images"):
        # CORRECTED: Properly join the base directory with the relative path.
        image_path = image_base_dir / row['image_path']
        try:
            with Image.open(image_path) as img:
                dims.append(img.size) # (width, height)
                aspect_ratios.append(img.width / img.height)
        except (FileNotFoundError, UnidentifiedImageError, IOError) as e:
            logger.warning(f"Could not read or process image at {image_path}: {e}")
            continue

    if not dims:
        logger.error("Could not read any images from the sample. Check file paths and integrity.")
        return

    dims_df = pd.DataFrame(dims, columns=['width', 'height'])

    # --- Log Statistics ---
    logger.info("--- Image Statistics Summary ---")
    logger.info(f"Height (pixels): Mean={dims_df['height'].mean():.2f}, Std={dims_df['height'].std():.2f}, Min={dims_df['height'].min()}, Max={dims_df['height'].max()}")
    logger.info(f"Width (pixels):  Mean={dims_df['width'].mean():.2f}, Std={dims_df['width'].std():.2f}, Min={dims_df['width'].min()}, Max={dims_df['width'].max()}")
    logger.info(f"Aspect Ratio:    Mean={np.mean(aspect_ratios):.2f}, Std={np.std(aspect_ratios):.2f}")

    # --- Save Plots ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=dims_df, x='width', y='height', bins=30, cmap="viridis")
    plt.title('Distribution of Image Dimensions')

    plt.subplot(1, 2, 2)
    sns.histplot(aspect_ratios, bins=30, kde=True)
    plt.title('Distribution of Aspect Ratios')
    plt.xlabel('Aspect Ratio (Width / Height)')

    plt.tight_layout()
    plt.savefig(output_dir / 'image_stats_distribution.png')
    plt.close()
    logger.info(f"Saved image statistics plot to {output_dir / 'image_stats_distribution.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis on the hybrid dataset.")
    parser.add_argument(
        "--metadata-path", type=Path, required=True,
        help="Path to the merged metadata CSV file."
    )
    parser.add_argument(
        "--image-base-dir", type=Path, required=True,
        help="The absolute base directory where image paths in the CSV are relative to (e.g., 'data/raw/')."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/reports/eda_hybrid_report"),
        help="Directory to save analysis reports and plots."
    )
    parser.add_argument(
        "--sample-size", type=int, default=500,
        help="Number of random images to sample for dimension/statistic analysis."
    )
    
    args = parser.parse_args()
    run_eda(args.metadata_path, args.image_base_dir, args.output_dir, args.sample_size)