"""
A command-line utility to analyze and visualize a dataset's distribution.

This script takes a metadata CSV file as input and generates visualizations for
class distribution and sample images, along with an imbalance analysis.
"""
import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

# Set up a logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Plots and saves a bar chart of the class distribution."""
    logger.info("Generating class distribution plot...")
    plt.figure(figsize=(15, 10))
    class_counts = df['label'].value_counts().sort_values(ascending=False)
    
    ax = sns.barplot(y=class_counts.index, x=class_counts.values, orient='h')
    
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.tight_layout()
    
    output_path = output_dir / 'class_distribution.png'
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Class distribution plot saved to: {output_path}")
    return class_counts

def plot_sample_images(df: pd.DataFrame, image_base_dir: Path, output_dir: Path, num_samples: int = 5) -> None:
    """Saves a grid of sample images for each class."""
    logger.info("Generating sample images for each class...")
    samples_dir = output_dir / 'sample_images_by_class'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in df['label'].unique():
        class_df = df[df['label'] == class_name]
        if class_df.empty:
            continue
            
        samples = class_df.sample(min(num_samples, len(class_df)))
        
        fig, axes = plt.subplots(1, len(samples), figsize=(15, 4), squeeze=False) # squeeze=False ensures axes is always 2D
        fig.suptitle(f'Class: {class_name} ({len(class_df)} total samples)', y=1.02)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            try:
                # Construct the full path robustly
                full_path = image_base_dir / row['image_path']
                if not full_path.is_file():
                    logger.warning(f"Image not found, skipping: {full_path}")
                    continue
                
                img = Image.open(full_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(Path(row['image_path']).name[:20] + '...', fontsize=8) # Shorten long filenames
                axes[0, i].axis('off')
            except Exception as e:
                logger.error(f"Could not load or plot image {full_path}: {e}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        safe_class_name = "".join(c for c in class_name if c.isalnum()).rstrip()
        plt.savefig(samples_dir / f"{safe_class_name}_samples.png")
        plt.close()
    
    logger.info(f"Sample images saved to: {samples_dir}")

def analyze_imbalance(class_counts: pd.Series, threshold: float = 0.1) -> None:
    """Analyzes and reports on class imbalance."""
    total_samples = class_counts.sum()
    avg_samples = total_samples / len(class_counts)
    imbalance_ratio = class_counts.max() / class_counts.min()

    logger.info("--- Class Imbalance Analysis ---")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Number of Classes: {len(class_counts)}")
    logger.info(f"Average Samples per Class: {avg_samples:.1f}")
    logger.info(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.1f}x")
    
    minority_classes = class_counts[class_counts < (avg_samples * threshold)]
    if not minority_classes.empty:
        logger.warning(f"\nClasses with very few samples (less than {threshold*100:.0f}% of average):")
        for cls, count in minority_classes.items():
            logger.warning(f"- {cls}: {count} samples")

def analyze_dataset(args: argparse.Namespace) -> None:
    """Main orchestration function for the analysis."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(args.metadata_path)
        logger.info(f"Loaded metadata with {len(df)} samples and {df['label'].nunique()} classes.")
    except FileNotFoundError:
        logger.error(f"Metadata file not found at: {args.metadata_path}")
        return

    # 1. Plot class distribution
    class_counts = plot_class_distribution(df, args.output_dir)
    
    # 2. Analyze imbalance
    analyze_imbalance(class_counts)
    
    # 3. Plot sample images
    plot_sample_images(df, args.image_base_dir, args.output_dir, args.num_samples)
    
    logger.info("\nAnalysis complete! Check the output directory for visualizations.")
    logger.info(f"Output directory: {args.output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize a dataset.")
    parser.add_argument(
        "--metadata-path", type=Path, required=True,
        help="Path to the metadata CSV file."
    )
    parser.add_argument(
        "--image-base-dir", type=Path, required=True,
        help="The base directory where the image paths from the CSV are relative to (e.g., 'data/raw')."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save the analysis reports and figures."
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of random sample images to show for each class."
    )
    
    args = parser.parse_args()
    analyze_dataset(args)
