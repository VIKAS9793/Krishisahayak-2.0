"""
Analyze the PlantDoc dataset structure and generate EDA reports.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_plantdoc_dataset(base_dir: Path, output_dir: Path):
    """Analyze the PlantDoc dataset and generate EDA reports."""
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data storage
    data = []
    
    # Process train and test splits
    for split in ['train', 'test']:
        split_path = base_dir / split
        if not split_path.exists():
            logger.warning(f"{split} directory not found at {split_path}")
            continue
            
        # Get all class directories
        for class_dir in tqdm(sorted(split_path.iterdir()), desc=f"Processing {split}"):
            if not class_dir.is_dir():
                continue
                
            # Get all images in the class directory
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                data.append({
                    'split': split,
                    'class': class_dir.name,
                    'path': str(img_path.relative_to(base_dir)),
                    'filename': img_path.name
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        logger.error("No data found in the dataset.")
        return
    
    # Basic stats
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Number of classes: {df['class'].nunique()}")
    logger.info(f"Train samples: {len(df[df['split'] == 'train'])}")
    logger.info(f"Test samples: {len(df[df['split'] == 'test'])}")
    
    # Class distribution
    class_dist = df['class'].value_counts()
    logger.info("\nClass distribution:")
    logger.info(class_dist)
    
    # Plot class distribution
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(y='class', data=df, order=df['class'].value_counts().index)
    plt.title('Class Distribution in PlantDoc Dataset')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    
    # Plot train/test split by class
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(y='class', hue='split', data=df, order=df['class'].value_counts().index)
    plt.title('Train/Test Split by Class')
    plt.xlabel('Number of Samples')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_split.png')
    plt.close()
    
    # Save statistics to file
    stats = {
        'total_samples': len(df),
        'num_classes': df['class'].nunique(),
        'train_samples': len(df[df['split'] == 'train']),
        'test_samples': len(df[df['split'] == 'test']),
        'class_distribution': class_dist.to_dict(),
        'avg_samples_per_class': len(df) / df['class'].nunique()
    }
    
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nAnalysis complete. Reports saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PlantDoc dataset')
    parser.add_argument('--data-dir', type=str, default='data/raw/plantdoc',
                       help='Path to the PlantDoc dataset directory')
    parser.add_argument('--output-dir', type=str, default='output/reports/plantdoc_analysis',
                       help='Directory to save analysis reports')
    
    args = parser.parse_args()
    
    analyze_plantdoc_dataset(Path(args.data_dir), Path(args.output_dir))
