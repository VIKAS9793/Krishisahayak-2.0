"""
Example script demonstrating how to use the HybridDataModule for training.

This script shows how to:
1. Initialize the HybridDataModule with a merged metadata file
2. Set up data transformations
3. Create data loaders
4. Visualize sample images
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.krishisahayak.data.hybrid_data_module import HybridDataModule
from src.krishisahayak.data.transforms import TransformFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid Dataset Example')
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to the merged metadata CSV file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Root directory containing the images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def show_batch(images: torch.Tensor, labels: torch.Tensor, class_names: list, nrows: int = 4, ncols: int = 8):
    """Display a grid of images with their labels."""
    plt.figure(figsize=(20, 10))
    
    # Convert from CHW to HWC for matplotlib
    images = images.permute(0, 2, 3, 1)
    
    # Undo normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    for i in range(min(len(images), nrows * ncols)):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i].item()])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize the data module
    data_module = HybridDataModule(
        metadata_path=args.metadata_path,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        train_val_test_split=(0.7, 0.15, 0.15),
        class_weights_path=None,  # Will be calculated automatically
        seed=args.seed
    )
    
    # Setup the data module (loads and splits the data)
    data_module.setup()
    
    # Get the data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Print dataset statistics
    logger.info(f"Number of training samples: {len(data_module.train_dataset)}")
    logger.info(f"Number of validation samples: {len(data_module.val_dataset)}")
    logger.info(f"Number of test samples: {len(data_module.test_dataset)}")
    logger.info(f"Number of classes: {data_module.num_classes}")
    logger.info(f"Class names: {data_module.classes}")
    logger.info(f"Class weights: {data_module.class_weights}")
    
    # Visualize a batch of training data
    train_iter = iter(train_loader)
    images, labels = next(train_iter)
    
    logger.info(f"Batch shape: {images.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Show the batch
    show_batch(images, labels, data_module.classes)
    
    # Example of iterating through the data loaders
    logger.info("Iterating through training batches...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Your training code here
        if batch_idx >= 2:  # Just show a couple of batches
            break
        logger.info(f"Batch {batch_idx}: {images.shape}, {labels.shape}")

if __name__ == "__main__":
    main()
