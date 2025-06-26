"""
A robust command-line utility to visualize data augmentations.
"""
import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from PIL import Image

# REFACTORED: Corrected import paths
from krishisahayak.data.dataset import UnifiedPlantDataset
from krishisahayak.utils.transforms import TransformConfig, TransformFactory

# --- (The rest of the file's logic is unchanged) ---
# ...

# Set up a logger for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def visualize_augmentations(
    dataset: UnifiedPlantDataset,
    output_dir: Path,
    norm_stats: Dict[str, Any],
    num_samples: int = 5,
    num_augmentations: int = 4,
) -> None:
    """
    Generates and saves visualizations of original vs. augmented images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        logger.warning("Dataset is empty. Cannot generate visualizations.")
        return

    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    mean = np.array(norm_stats.get('mean', [0.0, 0.0, 0.0]))
    std = np.array(norm_stats.get('std', [1.0, 1.0, 1.0]))

    logger.info(f"Generating visualizations for {len(indices)} random samples...")
    for idx in indices:
        fig = None
        full_image_path = None
        try:
            # Get the original image path directly from the dataframe held by the dataset
            original_row = dataset.df.iloc[idx]
            relative_image_path = original_row['image_path']
            class_name = str(original_row.get('label', 'unknown'))

            # The dataset's config holds the authoritative base directory
            full_image_path = dataset.config.data_dir / relative_image_path
            
            if not full_image_path.is_file():
                logger.warning(f"Image not found at expected path: {full_image_path}")
                continue

            original_image = Image.open(full_image_path).convert("RGB")
            
            fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(20, 4))
            fig.suptitle(f"Sample: {Path(relative_image_path).name}  |  Class: {class_name}", fontsize=16)
            axes[0].imshow(original_image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            # Generate and display augmented versions
            for i in range(1, num_augmentations + 1):
                # We call dataset[idx] multiple times to get different random augmentations
                augmented_item = dataset[idx]
                augmented_image_tensor = augmented_item['image']
                
                # Denormalize for correct visualization
                augmented_image = augmented_image_tensor.permute(1, 2, 0).numpy()
                denormalized_image = (augmented_image * std) + mean
                denormalized_image = np.clip(denormalized_image, 0, 1)
                
                axes[i].imshow(denormalized_image)
                axes[i].set_title(f"Augmented {i}")
                axes[i].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            safe_class_name = "".join(c for c in class_name if c.isalnum()).rstrip()
            save_path = output_dir / f"augmentations_sample_{idx}_{safe_class_name}.png"
            plt.savefig(save_path)

        except Exception as e:
            logger.error(f"Failed to visualize sample index {idx} (Path: {full_image_path}): {e}", exc_info=True)
        finally:
            if fig:
                plt.close(fig)

    logger.info(f"All visualizations saved to: {output_dir}")


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    logger.info("Starting augmentation verification process...")

    try:
        # 1. Load augmentation config into a validated Pydantic model
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        transform_config = TransformConfig(**config_dict)
        
        # 2. Use the canonical TransformFactory to create the pipeline
        factory = TransformFactory(transform_config)
        train_transform = factory.get_train_transform()
        logger.info("Successfully created augmentation pipeline using TransformFactory.")

        # 3. Load metadata and filter in memory (no temporary file)
        df = pd.read_csv(args.metadata_path)
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        logger.info(f"Loaded metadata. Found {len(train_df)} training samples.")

        if train_df.empty:
            logger.error("No training samples found in metadata. Aborting.")
            return

        # 4. Create dataset configuration
        # NOTE: This assumes UnifiedPlantDataset can accept a 'dataframe' argument.
        dataset_config = DatasetConfig(
            csv_path=args.metadata_path, # Pass for context, though not used if dataframe is provided
            data_dir=args.image_base_dir,
            task='classification',
            split='train' # The dataframe is already filtered, but this maintains consistency
        )
        
        # 5. Instantiate the dataset, passing the filtered dataframe directly
        dataset = UnifiedPlantDataset(
            config=dataset_config,
            transform=train_transform,
            dataframe=train_df 
        )
        logger.info(f"Dataset created successfully in-memory with {len(dataset)} samples.")

        # 6. Run visualization
        visualize_augmentations(
            dataset=dataset,
            output_dir=args.output_dir,
            norm_stats=transform_config.rgb_norm.model_dump(),
            num_samples=args.num_samples
        )
        logger.info("Verification process completed successfully.")
    
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify and visualize data augmentations.")
    parser.add_argument(
        "--config-path", type=Path, required=True,
        help="Path to the augmentation YAML config file (e.g., using TransformConfig schema)."
    )
    parser.add_argument(
        "--metadata-path", type=Path, required=True,
        help="Path to the project's master metadata CSV file."
    )
    parser.add_argument(
        "--image-base-dir", type=Path, required=True,
        help="The base directory where image paths in the CSV are relative to."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to save the visualization images."
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of random samples to visualize."
    )
    
    args = parser.parse_args()
    main(args)