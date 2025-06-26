"""
KrishiSahayak - Local Annotation & Correction Tool

A command-line utility to allow users to label or correct images and export
the results as a CSV file suitable for retraining loops.
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def run_annotation_session(args: argparse.Namespace):
    """Orchestrates the annotation workflow."""
    image_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    labels = args.labels.split(',')

    if not image_dir.is_dir():
        logger.critical(f"Input directory not found: {image_dir}")
        sys.exit(1)

    # Load existing annotations or create a new DataFrame
    if output_csv.exists():
        logger.info(f"Loading existing annotations from: {output_csv}")
        df = pd.read_csv(output_csv)
        processed_images = set(df['image_path'])
    else:
        df = pd.DataFrame(columns=['image_path', 'label'])
        processed_images = set()

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    try:
        for i, img_path in enumerate(image_paths):
            relative_path = str(img_path.relative_to(image_dir.parent))
            if relative_path in processed_images:
                continue

            print("\n" + "="*50)
            print(f"Annotating image {i+1}/{len(image_paths)}: {img_path.name}")
            
            # Optionally show the image
            if not args.headless:
                try:
                    Image.open(img_path).show()
                except Exception as e:
                    logger.warning(f"Could not display image: {e}")

            print("Please select a label:")
            for idx, label in enumerate(labels):
                print(f"  [{idx}] {label}")
            print("  [s] Skip this image")
            print("  [q] Quit and save")

            while True:
                choice = input("Enter your choice (number, 's', or 'q'): ").lower().strip()
                if choice == 'q':
                    raise KeyboardInterrupt # Graceful exit
                if choice == 's':
                    logger.info(f"Skipping {img_path.name}")
                    break
                try:
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(labels):
                        chosen_label = labels[choice_idx]
                        new_row = pd.DataFrame([{'image_path': relative_path, 'label': chosen_label}])
                        df = pd.concat([df, new_row], ignore_index=True)
                        logger.info(f"Labeled '{img_path.name}' as '{chosen_label}'.")
                        break
                    else:
                        print("Invalid number. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number, 's', or 'q'.")
        
    except KeyboardInterrupt:
        logger.info("\nQuit signal received.")
    finally:
        logger.info(f"Saving {len(df)} annotations to {output_csv}...")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("Save complete. Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KrishiSahayak - Local Annotation Tool")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of images to label.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated string of possible class labels (e.g., 'healthy,rust,scab').")
    parser.add_argument("--headless", action='store_true', help="Run without attempting to display images.")
    
    args = parser.parse_args()
    run_annotation_session(args)