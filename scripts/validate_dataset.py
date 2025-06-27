# scripts/validate_dataset.py
"""
KrishiSahayak - Dataset Validation and Cleaning Script

This script performs comprehensive validation and cleaning of the dataset, including:
1. Validating image file existence and integrity
2. Verifying label standardization
3. Analyzing data source bias
4. Generating a validation report and lists of problematic files.
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Setup logging to file and console
log_file_handler = logging.FileHandler('dataset_validation.log', mode='w')
log_stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[log_file_handler, log_stream_handler]
)
logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates and cleans a dataset for training."""
    
    def __init__(self, metadata_path: Path, label_map_path: Path, image_dir: Path):
        """Initialize the validator with dataset paths."""
        self.metadata_path = metadata_path
        self.label_map_path = label_map_path
        self.image_dir = image_dir
        self.df: pd.DataFrame | None = None
        self.label_map: Dict[str, List[str]] = {}
        self.report = {
            'validation_summary': {},
            'file_validation': {'missing': 0, 'corrupt': 0, 'valid': 0},
            'label_validation': {'standardized': 0, 'non_standard': 0},
            'data_source_analysis': {},
        }
        self.missing_files = []
        self.corrupt_files = []
    
    def load_data(self) -> None:
        """Load and validate the metadata and label map."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.df = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded metadata with {len(self.df)} entries from {self.metadata_path}")
        
        if not self.label_map_path.exists():
            raise FileNotFoundError(f"Label map file not found: {self.label_map_path}")
        with open(self.label_map_path, 'r') as f:
            label_data = yaml.safe_load(f)
            self.label_map = label_data.get('label_standardization_map', {})
        logger.info(f"Loaded label map with {len(self.label_map)} standardized labels")
    
    def validate_files(self) -> None:
        """Validate that all image files exist and are not corrupted."""
        if self.df is None: raise RuntimeError("Data not loaded. Call load_data() first.")
            
        valid_indices = []
        logger.info("Validating image file existence and integrity...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Validating files"):
            # Enforce the convention that paths in CSV are relative to a base directory
            img_path = self.image_dir / row['image_path']
            
            if not img_path.exists():
                self.missing_files.append(str(img_path))
                continue
                
            if not self._is_valid_image(img_path):
                self.corrupt_files.append(str(img_path))
                continue
                
            valid_indices.append(idx)
        
        # Update report with counts
        self.report['file_validation'] = {
            'missing': len(self.missing_files),
            'corrupt': len(self.corrupt_files),
            'valid': len(valid_indices),
        }
        
        # Filter the dataframe to only include valid entries
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        logger.info(f"File validation complete. Valid: {len(valid_indices)}, "
                   f"Missing: {len(self.missing_files)}, Corrupt: {len(self.corrupt_files)}")
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if a file is a valid, readable image using Pillow."""
        try:
            with Image.open(file_path) as img:
                img.load()  # This forces the image data to be read and will error on corruption
            return True
        except (IOError, UnidentifiedImageError, SyntaxError):
            return False
    
    def validate_labels(self) -> None:
        """Verify that all labels in the dataframe are standardized."""
        if self.df is None: raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Create a mapping from variant labels to standard labels
        variant_to_standard = {}
        for standard_label, variants in self.label_map.items():
            for variant in variants:
                variant_to_standard[variant] = standard_label
        
        # Check each label in the dataframe
        standardized_labels = []
        non_standard_labels = set()
        
        for label in self.df['label']:
            if label in variant_to_standard:
                standardized_labels.append(variant_to_standard[label])
            elif label in self.label_map:  # If it's already a standard label
                standardized_labels.append(label)
            else:
                non_standard_labels.add(label)
                standardized_labels.append(None)  # Will be filtered out later
        
        # Update the dataframe with standardized labels
        self.df['standardized_label'] = standardized_labels
        df_standardized = self.df.dropna(subset=['standardized_label'])
        
        num_standardized = len(df_standardized)
        num_non_standard = len(self.df) - num_standardized
        
        self.report['label_validation'] = {
            'standardized': num_standardized,
            'non_standard': num_non_standard,
            'non_standard_labels_found': sorted(list(non_standard_labels))
        }
        
        logger.info(f"Label validation complete. Standardized: {num_standardized}, Non-standard: {num_non_standard}")
        if non_standard_labels:
            logger.warning(f"Found {len(non_standard_labels)} non-standard labels. "
                         f"Consider updating the label map: {sorted(list(non_standard_labels))}")
    
    def analyze_data_sources(self) -> None:
        """Analyze the distribution of samples across different data sources."""
        if self.df is None: raise RuntimeError("Data not loaded. Call load_data() first.")
        
        self.df['source'] = self.df['image_path'].apply(lambda x: Path(x).parts[0])
        source_dist = self.df.groupby(['source', 'label']).size().unstack(fill_value=0)
        
        self.report['data_source_analysis'] = {
            'total_sources': len(self.df['source'].unique()),
            'samples_per_source': self.df['source'].value_counts().to_dict(),
            'classes_with_single_source': int((source_dist > 0).sum(axis=0).eq(1).sum()),
            'total_classes': len(source_dist.columns)
        }
        logger.info(f"Found {self.report['data_source_analysis']['total_sources']} data sources.")
    
    def generate_report(self, output_dir: Path) -> None:
        """Generate a detailed JSON report and lists of problematic files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=4)
        logger.info(f"Saved validation summary report to: {report_path}")
        
        if self.df is not None:
            cleaned_path = output_dir / 'cleaned_metadata.csv'
            self.df.to_csv(cleaned_path, index=False)
            logger.info(f"Saved cleaned metadata ({len(self.df)} rows) to: {cleaned_path}")

        # REFACTORED: Save complete lists of bad files for easier processing.
        if self.missing_files:
            missing_path = output_dir / 'missing_files.txt'
            missing_path.write_text('\n'.join(self.missing_files))
            logger.warning(f"Full list of {len(self.missing_files)} missing files saved to: {missing_path}")

        if self.corrupt_files:
            corrupt_path = output_dir / 'corrupt_files.txt'
            corrupt_path.write_text('\n'.join(self.corrupt_files))
            logger.warning(f"Full list of {len(self.corrupt_files)} corrupt files saved to: {corrupt_path}")

def main():
    """Main function to run dataset validation."""
    parser = argparse.ArgumentParser(
        description='Validate and clean a dataset for training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--metadata', type=Path, required=True, help='Path to the metadata CSV file')
    parser.add_argument('--label-map', type=Path, required=True, help='Path to the label map YAML file')
    parser.add_argument('--image-dir', type=Path, required=True, help='Base directory for images')
    parser.add_argument('--output-dir', type=Path, default=Path('./validation_output'), help='Directory to save validation results')
    args = parser.parse_args()
    
    try:
        validator = DatasetValidator(args.metadata, args.label_map, args.image_dir)
        validator.load_data()
        validator.validate_files()
        validator.validate_labels()
        validator.analyze_data_sources()
        validator.generate_report(args.output_dir)
        logger.info("Validation process completed.")
    except Exception as e:
        logger.critical(f"Validation process failed: {e}", exc_info=True)
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()