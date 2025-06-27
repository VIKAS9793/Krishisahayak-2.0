# KrishiSahayak - Data Documentation

This document provides comprehensive information about the datasets used in the KrishiSahayak project, including dataset structure, preparation steps, and analysis.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Collection](#data-collection)
3. [Data Preparation](#data-preparation)
4. [Dataset Statistics](#dataset-statistics)
5. [Data Preprocessing](#data-preprocessing)
6. [Data Augmentation](#data-augmentation)
7. [Dataset Splits](#dataset-splits)
8. [Data Versioning](#data-versioning)
9. [Data Privacy and Ethics](#data-privacy-and-ethics)

## Dataset Overview

KrishiSahayak uses a hybrid dataset combining multiple plant disease datasets:

- **PlantVillage Dataset**: Contains images of healthy and diseased crop leaves
- **PlantDoc Dataset**: Additional plant disease images with varied backgrounds
- **Custom Data**: Any user-contributed or custom-collected images

## Data Collection

### Source Datasets

1. **PlantVillage**
   - Source: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
   - License: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication
   - Images: 54,305 images of healthy and diseased plant leaves
   - Classes: 38 plant-disease combinations
   - Resolution: Varies, typically around 256×256 pixels
   - Environment: Lab conditions with controlled backgrounds

2. **PlantDoc**
   - Source: [PlantDoc Dataset on GitHub](https://github.com/pratikkayal/PlantDoc-Dataset)
   - License: MIT License
   - Images: 2,598 images across 27 plant species
   - Classes: 17 disease classes plus healthy samples
   - Resolution: Varies, typically higher resolution than PlantVillage
   - Environment: Real-field conditions with complex backgrounds

### Data Organization

All datasets follow this directory structure:

```
data/
├── raw/                    # Original, unprocessed data
│   ├── plantvillage/      # PlantVillage dataset
│   └── plantdoc/          # PlantDoc dataset
└── processed/             # Processed and cleaned data
    ├── metadata_plantvillage.csv
    ├── metadata_plantdoc.csv
    └── merged_metadata.csv
```

## Data Preparation

### 1. Merging Datasets

To combine multiple datasets into a unified training set:

```bash
python scripts/merge_datasets.py \
    --inputs data/processed/metadata_plantvillage.csv \
             data/processed/metadata_plantdoc.csv \
    --output data/processed/merged_metadata.csv \
    --label-map configs/label_map.yaml
```

### 2. Label Standardization

Labels are standardized using the mapping defined in `configs/label_map.yaml`:

```yaml
label_standardization_map:
  tomato_healthy: ["tomato_healthy", "healthy_tomato"]
  tomato_late_blight: ["tomato_late_blight", "late_blight_tomato"]
  # ... more mappings
```

## Dataset Statistics

### Class Distribution

Run the EDA script to generate detailed statistics:

```bash
python scripts/eda_hybrid_dataset.py
```

This will generate visualizations in `reports/figures/` including:
- Class distribution
- Dataset splits
- Image dimensions and aspect ratios

### Key Metrics

- **Total Samples**: 41,775 images
- **Number of Classes**: 38
- **Average Image Size**: Varies by dataset
- **Class Imbalance**: Present (see generated reports for details)

## Data Preprocessing

All images undergo the following preprocessing steps:

1. **Resizing**: Images are resized to 256×256 pixels
2. **Normalization**: Pixel values are normalized to [0, 1]
3. **Color Space**: Converted to RGB
4. **Label Encoding**: Class labels are encoded as integers

## Data Augmentation

The following augmentations are applied during training:

- Random horizontal and vertical flips
- Random rotations (up to 15 degrees)
- Random brightness and contrast adjustments
- Random zoom (up to 10%)

Configuration can be found in `configs/augmentations/`.

## Dataset Splits

The data is split into three sets:

- **Training**: 70% of the data
- **Validation**: 15% of the data
- **Test**: 15% of the data

Splits are stratified to maintain class distribution.

## Data Versioning

All datasets are versioned using DVC (Data Version Control). To track changes:

```bash
dvc add data/raw/plantvillage
dvc add data/raw/plantdoc
dvc push
```

## Data Privacy and Ethics

- All data is sourced from publicly available datasets
- No personally identifiable information (PII) is collected
- Model predictions should be used for guidance only and not as a substitute for professional agricultural advice

## Troubleshooting

### Common Issues

1. **Missing Images**
   - Verify image paths in the metadata files
   - Ensure the working directory is set to the project root

2. **Class Imbalance**
   - Consider using class weights during training
   - Apply data augmentation to minority classes

3. **Memory Issues**
   - Reduce batch size
   - Use a smaller image resolution
   - Enable mixed-precision training
