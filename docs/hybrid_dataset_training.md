# Hybrid Dataset Training

This document explains how to train models on the combined PlantVillage and PlantDoc dataset.

## Overview

The hybrid training pipeline consists of the following components:

1. **Data Preparation**: Merging metadata from PlantVillage and PlantDoc datasets
2. **Data Loading**: Using `HybridDataModule` for efficient data loading and augmentation
3. **Model Training**: Training a classification model on the hybrid dataset
4. **Evaluation**: Evaluating model performance on the test set

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning 1.6+
- Required Python packages (install with `pip install -r requirements.txt`)

## Data Preparation

Before training, you need to prepare the merged dataset:

1. Place your PlantVillage and PlantDoc datasets in the `data/raw` directory
2. Run the dataset merging script:

```bash
python scripts/merge_datasets.py \
    --plantvillage-dir data/raw/PlantVillage \
    --plantdoc-dir data/raw/PlantDoc \
    --output-file data/processed/merged_metadata.csv \
    --label-mapping configs/label_mapping.yaml
```

## Training

### Using Configuration File (Recommended)

1. Edit the configuration file at `configs/hybrid_training.yaml`
2. Start training:

```bash
python scripts/train_from_config.py --config configs/hybrid_training.yaml
```

### Using Command Line

For more control, you can run the training script directly:

```bash
python scripts/train_hybrid.py \
    --metadata-path data/processed/merged_metadata.csv \
    --image-dir data/raw \
    --model-name resnet50 \
    --pretrained \
    --batch-size 64 \
    --epochs 50 \
    --learning-rate 1e-3 \
    --gpus 1
```

### Configuration Options

Key configuration options:

- **Data**:
  - `metadata_path`: Path to the merged metadata CSV
  - `image_dir`: Base directory containing the images
  - `train_split`/`val_split`/`test_split`: Dataset split ratios

- **Model**:
  - `model_name`: Model architecture (e.g., resnet50, efficientnet_b0)
  - `pretrained`: Use pretrained weights
  - `dropout`: Dropout rate

- **Training**:
  - `batch_size`: Batch size
  - `epochs`: Number of training epochs
  - `learning_rate`: Initial learning rate
  - `weight_decay`: Weight decay (L2 penalty)
  - `lr_scheduler`: Learning rate scheduler

## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=logs/
```

Or using Weights & Biases (if enabled in config):

```bash
wandb login
```

## Model Checkpoints

Checkpoints are saved to the `checkpoints` directory by default. The best model based on validation loss is saved as `checkpoints/best.ckpt`.

## Evaluation

To evaluate a trained model on the test set:

```bash
python scripts/train_hybrid.py \
    --resume-from checkpoints/best.ckpt \
    --test
```

## Hyperparameter Tuning

For hyperparameter tuning, you can use the `--overrides` flag with `train_from_config.py`:

```bash
python scripts/train_from_config.py \
    --overrides training.learning_rate=0.001 training.batch_size=32 model.dropout=0.3
```

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
- **Slow Training**: Increase `num_workers` or use mixed precision
- **Poor Performance**: Try different learning rates, data augmentation, or model architectures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
