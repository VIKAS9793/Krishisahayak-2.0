# Hybrid Plant Disease Classification

This guide explains how to train a hybrid model on both PlantVillage and PlantDoc datasets for plant disease classification.

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning 1.6+
- Other dependencies listed in `requirements.txt`

## Dataset Preparation

1. **PlantVillage Dataset**:
   - Download the PlantVillage dataset and place it in `data/raw/PlantVillage`
   - The dataset should have the following structure:
     ```
     PlantVillage/
     ├── train/
     │   ├── class1/
     │   │   ├── img1.jpg
     │   │   └── ...
     │   └── class2/
     │       └── ...
     ├── val/
     │   └── ...
     └── test/
         └── ...
     ```

2. **PlantDoc Dataset**:
   - Download the PlantDoc dataset and place it in `data/raw/PlantDoc`
   - The dataset should have a similar structure to PlantVillage

3. **Preprocess the datasets**:
   - Run the data preparation script:
     ```bash
     python scripts/prepare_data.py
     ```
   - This will process the raw data and save it in `data/processed/`

## Calculating Class Weights

Before training, calculate the class weights for the hybrid dataset:

```bash
python scripts/calculate_hybrid_class_weights.py \
    --plantvillage-root data/processed/PlantVillage \
    --plantdoc-root data/processed/PlantDoc \
    --output-path data/processed/class_weights_hybrid.json
```

## Training the Hybrid Model

### Using the Configuration File

1. Update the paths in `configs/hybrid_config.yaml` to match your setup
2. Run the training script:
   ```bash
   python scripts/train_hybrid.py --config configs/hybrid_config.yaml
   ```

### Using Command Line Arguments

Alternatively, you can specify all parameters via command line:

```bash
python scripts/train_hybrid.py \
    --plantvillage-root data/processed/PlantVillage \
    --plantdoc-root data/processed/PlantDoc \
    --class-weights data/processed/class_weights_hybrid.json \
    --model-name resnet50 \
    --pretrained \
    --batch-size 32 \
    --epochs 50 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --experiment-name hybrid_experiment \
    --gpus 1
```

## Monitoring Training

- **TensorBoard**:
  ```bash
  tensorboard --logdir=logs/hybrid_experiment
  ```

- **Weights & Biases**:
  - Set `use_wandb: true` in the config file
  - Make sure to log in to W&B first:
    ```bash
    wandb login
    ```

## Resuming Training

To resume training from a checkpoint:

```bash
python scripts/train_hybrid.py \
    --resume-from checkpoints/hybrid_experiment/last.ckpt
```

## Evaluating the Model

The training script automatically evaluates the model on the test set after training. To evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/hybrid_experiment/best.ckpt \
    --data-dir data/processed/PlantVillage  # or PlantDoc for PlantDoc test set
```

## Model Architecture

The hybrid model uses a shared backbone (e.g., ResNet50) with the following features:

- **Input**: 224x224 RGB images from both datasets
- **Backbone**: Pretrained CNN (configurable)
- **Classifier**: Fully connected layer with dropout
- **Loss**: Cross-entropy with class weights for handling class imbalance
- **Optimizer**: Adam with weight decay
- **Learning Rate Scheduler**: Reduce on Plateau

## Customization

### Using a Different Backbone

You can specify any model from `torchvision.models` or `timm` by changing the `model.name` parameter in the config file.

### Data Augmentation

Modify the `transforms` section in the config file to customize data augmentation.

### Hyperparameters

Adjust the following parameters in the config file:

- Learning rate
- Batch size
- Weight decay
- Scheduler parameters
- Early stopping patience

## Troubleshooting

1. **Out of Memory (OOM) Errors**:
   - Reduce the batch size
   - Use gradient accumulation
   - Use mixed precision training

2. **Slow Training**:
   - Increase `num_workers` for faster data loading
   - Use a larger batch size if possible
   - Enable mixed precision training

3. **Poor Performance**:
   - Try a different learning rate
   - Adjust the class weights
   - Try a different model architecture
   - Check for data leakage between train/val/test splits

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
