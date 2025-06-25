# Hybrid RGB-Multispectral Model Guide

This guide provides detailed information about the hybrid RGB-Multispectral model in KrishiSahayak, including its architecture, configuration, and usage.

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

The hybrid model combines RGB and multispectral (MS) data to improve plant disease classification accuracy. It's designed to work in three modes:
- **RGB-only**: Standard RGB image classification
- **RGB + Real MS**: Uses actual multispectral data when available
- **RGB + Synthetic MS**: Falls back to synthetic MS data when real MS data is unavailable

## Model Architecture

### Key Components

#### Input Specifications
- **RGB Input**: 3 channels (224×224 pixels)
- **MS Input**: 4 channels (224×224 pixels)
- **Normalization**: ImageNet mean and std
- **Data Type**: Float32 (normalized to [0,1])

#### 1. RGB Branch
- **Backbone**: EfficientNetV2-RW-S (pretrained on ImageNet-1k)
- **Feature Extraction**:
  - Stem: 3×3 conv with stride 2
  - 7×7 MBConv blocks with SE
  - Output: 1280-D feature vector

#### 2. MS Branch
- **Backbone**: EfficientNetV2-RW-S (random init)
- **Input Adapter**: 1×1 conv to project to 3 channels
- **Feature Extraction**: Same as RGB branch
- **Output**: 1280-D feature vector

#### 3. Feature Fusion Methods

##### Concatenation (Default)
```python
fused_features = torch.cat([rgb_features, ms_features], dim=1)
# Output dim: 2560 (1280 + 1280)
```

##### Element-wise Addition
```python
fused_features = rgb_features + ms_features  # Requires same dimensions
# Output dim: 1280
```

##### Attention-based Fusion
```python
attention = torch.sigmoid(self.attention(torch.cat([rgb_features, ms_features], dim=1)))
fused_features = attention * rgb_features + (1 - attention) * ms_features
# Output dim: 1280
```

#### 4. Classification Head
- Global Average Pooling
- Layer Normalization
- Dropout (p=0.2)
- Fully Connected Layer (→ num_classes)

## Performance Characteristics

### Expected Performance

Model performance will vary based on your specific dataset and hardware. The hybrid model is designed to provide improved accuracy over RGB-only models by incorporating multispectral information.

### Resource Requirements

| Resource        | Training (per GPU) | Inference (CPU) | Inference (GPU) |
|----------------|-------------------|-----------------|-----------------|
| RAM            | 16GB+            | 4GB+            | 2GB+            |
| VRAM           | 8GB+             | N/A             | 4GB+            |
| Disk Space     | 10GB             | 500MB           | 500MB           |
| OS             | Linux/Windows     | Any             | Any             |

### Resource Requirements

| Resource        | Training (per GPU) | Inference (CPU) | Inference (GPU) |
|----------------|-------------------|-----------------|-----------------|
| RAM            | 16GB+            | 4GB+            | 2GB+            |
| VRAM           | 8GB+             | N/A             | 4GB+            |
| Disk Space     | 10GB             | 500MB           | 500MB           |
| OS             | Linux/Windows     | Any             | Any             |

## Configuration

The model is configured through the main `config.yaml` file. Key parameters:

```yaml
model:
  model_type: "hybrid"
  num_classes: 38
  backbone:
    name: "efficientnetv2_rw_s"
    pretrained: true
    freeze: false
  head:
    dropout_rate: 0.1
  fusion:
    method: "concat"  # Options: concat, add, attention
    hidden_size: 256

data:
  ms_data:
    enabled: true
    use_synthetic: true
    min_bands: 3
    max_bands: 8
    band_names: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]
    check_band_consistency: true
```

## Training

### Data Preparation

1. **Directory Structure**
   ```
   data/
   ├── images/           # RGB images
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── multispectral/   # Real MS data
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── synthetic_multispectral/  # Synthetic MS data
       ├── train/
       ├── val/
       └── test/
   ```

2. **Metadata CSV**
   The CSV should contain these columns:
   - `image_path`: Path to RGB image
   - `ms_path`: Path to corresponding MS image (optional)
   - `label`: Class label (integer)
   - `split`: Dataset split ('train', 'val', or 'test')

### Starting Training

```bash
python scripts/train.py --config config.yaml
```

### Training Options

- `--use_ms`: Enable MS data processing
- `--synthetic_ms`: Fall back to synthetic MS data
- `--fusion_method`: Choose fusion strategy (concat/add/attention)
- `--batch_size`: Adjust based on available GPU memory

## Inference

### Loading the Model

```python
from src.models.hybrid import HybridModel

# Initialize model
model = HybridModel(
    num_classes=38,
    use_ms=True,
    ms_channels=4,
    fusion_method='concat'
)

# Load weights
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

### Making Predictions

```python
# Single image prediction
def predict(rgb_image, ms_image=None):
    with torch.no_grad():
        batch = {'image': rgb_image}
        if ms_image is not None:
            batch['ms_data'] = ms_image
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        return probs
```

## Troubleshooting

### Common Issues

1. **MS Data Not Found**
   - Check file paths in metadata CSV
   - Verify `ms_dir` in config points to the correct directory
   - Ensure `use_ms` is set to `true` in config

2. **Dimension Mismatch**
   - Verify `ms_channels` matches your data
   - Check that input images have the expected dimensions

3. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

## Best Practices

1. **Data Augmentation**
   - Use the same augmentation for RGB and MS data
   - Consider band-specific normalization for MS data

2. **Training Strategy**
   - Start with RGB-only training
   - Gradually unfreeze MS branch
   - Use learning rate warmup

3. **Model Selection**
   - Start with a small backbone for quick experimentation
   - Use larger backbones for better accuracy
   - Consider model size for deployment constraints

4. **Monitoring**
   - Track both RGB and MS branch losses
   - Monitor feature similarity between branches
   - Visualize attention maps for interpretability
