# Model Explainability with Grad-CAM

## Overview

This document explains the implementation and usage of Gradient-weighted Class Activation Mapping (Grad-CAM) in the KrishiSahayak project. The implementation supports various CAM methods and provides intuitive visualizations of where the model is "looking" when making predictions, a critical feature for building trustworthy AI.

## Supported CAM Methods

The system supports the following methods, which can be specified in the `xai_config`:

1. **`gradcam`**: Standard Grad-CAM, a fast and reliable baseline.
2. **`gradcam++`**: An enhanced version of Grad-CAM with better localization of objects in the image.

## Core Components

### 1. `InferenceHandler` Class

This is the primary high-level interface for making predictions and generating explanations. It manages all underlying components like the model loader and predictor.

```python
from krishisahayak.inference import InferenceHandler
from krishi_sahayak.utils.hardware import auto_detect_accelerator

# Initialize the handler with a model checkpoint
handler = InferenceHandler(
    checkpoint_path="path/to/your/model.ckpt",
    device=auto_detect_accelerator()
)

# Get predictions with explanations
result = handler.run_single(
    image_path="path/to/image.jpg",
    top_k=3
)
```

### 2. `visualize_prediction` Utility

This helper function takes the output from the InferenceHandler and creates a rich visualization.

```python
from krishi_sahayak.utils.visualization import visualize_prediction

# The result dictionary from the handler contains everything needed
visualize_prediction(
    result=result,
    explanation=result.get('explanation'),
    output_path='output/explanation.png',
    show=True
)
```

## Usage Example

This example demonstrates the end-to-end workflow for getting an explained prediction.

```python
import torch
from pathlib import Path
from krishi_sahayak.inference import InferenceHandler
from krishi_sahayak.utils.hardware import auto_detect_accelerator
from krishi_sahayak.utils.visualization import visualize_prediction

# --- 1. Setup ---
MODEL_CHECKPOINT = Path("path/to/your/model.ckpt")
IMAGE_PATH = Path("path/to/image.jpg")
OUTPUT_DIR = Path("output/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Initialize the high-level handler ---
# The handler abstracts away model loading, device placement, and predictor setup.
handler = InferenceHandler(
    checkpoint_path=MODEL_CHECKPOINT,
    device=auto_detect_accelerator()
)

# --- 3. Get predictions and explanations in a single call ---
# The handler calls the predictor internally to generate CAMs.
result = handler.run_single(
    image_path=IMAGE_PATH,
    top_k=3
)

# --- 4. Visualize the result ---
if result:
    visualize_prediction(
        result=result,
        explanation=result.get('explanation'),
        output_path=OUTPUT_DIR / f"{IMAGE_PATH.stem}_explained.png"
    )
```

## Best Practices

1. **Use the `InferenceHandler`**: For all standard use cases, interact with the system via the `InferenceHandler`. It provides the simplest and most robust interface.

2. **Target Layer Selection**: The `Predictor` class attempts to auto-detect the best convolutional layer for CAMs. For custom or unusual architectures, you may need to provide this manually during initialization.

3. **Smoothing**: For more robust explanations on noisy images, consider enabling test-time augmentation via the `aug_smooth` flag in a custom `xai_config`.

4. **Visualization**: Use alpha blending between 0.4-0.6 for optimal heatmap visibility. The default in `visualize_prediction` is 0.5.

## Troubleshooting

### Common Issues

**Poor Localization**: The heatmap doesn't highlight the correct object.

*Solution*: Try the `gradcam++` method, as it often provides better localization. You can also experiment with different `target_layers` if auto-detection is not optimal for your model.

**Error during Explanation Generation**:

*Solution*: The `InferenceHandler` and `Predictor` automatically manage the model's evaluation state and gradient requirements. This type of error is unlikely unless a highly custom, unsupported model architecture is being used. Ensure your model checkpoint contains the necessary metadata.

## References

1. [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
2. [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
3. [PyTorch Grad-CAM (jacobgil)](https://github.com/jacobgil/pytorch-grad-cam) - The underlying library that inspired the project's implementation.
