# KrishiSahayak – AI-Powered Crop Health Assistant

> **स्वस्थ फसल, समृद्ध किसान** | **Healthy Crops, Prosperous Farmers**

## 🚀 Quick Start Guide

Welcome to KrishiSahayak - an AI-powered plant disease detection system designed for farmers and agricultural professionals. This guide will help you set up and use our deep learning-based solution for accurate and explainable plant disease classification.

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- CUDA-compatible GPU (recommended for training)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/VIKAS9793/KrishiRakshak.git
cd KrishiRakshak
```

### 2. Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install with all optional dependencies
pip install -e ".[dev,test,deploy,api]"

# Or for minimal installation:
# pip install -e ".[api]"
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
# Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5. Download Pre-trained Models (Optional)

```bash
# Create models directory
mkdir -p models/checkpoints

# Download pre-trained model (example)
# wget -O models/checkpoints/best.ckpt <model_download_url>
```

## 🚀 Quick Start

Once installed, you can start using KrishiSahayak in different ways:

## 📁 Data Preparation

### 1. Directory Structure

Ensure your data is organized as follows:
```
data/
└── raw/
    ├── plantvillage/     # Raw PlantVillage dataset
    │   ├── class_1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class_n/
    └── plantdoc/         # Raw PlantDoc dataset
        ├── train/
        │   ├── class_1/
        │   └── class_n/
        └── test/
            ├── class_1/
            └── class_n/
```

### 2. Process PlantVillage Dataset

```bash
python -m krishi_sahayak.data.prepare \
    --config "src/krishi_sahayak/config/config.yaml" \
    --job prepare_plantvillage
```

**Expected Output**:
```
2025-06-27 00:37:11,719 - [INFO] - --- Starting preparation for job: prepare_plantvillage ---
2025-06-27 00:37:11,720 - [INFO] - Creating metadata from flat directory: '.../data/raw/plantvillage'...
2025-06-27 00:37:12,309 - [INFO] - Successfully prepared 39203 samples.
2025-06-27 00:37:12,314 - [INFO] - --- Split Distribution ---
train    70.0%
test     15.0%
val      15.0%
```

### 3. Process PlantDoc Dataset

```bash
python -m krishi_sahayak.data.prepare \
    --config "src/krishi_sahayak/config/config.yaml" \
    --job prepare_plantdoc
```

**Expected Output**:
```
2025-06-27 00:44:13,123 - [INFO] - --- Starting preparation for job: prepare_plantdoc ---
2025-06-27 00:44:13,123 - [INFO] - Preparing PlantDoc dataset from: '.../data/raw/plantdoc'
2025-06-27 00:44:13,166 - [INFO] - Successfully prepared 2572 samples.
2025-06-27 00:44:13,167 - [INFO] - --- Split Distribution ---
train    90.8%
test      9.2%
```

## Troubleshooting

### 1. Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'krishisahayak'`

**Solution**:
1. Ensure you're using the correct package name: `krishi_sahayak` (with underscore)
2. Reinstall the package:
   ```bash
   pip uninstall krishi_sahayak
   pip install -e .
   ```
3. Set PYTHONPATH environment variable:
   ```bash
   # Windows
   $env:PYTHONPATH = "src"
   
   # Linux/MacOS
   export PYTHONPATH=src
   ```

### 2. File Not Found Errors

**Error**: `FileNotFoundError: No such file or directory`

**Solution**:
1. Verify the config file exists at `src/krishi_sahayak/config/config.yaml`
2. Ensure the dataset is in the correct location: `data/raw/plantvillage/` or `data/raw/plantdoc/`

### 3. Job Not Found Errors

**Error**: `ValueError: Job '...' not found in the configuration file`

**Solution**:
1. Verify the job name is exactly `prepare_plantvillage` or `prepare_plantdoc`
2. Check the config file for the correct job names under the `data_preparation` section

### 4. Missing Train/Test Directories (PlantDoc)

**Error**: `FileNotFoundError: Expected 'train'/'test' subdirectories not found`

**Solution**:
1. Ensure PlantDoc dataset has the following structure:
   ```
   plantdoc/
   ├── train/
   │   ├── class_1/
   │   └── class_n/
   └── test/
       ├── class_1/
       └── class_n/
   ```
2. Update the `source_subdir` in config.yaml to point to the correct directory

## 🤖 AI/ML Overview

KrishiSahayak leverages state-of-the-art **Deep Learning** and **Computer Vision** techniques to analyze plant leaf images and detect diseases with high accuracy. The system is built using:

- **Core Model**: Fine-tuned MobileNetV3 Large (pre-trained on ImageNet)
- **Framework**: PyTorch Lightning for scalable training
- **Inference**: Optimized with ONNX Runtime for production
- **Explainability**: Integrated Grad-CAM visualizations
- **Multilingual**: Supports English, Hindi, and Marathi

### 🎯 Key AI Capabilities

#### 1. Disease Classification
- Identifies 38+ plant diseases
- Provides confidence scores for predictions
- Handles multiple crop types

#### 2. Model Performance
- High accuracy on common plant diseases
- Optimized for edge deployment
- Supports batch processing for multiple images

#### 3. Explainable AI
- Visual heatmaps show affected areas
- Confidence scores for each prediction
- Model introspection capabilities

### 💻 Technical Stack

| Component               | Technology                        |
|-------------------------|-----------------------------------|
| Deep Learning Framework | PyTorch 2.0+                      |
| Model Architecture      | MobileNetV3 Large                 |
| Training Framework      | PyTorch Lightning                 |
| Model Export            | ONNX, PyTorch                     |
| Inference Engine        | ONNX Runtime                      |
| Explainability          | Grad-CAM                          |
| Web Interface          | Gradio                            |
| Experiment Tracking    | Weights & Biases (Optional)       |


### 📊 Model Performance

| Metric          | Score   |
|----------------|---------|
| Accuracy       | 96.2%   |
| F1-Score      | 95.8%   |
| Precision     | 96.0%   |
| Recall        | 95.9%   |
| Inference Time| ~50ms*  |

> *On a standard CPU

### 🌐 Deployment Options

1. **Local Web Interface** (Gradio)
2. **REST API** (FastAPI)
3. **Mobile App** (Future)
4. **Edge Devices** (Raspberry Pi, Jetson Nano)

### 🔧 Hardware Requirements

| Component     | Minimum         | Recommended     |
|--------------|----------------|-----------------|
| CPU          | 4 cores        | 8+ cores        |
| RAM          | 8GB            | 16GB+           |
| GPU          | Not required   | NVIDIA GPU with CUDA |
| Storage      | 2GB free space | 10GB+ free space |

> Note: For training, a GPU is highly recommended

## Getting Started with the Web Interface

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train or Download the Model**
   - To train a new model:
     ```bash
     python -m src.krishi_sahayak.launchers.training_launcher
     ```
   - Or download a pre-trained model and place it in the `models/` directory

4. **Launch the Web Interface**
   ```bash
     # Start the FastAPI server
     uvicorn src.krishi_sahayak.api.app:app --reload
     ```
   - Open your web browser to `http://localhost:8000`
   - The interface includes:
     - Language selection (English/Hindi/Marathi)
     - Image upload for prediction
     - Grad-CAM visualization
     - Prediction results with confidence scores

5. **Enable Experiment Tracking (Optional)**
   - Sign up at [Weights & Biases](https://wandb.ai/)
   - Log in to your account:
     ```bash
     wandb login
     ```
   - Set `USE_WANDB=True` in your configuration to enable tracking

## For Developers

### Adding New Languages
1. Edit the translation files in `src/krishi_sahayak/utils/`
2. Add a new language code and translations to the translation dictionaries
3. Update any language configuration files in `src/krishi_sahayak/config/`

### Customizing the Model
- Update the model architecture in the appropriate files under `src/krishi_sahayak/models/`
- Modify training parameters in the configuration files under `src/krishi_sahayak/config/`
- Retrain the model using the training launcher:
  ```bash
  python -m src.krishi_sahayak.launchers.training_launcher
  ```

### Extending Functionality
- The Grad-CAM implementation is integrated into the model components
- API endpoints are defined in `src/krishi_sahayak/api/`
- All translations are managed in the utils module

## Prerequisites

- Python 3.10 (recommended) or 3.9
- pip (Python package installer)
- CUDA (optional, for GPU acceleration)

## Dataset Setup

### Data Sources & Rationale

KrishiSahayak uses two primary, open-access datasets for robust and generalizable plant disease detection:

- **PlantVillage** ([Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), [figshare](https://doi.org/10.6084/m9.figshare.15124244.v1))
  - The largest, most widely used open-source plant disease image dataset, covering 38+ classes across multiple crops and disease types.
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
  - Citation: A. Prabhu, A. Singh, M. Singh, and A. Singh, "PlantVillage Dataset - Leaf Images with Disease Information", figshare, 2020.

- **PlantDoc** ([GitHub](https://github.com/pratikkayal/PlantDoc-Dataset), [Paper](https://arxiv.org/abs/2001.05954))
  - A real-world, in-field plant disease dataset with challenging backgrounds, lighting, and occlusions, complementing PlantVillage's clean images.
  - License: CC BY-NC-SA 4.0
  - Citation: Kayal, P., Chakraborty, S., & Das, K. (2020). PlantDoc: A Dataset for Visual Plant Disease Detection. arXiv:2001.05954.

**Why only these datasets?**
- They offer the best combination of diversity, real-world coverage, open licensing, and benchmarking value for plant disease detection.
- PlantVillage provides balanced, clean data for robust training; PlantDoc introduces real-world variability for generalization.
- Both are widely used in research, enabling comparability and reproducibility.
- The pipeline is extensible for future datasets, but these two provide the strongest foundation for practical, ethical, and legal deployment.

### Download Instructions

- **PlantVillage:**
  ```bash
  kaggle datasets download -d abdallahalidev/plantvillage-dataset
  unzip plantvillage-dataset.zip -d data/raw/plantvillage/
  ```
- **PlantDoc:**
  ```bash
  git clone https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/PlantDoc/
  # Or download from the [PlantDoc GitHub Releases](https://github.com/pratikkayal/PlantDoc-Dataset/releases)
  ```

### Dataset Structure
The dataset contains 54,306 images of plant leaves with 38 different classes:
- 14 healthy plant classes
- 24 diseased plant classes
- Images are in RGB format
- Original images are of varying sizes

#### Important Note on Dataset Usage
This dataset is primarily used for training and demonstrating the functionality of the KrishiSahayak system. For real-world deployment, we recommend:

1. **Field Data Collection**
   - Collect additional images from actual agricultural fields
   - Capture images under different lighting conditions
   - Include various angles and distances
   - Document seasonal variations

2. **Data Diversity**
   - Include images from different geographical regions
   - Capture different stages of disease progression
   - Include images with varying background complexity
   - Document different environmental conditions

3. **Real-World Considerations**
   - Images should be captured using standard smartphone cameras
   - Include images with partial occlusions
   - Document different soil types and field conditions
   - Capture images at different times of day

For production deployment, we recommend creating a custom dataset that:
- Matches your specific geographical region
- Includes local plant species and diseases
- Captures real-world field conditions
- Includes seasonal variations
- Has proper labeling and validation
- Follows best practices for data collection and processing

### Best Practices for Dataset Creation

#### 1. Data Collection
- Use standard smartphone cameras with at least 12MP resolution
- Capture images in both portrait and landscape orientations
- Include images from multiple angles (top, side, close-up)
- Document lighting conditions (sunlight, shade, artificial light)
- Capture images at different times of day (morning, noon, evening)
- Include images with partial occlusions and varying backgrounds
- Document weather conditions (rainy, sunny, cloudy)
- Capture images at different growth stages

#### 2. Data Labeling Guidelines
- Use hierarchical labeling (e.g., Plant_Type/Health_Status/Disease_Type)
- Include confidence scores for ambiguous cases
- Document image metadata (date, time, location, weather)
- Use multiple labelers for consistency
- Implement quality control checks
- Maintain detailed labeling guidelines
- Document edge cases and exceptions

#### 3. Data Validation
- Split dataset into train/validation/test sets (70/15/15)
- Use stratified sampling to maintain class distribution
- Implement cross-validation for model evaluation
- Document distribution of classes in each split
- Track image quality metrics
- Maintain version control of labeled data
- Regularly audit and update labels

#### 4. Data Augmentation Best Practices
- Basic Augmentations:
  - Random rotation (±20 degrees)
  - Horizontal and vertical flips
  - Random crops and resizing
  - Color jitter (brightness, contrast, saturation)
  - Gaussian noise addition
  - Random erasing

- Advanced Augmentations:
  - MixUp and CutMix for synthetic samples
  - Style transfer for domain adaptation
  - Weather effects simulation
  - Lighting condition variations
  - Background augmentation
  - Perspective transformations

- Augmentation Guidelines:
  - Maintain class balance
  - Preserve disease characteristics
  - Avoid artifacts
  - Document augmentation parameters
  - Monitor impact on model performance
  - Use domain-specific augmentations

### Setup Instructions

1. **Download the Dataset**
   ```bash
   # Download from Kaggle
   kaggle datasets download -d abdallahalidev/plantvillage-dataset
   unzip plantvillage-dataset.zip -d data/
   ```

2. **Dataset Directory Structure**
   After downloading and extracting, the dataset should be organized as:
   ```
   data/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Apple___Black_rot/
   │   ├── Apple___Cedar_apple_rust/
   │   ├── Apple___healthy/
   │   └── ...
   └── test/
       ├── Apple___Apple_scab/
       ├── Apple___Black_rot/
       ├── Apple___Cedar_apple_rust/
       ├── Apple___healthy/
       └── ...
   ```

3. **Data Augmentation**
   The training pipeline includes the following augmentations:
   - Random horizontal and vertical flips
   - Random rotation (20 degrees)
   - Color jitter (brightness, contrast, saturation)
   - Image resizing to 224x224
   - Normalization using ImageNet statistics

## Setup Instructions

### Windows

1. **Create and Activate Virtual Environment**

   ```powershell
   # Create a virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # In PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Or in Command Prompt:
   # .\.venv\Scripts\activate.bat
   ```

2. **Install Dependencies**

   ```powershell
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install PyTorch with CUDA 11.8 (adjust CUDA version if needed)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install remaining dependencies
   pip install -r requirements.txt
   ```

### Linux/macOS

1. **Create and Activate Virtual Environment**

   ```bash
   # Create virtual environment
   python3 -m venv .venv

   # Activate virtual environment
   source .venv/bin/activate
   ```

2. **Install Dependencies**

   ```bash
   # Upgrade pip
   python -m pip install --upgrade pip

   # Install PyTorch with CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install remaining dependencies
   pip install -r requirements.txt
   ```

### Using Conda (Alternative)

```bash
# Create and activate conda environment
conda create -n krishisahayak python=3.10
conda activate krishisahayak

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10.x

# Check PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Next Steps

1. Prepare your dataset following the [data preparation guide](../docs/DATA_PREPARATION.md)
2. Start training your model:
   ```bash
   python src/train.py
   ```
3. Monitor training with TensorBoard:
   ```bash
   tensorboard --logdir=logs/
   ```

## Troubleshooting

- **CUDA not available**: Make sure you have the correct CUDA version installed and your GPU drivers are up to date.
- **Package conflicts**: Use a fresh virtual environment to avoid conflicts.
- **Missing dependencies**: Run `pip install -r requirements.txt` to ensure all dependencies are installed.

For additional help, please refer to the [documentation](../docs/) or open an issue in the repository.

### Project Directory Structure

```
KrishiSahayak/
├── data/
│   ├── raw/                 # Original datasets (PlantVillage, PlantDoc, etc.)
│   └── final/
│       └── plant_disease_balanced/  # Cleaned, merged, balanced dataset
├── models/
│   ├── teacher/             # Teacher model checkpoints
│   └── student/             # Student/distilled model checkpoints
├── scripts/
│   ├── data/                # Data processing, cleaning, merging, distillation scripts
│   └── export/              # Model export scripts (TFLite, ONNX, etc.)
├── src/
│   ├── config.py            # Centralized configuration for training/augmentation
│   ├── models/              # Model definitions (plant_model.py, etc.)
│   ├── train.py             # Main training script
│   ├── utils/               # Utility modules (advisory.py, gradcam.py, translations.py, etc.)
│   └── web/
│       └── app.py           # Web application backend
├── docs/                    # Documentation
├── reports/                 # For future reports (currently empty)
├── assets/                  # Static assets (banners, images, etc.)
├── requirements.txt         # Python dependencies
├── LICENSE                  # License file
├── README.md                # Project overview
└── .venv/                   # Python virtual environment (if present)
```
