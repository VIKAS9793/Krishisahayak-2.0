# KrishiSahayak – AI-Powered Crop Health Assistant
*स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers*

<div align="center">
  <img src="https://raw.githubusercontent.com/VIKAS9793/Krishisahayak-2.0/master/assets/banners/banner.png" alt="KrishiSahayak Banner" width="100%">
</div>

<div align="center">
<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-API%20%7C%20QuickStart-4B0082)](https://github.com/VIKAS9793/KrishiSahayak#-documentation)

</div>
</div>

KrishiSahayak is a state-of-the-art, multi-modal AI system designed to provide accurate and accessible plant disease diagnostics for farmers. It leverages advanced deep learning techniques, including hybrid sensor fusion and generative models, to deliver reliable results even in challenging real-world conditions.

---

## 🚀 Key Features

* **Hybrid Model Architecture**: Combines RGB and multispectral (MS) data processing with a confidence-based fallback system for robust predictions.
* **GAN-Powered NIR Generation**: Uses Pix2Pix GAN to synthesize NIR channels when multispectral data is unavailable.
* **Unified Data Pipeline**: Handles both RGB and MS data with built-in error handling and augmentation.
* **Production-Ready Inference**: Optimized for deployment with support for batch processing and hardware acceleration.
* **Comprehensive Testing**: Includes unit, integration, and API tests with continuous integration.

---

## 🏗️ Project Structure

```
KrishiSahayak/
├── configs/                  # Configuration files
│   └── augmentations/        # Data augmentation configurations
├── data/                     # Local data storage (gitignored)
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed data and metadata
├── src/                      # Source code
│   └── krishi_sahayak/       # Main package
│       ├── api/              # FastAPI application endpoints
│       ├── config/           # Configuration management
│       ├── data/             # Data loading and processing
│       │   ├── datasets/     # PyTorch Dataset implementations
│       │   └── transforms/   # Data augmentation
│       ├── inference/        # Model serving components
│       ├── models/           # Model implementations
│       │   ├── base/         # Base model classes
│       │   ├── core/         # Core model architectures
│       │   └── gan/          # GAN implementations
│       └── utils/            # Utility functions
├── tests/                    # Test suite
│   ├── api/                  # API tests
│   ├── integration/          # Integration tests
│   └── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── unit/                 # Unit tests
├── .github/                  # GitHub configurations
├── docs/                     # Documentation files
├── .env.example              # Environment variables template
├── .pre-commit-config.yaml   # Code quality hooks
└── pyproject.toml            # Project metadata and dependencies
```

## 🚀 Quick Start

Get started with KrishiSahayak in minutes:

1. **Clone and set up**
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -e ".[dev,test,deploy,api]"
   ```

2. **Run the API**
   ```bash
   uvicorn src.krishi_sahayak.api.main:app --reload
   ```
   Then visit `http://localhost:8000/docs` to explore the API.

For detailed setup and advanced usage, see [QUICKSTART.md](docs/QUICKSTART.md).

## 🧩 Features

- **Plant Disease Detection**: Identify diseases from plant leaf images
- **RESTful API**: Easy integration with web and mobile apps
- **Model Training**: Train custom models with your dataset
- **Pre-trained Models**: Get started quickly with our pre-trained models
- **Scalable**: Designed to work on both CPU and GPU

## 📚 Documentation

- [Quick Start](docs/QUICKSTART.md) - Get up and running quickly
- [API Reference](API_README.md) - Detailed API documentation
- [Architecture](docs/ARCHITECTURE.md) - System design and components
- [Deployment](docs/DEPLOYMENT.md) - Production deployment guides

## 🔍 Documentation Checker

The project includes a documentation reference checker to ensure all file and symbol references in the documentation are valid. The checker can be run locally to verify documentation integrity before committing changes.

### Running the Documentation Checker

```bash
# Check all documentation files
python scripts/doc_ref_checker.py --doc docs/ARCHITECTURE.md
python scripts/doc_ref_checker.py --doc docs/QUICKSTART.md
# Add other documentation files as needed

# Check a specific file with custom ignored symbols
python scripts/doc_ref_checker.py --doc docs/DEPLOYMENT.md --ignore-symbols black,isort
```

The checker verifies that:
- All referenced Python files exist in the project
- All referenced classes and functions exist in the codebase
- No broken or outdated references are present

### Automatic Checks in CI

Documentation references are automatically checked on pull requests to the main branch. The CI will fail if any broken references are found.

## 🧪 Testing

Run the test suite:

```bash
pytest
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
