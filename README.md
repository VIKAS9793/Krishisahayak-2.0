# KrishiSahayak – AI-Powered Crop Health Assistant
*स्वस्थ फसल, समृद्ध किसान | Healthy Crops, Prosperous Farmers*

<div align="center">
  <img src="https://raw.githubusercontent.com/VIKAS9793/Krishisahayak-2.0/master/assets/banners/banner.png" alt="KrishiSahayak Banner" width="100%">
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-API%20%7C%20QuickStart-4B0082)](docs/)

</div>

KrishiSahayak is a state-of-the-art, multi-modal AI system designed to provide accurate and accessible plant disease diagnostics for farmers. It leverages advanced deep learning techniques, including hybrid sensor fusion and generative models, to deliver reliable results even in challenging real-world conditions.

---

## 🚀 Key Features

* **Hybrid Model Architecture**: Combines RGB and multispectral (MS) data with a confidence-based fallback system.
* **GAN-Powered NIR Synthesis**: Uses a Pix2Pix GAN to synthesize Near-Infrared channels when real MS data is unavailable.
* **Comprehensive Dataset Support**:
  - [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset): 54,305 lab-condition images across 38 plant-disease combinations
  - [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset): 2,598 real-field images across 27 plant species with complex backgrounds
* **Unified Data Pipeline**: Processes diverse datasets with standardized labeling and validation.
* **Production-Ready API**: A robust FastAPI service with structured logging, dependency injection, and support for containerized deployment.
* **Comprehensive Testing & QA**: Includes a full test suite and automated checks for documentation integrity and code quality.

---

## 🏗️ Project Structure

```
KrishiRakshak/
├── .github/                  # GitHub related files (CI/CD, issue templates)
├── configs/                  # Configuration files
├── data/                     # Data files
├── docs/                     # Documentation
├── examples/                 # Example usage
├── models/                   # Trained models
├── output/                   # Output files
├── reports/                  # Reports and analysis
├── scripts/                  # Utility scripts
├── src/                      # Source code
│   └── krishi_sahayak/       # Main package
│       ├── api/              # API endpoints and FastAPI app
│       ├── config/           # Configuration management
│       ├── data/             # Data processing
│       ├── inference/        # Model inference code
│       ├── launchers/        # Script launchers
│       ├── models/           # Model architectures
│       ├── pipelines/        # Training pipelines
│       └── utils/            # Utility functions
├── tests/                    # Test files
├── .gitignore.bk
├── .pre-commit-config.yaml   # Pre-commit hooks
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile               # Docker configuration
├── Dockerfile.api           # API specific Dockerfile
├── docker-compose.yaml      # Docker compose for services
├── docker-compose.dev.yaml  # Development docker-compose
├── Makefile                # Common commands
├── MANIFEST.in             # Package data files
├── poetry.lock             # Dependencies lock file
└── pyproject.toml          # Project metadata and dependencies
```

## 📚 Documentation

For complete guidance on setup, usage, and project architecture, please refer to our detailed documentation:

* **[Quick Start Guide](docs/QUICKSTART.md)**: The fastest way to get the project up and running.
* **[API Reference](docs/API_README.md)**: Detailed API documentation, endpoints, and deployment guides.
* **[Project Architecture](docs/ARCHITECTURE.md)**: An in-depth look at the system design and components.
* **[Deployment Guide](docs/DEPLOYMENT.md)**: Instructions for deploying the service in production.
* **[Project Roadmap](ROADMAP.md)**: Our plans for the future of KrishiSahayak.

---

## 🧪 Testing and Quality Assurance

We enforce a high standard of code quality and documentation integrity.

### Running Tests

To run the complete test suite:
```bash
pytest
```

### Documentation Checker

This project includes a custom script to find broken links and code references in our documentation. It is run automatically in CI, but you can also run it locally:

```bash
python scripts/doc_ref_checker.py --doc docs/ARCHITECTURE.md
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to set up your development environment and submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
