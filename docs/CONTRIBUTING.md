# Contributing to KrishiSahayak

Thank you for your interest in contributing to KrishiSahayak! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a new branch for your changes
5. Make your changes
6. Run tests and ensure they pass
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- CUDA-compatible GPU (recommended for development)
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VIKAS9793/KrishiSahayak.git
   cd KrishiSahayak
   ```

2. **Set up a virtual environment** (if not using Poetry):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   
   Using Poetry (recommended):
   ```bash
   poetry install --with dev,test,api
   ```

   Or using pip:
   ```bash
   pip install -e ".[dev,test,api]"
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests locally**:
   ```bash
   pytest
   ```

4. **Run linters and formatters**:
   ```bash
   # Auto-format code
   black .
   isort .
   
   # Run linters
   flake8
   mypy .
   ```

5. **Update documentation** if your changes affect the API or behavior

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **Flake8** for linting
- **mypy** for type checking
- **pre-commit** to run these checks before commit

Before committing, please run:
```bash
black .
isort .
flake8
mypy .
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/path/to/test_file.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention `test_*.py` for test files
- Use descriptive test function names starting with `test_`
- Use fixtures for common test data and setup

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

### Documentation Guidelines

- Keep documentation up-to-date with code changes
- Use Google-style docstrings for Python code
- Update README.md and relevant .md files for significant changes
- Add examples for new features

## Submitting a Pull Request

1. Ensure all tests pass and code is properly formatted
2. Update the CHANGELOG.md with your changes
3. Push your changes to your fork
4. Open a pull request against the `main` branch
5. Fill out the PR template with details about your changes

## Reporting Issues

When reporting issues, please include:

- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Any relevant error messages or logs

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create a release tag:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```
4. Create a GitHub release with release notes
5. Publish to PyPI (for maintainers):
   ```bash
   rm -rf dist/*
   poetry build
   poetry publish
   ```

## License

By contributing to KrishiSahayak, you agree that your contributions will be licensed under the [MIT License](LICENSE).
