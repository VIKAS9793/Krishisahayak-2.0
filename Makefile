.PHONY: help install-dev install test lint format type-check clean clean-build clean-pyc clean-test docs

# Default target
help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  install           install the package in development mode"
	@echo "  install-dev       install development dependencies"
	@echo "  test              run tests quickly with the default Python"
	@echo "  test-all          run tests on every Python version with tox"
	@echo "  lint              check style with flake8 and isort"
	@echo "  format            format code with black and isort"
	@echo "  type-check        check type annotations with mypy"
	@echo "  docs              generate Sphinx HTML documentation, including API docs"
	@echo "  clean             remove all build, test, coverage and Python artifacts"
	@echo "  clean-build       remove build artifacts"
	@echo "  clean-pyc         remove Python file artifacts"
	@echo "  clean-test        remove test and coverage artifacts"

# Install the package in development mode
install:
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run tests
# Add -s to show output, -v for verbose, -x to stop on first failure
test:
	pytest tests/ -v

# Run tests on all Python versions with tox
test-all:
	tox

# Run linters
lint:
	flake8 src tests
	isort --check-only --profile black src tests

# Format code
format:
	black src tests
	isort --profile black src tests

# Type checking
type-check:
	mypy src

# Build documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Clean up
clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/
	rm -fr .mypy_cache/
	find . -name '.coverage.*' -exec rm -f {} +
