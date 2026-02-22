# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parametric UMAP is a PyTorch implementation of Parametric UMAP for dimensionality reduction. It extends the original UMAP algorithm by learning a neural network that can transform new data points without recomputing the entire embedding, making it efficient for large-scale applications.

## Development Commands

### Package Management (uv)
- `uv sync --extra dev --extra test --extra examples --extra cpu` - Install all dependencies (CPU torch)
- On Linux with CUDA, replace `--extra cpu` with e.g. `--extra cu126`
- Do NOT use `--all-extras` (CUDA extras conflict with each other)
- `uv build` - Build the package for distribution
- `uv run <command>` - Run commands in the uv environment

### Code Quality
- `uv run ruff check .` - Run linting checks
- `uv run ruff format .` - Format code according to project standards
- `uv run pre-commit run --all-files` - Run all pre-commit hooks
- Ruff is configured with all rules enabled and 120-character line length

### Testing
- `uv run pytest` - Run the test suite
- Tests are in `tests/` directory
- Coverage reports generated automatically (HTML, XML, terminal)
- `examples/swiss_roll.ipynb` - Main example demonstrating usage

### CI/CD
- GitHub Actions CI runs lint + tests on push/PR to `main`
- Release workflow triggers on `v*` tags, builds and publishes to PyPI
- Pre-commit hooks enforce code quality locally

## Architecture Overview

### Core Components
- `parametric_umap/core.py` - Main `ParametricUMAP` class with fit/transform methods
- `parametric_umap/models/mlp.py` - Neural network architectures for embedding
- `parametric_umap/utils/` - Graph construction, loss functions, and utilities
- `parametric_umap/datasets/` - Data handling and preprocessing classes

### Key Design Patterns
- **Device-agnostic processing**: Automatic GPU/CPU handling throughout
- **Memory-efficient operations**: Uses sparse matrices and batch processing
- **FAISS integration**: Scalable k-nearest neighbor computation
- **Modular architecture**: Clear separation between graph construction, model training, and embedding

### Dependencies
- PyTorch 2.3.1+ (core framework)
- FAISS (efficient nearest neighbor search)
- NumPy/SciPy (numerical operations and sparse matrices)
- Managed via uv with pyproject.toml

## Code Conventions

- **Type hints** extensively used throughout (`py.typed` marker present)
- **NumPy-style docstrings** for all public methods
- **120-character line length** limit
- **Ruff formatting** with all rules enabled
- **Conventional commits** for version management
- **Device handling**: Always check and respect device placement for tensors
- **Memory management**: Use batch processing for large datasets

## Important Notes

- The neural network models expect specific input dimensions - check embedding dimensions when modifying architectures
- FAISS operations are CPU-bound even when using GPU tensors - handle device transfers appropriately
- Graph construction can be memory-intensive for large datasets - consider batch processing
- Examples in `examples/` directory serve as both documentation and integration tests
