# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parametric UMAP is a PyTorch implementation of Parametric UMAP for dimensionality reduction. It extends the original UMAP algorithm by learning a neural network that can transform new data points without recomputing the entire embedding, making it efficient for large-scale applications.

## Development Commands

### Package Management (uv)
- `uv sync` - Install dependencies and development environment
- `uv build` - Build the package for distribution
- `uv run <command>` - Run commands in the uv environment

### Code Quality
- `ruff check .` - Run linting checks
- `ruff format .` - Format code according to project standards
- Ruff is configured with all rules enabled and 120-character line length

### Testing
- No formal test suite; testing is done through example notebooks
- `examples/swiss_roll.ipynb` - Main example demonstrating usage
- Run examples to verify functionality after changes

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
- PyTorch 2.6.0+ (core framework)
- FAISS (efficient nearest neighbor search)
- NumPy/SciPy (numerical operations and sparse matrices)
- Managed via uv with pyproject.toml

## Code Conventions

- **Type hints** extensively used throughout
- **NumPy-style docstrings** for all public methods
- **120-character line length** limit
- **Ruff formatting** with all rules enabled
- **Device handling**: Always check and respect device placement for tensors
- **Memory management**: Use batch processing for large datasets

## Important Notes

- The neural network models expect specific input dimensions - check embedding dimensions when modifying architectures
- FAISS operations are CPU-bound even when using GPU tensors - handle device transfers appropriately
- Graph construction can be memory-intensive for large datasets - consider batch processing
- Examples in `examples/` directory serve as both documentation and integration tests