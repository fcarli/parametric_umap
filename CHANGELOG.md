# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2025-06-27

### Added
- Semantic versioning framework with automated release workflow
- pytest test suite with coverage reporting
- Makefile for common development tasks

### Changed
- Migrated to uv for package management
- Switched to Python 3.11 as minimum version
- CUDA-accelerated PyTorch auto-installed on Linux

## [0.1.0] - 2025-01-15

### Added
- Initial release of parametric-umap
- Core `ParametricUMAP` class with fit/transform API
- MLP-based neural network encoder
- FAISS-accelerated k-nearest neighbor computation
- Correlation loss term for preserving distance relationships
- GPU acceleration support
- Model saving and loading
- Swiss roll example notebook
