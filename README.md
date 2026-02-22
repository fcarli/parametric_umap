# Parametric UMAP

[![CI](https://github.com/fcarli/parametric_umap/actions/workflows/ci.yml/badge.svg)](https://github.com/fcarli/parametric_umap/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/parametric-umap)](https://pypi.org/project/parametric-umap/)
[![Python versions](https://img.shields.io/pypi/pyversions/parametric-umap)](https://pypi.org/project/parametric-umap/)
[![License](https://img.shields.io/pypi/l/parametric-umap)](https://github.com/fcarli/parametric_umap/blob/main/LICENSE)

A PyTorch implementation of Parametric UMAP (Uniform Manifold Approximation and Projection) for learning low-dimensional parametric embeddings of high-dimensional data.

## Install

```bash
pip install parametric-umap
```

Or install the latest version from the repository:
```bash
pip install git+https://github.com/fcarli/parametric_umap.git
```

### GPU acceleration

The pip install pulls the default PyTorch build from PyPI. If you need a specific CUDA version, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/), then install this package.

For developers using [uv](https://docs.astral.sh/uv/), CUDA version selection is built in via extras:

```bash
# macOS / Windows (CPU automatic, no extra needed)
uv sync --extra dev --extra test --extra examples

# Linux — pick your CUDA version
uv sync --extra dev --extra test --extra examples --extra cu126

# Linux — CPU only
uv sync --extra dev --extra test --extra examples --extra cpu
```

Available CUDA extras: `cu118`, `cu121`, `cu124`, `cu126`, `cu128`.

**Apple Silicon Macs** are automatically detected and use the [MPS backend](https://developer.apple.com/metal/pytorch/) — no extra configuration needed. You can also pass `device='mps'` explicitly.

## Overview

Parametric UMAP ([original paper](https://arxiv.org/abs/2009.12981)) extends the original UMAP algorithm by learning a neural network that can map new data points to the lower-dimensional space without having to rerun the entire optimization. This (unofficial) implementation provides a flexible and efficient way to perform parametric dimensionality reduction leveraging PyTorch and FAISS.

## Features

- Neural network-based parametric mapping
- Efficient nearest neighbor computation using FAISS
- Sparse matrix operations for memory efficiency
- GPU acceleration support
- Model saving and loading capabilities
- Correlation loss term to preserve distance relationships

## Quick Start

```python
from parametric_umap import ParametricUMAP
from sklearn.datasets import make_swiss_roll
import numpy as np

# Create sample data
n_samples = 1000
X, color = make_swiss_roll(n_samples=n_samples, random_state=42)

# Initialize and fit the model (auto-detects CUDA / MPS / CPU)
pumap = ParametricUMAP(
    n_components=2,
    hidden_dim=128,
    n_layers=3,
    n_epochs=10,
)

# Fit and transform the data
embeddings = pumap.fit_transform(X)

# Transform new data
X_new = np.random.rand(100, 3)
new_embeddings = pumap.transform(X_new)
```

You can also specify the device explicitly:

```python
pumap = ParametricUMAP(device='cuda:0')   # specific CUDA GPU
pumap = ParametricUMAP(device='mps')      # Apple Silicon GPU
pumap = ParametricUMAP(device='cpu')      # force CPU
```

Note that by default the data is moved to the specified device before training to accelerate training process. However, if your GPU card cannot fit the entire dataset in memory you can override this behavior by setting the `low_memory` argument to true as follows:

```python
embeddings = pumap.fit_transform(X, low_memory=True)
```

## Key Parameters

**UMAP parameters**
- `n_neighbors`: Number of nearest neighbors for the UMAP knn graph (default: 15)
- `a`: Parameter for scaling distances between embedded points (default: 0.1)
- `b`: Parameter for controlling sharpness of the curve's transition between attraction and repulsion (default: 1.0)

**Parametric model**
- `device`: Compute device — auto-detected by default (CUDA > MPS > CPU). Pass a specific device like `'cuda:1'` or `'mps'` to override
- `n_components`: Dimension of the output embedding (default: 2)
- `hidden_dim`: Dimension of hidden layers in the MLP (default: 1024)
- `n_layers`: Number of hidden layers (default: 3)
- `correlation_weight`: Weight of the correlation loss term (default: 0.1)
- `learning_rate`: Learning rate for optimization (default: 1e-4)
- `n_epochs`: Number of training epochs (default: 10)
- `batch_size`: Training batch size (default: 32)
- `use_batchnorm`: Whether to use batch normalization in the embedding MLP (default: False)
- `use_dropout`: Whether to use dropout in the embedding MLP (default: False)

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

```bash
make install    # Install all dependencies (CPU torch)
make test       # Run tests
make lint       # Lint checks
make format     # Format code
```

## Citation

If you use this package in your research, please cite the original Parametric UMAP paper:

```bibtex
@article{sainburg2021parametric,
  title={Parametric UMAP Embeddings for Representation and Semisupervised Learning},
  author={Sainburg, Tim and McInnes, Leland and Gentner, Timothy Q},
  journal={Neural Computation},
  volume={33},
  number={11},
  pages={2881--2907},
  year={2021},
  publisher={MIT Press}
}
```

## License

BSD License
