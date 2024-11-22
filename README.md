# Parametric UMAP
A PyTorch implementation of Parametric UMAP (Uniform Manifold Approximation and Projection) for learning low-dimensional parametric embeddings of high-dimensional data.

## Install
A proper installation of [PyTorch](https://pytorch.org) (possibly with GPU accelaration) is suggested before installing this package. The package can then be installed with

```bash
pip install parametric_umap
```

To instead install the latest version of this repository use
```bash
pip install git+https://github.com/mr-fcharles/parametric_umap.git
```

## Overview
Parametric UMAP ([original paper](https://arxiv.org/abs/2009.12981)) extends the original UMAP algorithm by learning a neural network that can map new data points to the lower-dimensional space without having to rerun the entire optimization. This (unofficial) implementation provides a flexible and efficient way to perform parametric dimensionality reduction leveraging PyTorch and FAISS.

## Features
- Neural network-based parametric mapping
- Efficient nearest neighbor computation using FAISS
- Sparse matrix operations for memory efficiency
- GPU acceleration support
- Model saving and loading capabilities
- Correlation loss term to preserve distance relationships

## Quick start
```python
from parametric_umap import ParametricUMAP
from sklearn.datasets import make_swiss_roll
import numpy as np

# Create sample data
n_samples = 1000
X, color = make_swiss_roll(n_samples=n_samples, random_state=42)

# Initialize and fit the model
pumap = ParametricUMAP(
    device='cuda:0'
    n_components=2,
    hidden_dim=128,
    n_layers=3,
    n_epochs=10
)

# Fit and transform the data
embeddings = pumap.fit_transform(X)

# Transform new data
X_new = np.random.rand(100, 3)
new_embeddings = pumap.transform(X_new)
```

Note that by default the data is moved to the specified device before training to accelerate training process. However, if your GPU card cannot fit the entire dataset in memory you can override this behavior by setting the `low_memory` argument to true as follows:

```python
embeddings = pumap.fit_transform(X,low_memory=True)
```

## Key Parameters
Hyperparameters default values follow the [original UMAP implementation](https://umap-learn.readthedocs.io/en/latest/)

**UMAP parameters**
- `a`: parameter for scaling distances between embedded points
- `b`: parameter for controlling sharpness of the curve's transition between attraction and repulsion
- `n_neighbors`: number of neighbors to compute for UMAP knn graph (default: 15)

**Parametric model**
- `device`: 'cpu' or 'cuda' (also specific device 'cuda:1', automatically uses GPU acceleration if GPU card is detected)
- `n_components`: Dimension of the output embedding (default: 2)
- `hidden_dim`: Dimension of hidden layers in the MLP (default: 1024)
- `n_layers`: Number of hidden layers (default: 3)
- `n_neighbors`: Number of nearest neighbors (default: 15)
- `correlation_weight`: Weight of the correlation loss term (default: 0.1)
- `learning_rate`: Learning rate for optimization (default: 1e-4)
- `n_epochs`: Number of training epochs (default: 10)
- `batch_size`: Training batch size (default: 32)
- `use_batchnorm`: Whether to use batch normalization in the embedding MLP (default: False)
- `use_dropout`: Whether to use dropout in the embedding MLP (default: False)