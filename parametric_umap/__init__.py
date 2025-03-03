"""Parametric UMAP: A parametric implementation of UMAP for dimensionality reduction.

This package provides a parametric implementation of UMAP (Uniform Manifold Approximation and Projection)
that learns a neural network to perform dimensionality reduction. The model can transform new data points
without having to recompute the entire embedding.

Main Features:
    - Neural network-based implementation of UMAP
    - Fast transformation of new data points
    - Support for batch processing
    - GPU acceleration
    - Optional batch normalization and dropout
    - Flexible architecture configuration

Example:
    >>> from parametric_umap import ParametricUMAP
    >>> import numpy as np
    >>> X = np.random.rand(100, 10)
    >>> model = ParametricUMAP(n_components=2)
    >>> embedding = model.fit_transform(X)

"""

from .core import ParametricUMAP

__version__ = "0.1.1"
__all__ = ["ParametricUMAP"]
