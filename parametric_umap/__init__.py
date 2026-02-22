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

import os as _os
import sys as _sys

# PyTorch and FAISS each bundle their own copy of libomp on macOS.  This causes
# either an "OMP: Error #15" abort or a segfault when both libraries are loaded,
# and a deadlock during multi-threaded backward passes on CPU.
#
# Work-arounds (macOS only — Linux/CUDA is unaffected):
#   1. KMP_DUPLICATE_LIB_OK  – allow the second libomp copy to coexist.
#   2. Import FAISS first    – let its libomp initialise before PyTorch's.
#   3. OMP_NUM_THREADS=1     – prevent the corrupted thread-pool from deadlocking
#                              during torch CPU backward passes.
if _sys.platform == "darwin":
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    import faiss as _faiss  # noqa: F401, E402  # must load before torch

from .core import ParametricUMAP

__version__ = "0.1.1"
__all__ = ["ParametricUMAP"]
