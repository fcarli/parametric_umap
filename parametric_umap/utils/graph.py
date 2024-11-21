import numpy as np
import faiss
from scipy import sparse

import faiss
import numpy as np
from scipy import sparse
import unittest
import torch

def compute_sigma_i(X, k, tol=1e-5, max_iter=100):
    """
    Compute sigma_i for each sample in the dataset using FAISS for k-nearest neighbors.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features), dataset.
    - k: int, number of nearest neighbors.
    - tol: float, tolerance for binary search.
    - max_iter: int, maximum iterations for binary search.

    Returns:
    - sigma: np.ndarray of shape (n_samples,), computed sigma_i for each sample.
    - rho: np.ndarray of shape (n_samples,), distance to the nearest neighbor for each sample.
    - distances: np.ndarray of shape (n_samples, k), Euclidean distances to the k nearest neighbors for each sample.
    - neighbors: np.ndarray of shape (n_samples, k), indices of the k nearest neighbors for each sample.
    """
    X = X.astype(np.float32)  # Ensure data is float32 for FAISS
    n_samples, n_features = X.shape

    # Step 1: Use FAISS to compute k-nearest neighbors
    index = faiss.IndexFlatL2(n_features)  # L2 (squared Euclidean) index
    index.add(X)
    distances_sq, neighbors = index.search(X, k + 1)  # Include self as the 0th neighbor

    # Remove self distances and neighbors
    distances_sq = distances_sq[:, 1:].astype(np.float32)  # Shape: (n_samples, k)
    neighbors = neighbors[:, 1:]

    # Convert squared distances to Euclidean distances
    distances = np.sqrt(distances_sq).astype(np.float32)

    # Step 2: Initialize rho and sigma arrays
    rho = distances[:, 0].copy()  # Distance to the nearest neighbor, Shape: (n_samples,)
    target = np.log2(k).astype(np.float32)  # Target sum of probabilities

    # Step 3: Vectorized Binary Search to find sigma_i
    # Initialize low and high bounds for all samples
    low = np.full(n_samples, 1e-5, dtype=np.float32)
    high = np.full(n_samples, 10.0, dtype=np.float32)
    sigma = np.zeros(n_samples, dtype=np.float32)

    # Initialize mask to track convergence
    converged = np.zeros(n_samples, dtype=bool)

    for _ in range(max_iter):
        # Compute mid values where not yet converged
        mid = (low + high) / 2.0

        # Compute probabilities: exp(-max(d_ij - rho_i, 0) / sigma_i)
        exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / mid[:, np.newaxis]
        probs = np.exp(exponent)

        # Sum probabilities for each sample
        prob_sum = probs.sum(axis=1)

        # Check convergence
        diff = prob_sum - target
        abs_diff = np.abs(diff)
        newly_converged = (abs_diff < tol) & (~converged)
        converged |= newly_converged

        # Assign sigma where converged
        sigma[newly_converged] = mid[newly_converged]

        # Update high and low based on comparison with the target
        high = np.where(prob_sum > target, mid, high)
        low = np.where(prob_sum <= target, mid, low)

        # If all samples have converged, break early
        if converged.all():
            break

    # For any samples not converged within max_iter, assign the last mid
    sigma[~converged] = mid[~converged]

    return sigma, rho, distances, neighbors

def compute_p_umap(sigma, rho, distances, neighbors):
    """
    Compute the conditional probabilities p(UMAP_{j|i}) for each neighbor pair.

    Parameters:
    - sigma: np.ndarray of shape (n_samples,), computed sigma_i for each sample.
    - rho: np.ndarray of shape (n_samples,), distance to the nearest neighbor for each sample.
    - distances: np.ndarray of shape (n_samples, k), Euclidean distances to the k nearest neighbors for each sample.
    - neighbors: np.ndarray of shape (n_samples, k), indices of the k nearest neighbors for each sample.

    Returns:
    - P: scipy.sparse.csr_matrix of shape (n_samples, n_samples), conditional probabilities p(UMAP_{j|i}).
    """
    n_samples, k = distances.shape

    # Ensure sigma has no zero values to avoid division by zero
    sigma = np.maximum(sigma, 1e-10).astype(np.float32)

    # Compute the exponent term: -max(d_ij - rho_i, 0) / sigma_i
    exponent = -np.maximum(distances - rho[:, np.newaxis], 0) / sigma[:, np.newaxis]

    # Compute p_j|i using the exponent
    p_j_i = np.exp(exponent).astype(np.float32)

    # Create a COO sparse matrix
    row_indices = np.repeat(np.arange(n_samples), k)
    col_indices = neighbors.flatten()
    data = p_j_i.flatten()

    P = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(n_samples, n_samples))
    P = P.tocsr()  # Convert to CSR format for efficient arithmetic operations

    return P

def compute_p_umap_symmetric(P):
    """
    Compute the symmetric UMAP probabilities p^{UMAP}_{ij} = p(UMAP_{j|i}) + p(UMAP_{i|j}) - p(UMAP_{j|i}) * p(UMAP_{i|j}).

    Parameters:
    - P: scipy.sparse.csr_matrix of shape (n_samples, n_samples), conditional probabilities p(UMAP_{j|i}).

    Returns:
    - P_sym: scipy.sparse.csr_matrix of shape (n_samples, n_samples), symmetric probabilities p^{UMAP}_{ij}.
    """
    # Compute P + P.T
    P_transpose = P.transpose()
    P_plus_PT = P + P_transpose

    # Compute element-wise multiplication P.multiply(P_transpose)
    P_mul_PT = P.multiply(P_transpose)

    # Compute symmetric probabilities
    P_sym = P_plus_PT - P_mul_PT

    # Optionally, eliminate zeros to maintain sparsity
    P_sym.eliminate_zeros()

    return P_sym

def compute_all_p_umap(X, k, tol=1e-5, max_iter=100,return_dist_and_neigh=False):
    """
    Wrapper function to compute symmetric UMAP probabilities p^{UMAP}_{ij}.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features), dataset.
    - k: int, number of nearest neighbors.
    - tol: float, tolerance for binary search.
    - max_iter: int, maximum iterations for binary search.

    Returns:
    - P_sym: scipy.sparse.csr_matrix of shape (n_samples, n_samples), symmetric probabilities p^{UMAP}_{ij}.
    """
    # Step 1: Compute sigma, rho, distances, and neighbors
    sigma, rho, distances, neighbors = compute_sigma_i(X, k, tol, max_iter)

    # Step 2: Compute p_j|i
    P = compute_p_umap(sigma, rho, distances, neighbors)

    # Step 3: Compute symmetric probabilities p^{UMAP}_{ij}
    P_sym = compute_p_umap_symmetric(P)

    if return_dist_and_neigh:
        return P_sym, distances, neighbors
    else:
        return P_sym