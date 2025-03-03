"""Utility functions for the Parametric UMAP implementation."""
import torch


def compute_correlation_loss(X_distances: torch.Tensor, Z_distances: torch.Tensor) -> torch.Tensor:
    """Compute the negative Pearson correlation between distances in input and embedding spaces.

    This function calculates the Pearson correlation coefficient between distances in the
    original space (X_distances) and distances in the embedding space (Z_distances),
    and returns its negative value since we want to maximize the correlation during
    optimization.

    Parameters
    ----------
    X_distances : torch.Tensor
        Distances in input space, shape (batch_size,)
    Z_distances : torch.Tensor
        Distances in embedding space, shape (batch_size,)

    Returns
    -------
    torch.Tensor
        Negative Pearson correlation coefficient between X_distances and Z_distances

    Notes
    -----
    The correlation is computed using the formula:
        corr = (E[XZ] - E[X]E[Z]) / (std(X) * std(Z))
    where E[] denotes expectation and std() denotes standard deviation.

    """
    # Compute means
    X_mean = X_distances.mean()
    Z_mean = Z_distances.mean()

    # Center the variables
    X_centered = X_distances - X_mean
    Z_centered = Z_distances - Z_mean

    # Compute correlation
    numerator = (X_centered * Z_centered).mean()
    X_std = torch.sqrt((X_centered**2).mean())
    Z_std = torch.sqrt((Z_centered**2).mean())

    correlation = numerator / (X_std * Z_std)

    # Return negative correlation as we want to maximize correlation
    return -correlation
