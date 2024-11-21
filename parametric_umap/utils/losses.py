import torch

def compute_correlation_loss(X_distances, Z_distances):
    """
    Compute Pearson correlation between distances in X and Z spaces
    
    Args:
        X_distances: Distances in input space (batch_size,)
        Z_distances: Distances in embedding space (batch_size,)
    """
    # Compute means
    X_mean = X_distances.mean()
    Z_mean = Z_distances.mean()
    
    # Center the variables
    X_centered = X_distances - X_mean
    Z_centered = Z_distances - Z_mean
    
    # Compute correlation
    numerator = (X_centered * Z_centered).mean()
    X_std = torch.sqrt((X_centered ** 2).mean())
    Z_std = torch.sqrt((Z_centered ** 2).mean())
    
    correlation = numerator / (X_std * Z_std)
    
    # Return negative correlation as we want to maximize correlation
    return -correlation