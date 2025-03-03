"""Parametric UMAP implementation for dimensionality reduction using neural networks."""

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from parametric_umap.datasets.covariates_datasets import TorchSparseDataset, VariableDataset
from parametric_umap.datasets.edge_dataset import EdgeDataset
from parametric_umap.models.mlp import MLP
from parametric_umap.utils.graph import compute_all_p_umap
from parametric_umap.utils.losses import compute_correlation_loss


class ParametricUMAP:
    """A parametric implementation of UMAP (Uniform Manifold Approximation and Projection).

    This class implements a parametric version of UMAP that learns a neural network to perform
    dimensionality reduction. The model can transform new data points without having to recompute
    the entire embedding.

    Attributes:
        n_components (int): Number of dimensions in the output embedding
        hidden_dim (int): Dimension of hidden layers in the MLP
        n_layers (int): Number of hidden layers in the MLP
        n_neighbors (int): Number of neighbors to consider for each point
        a (float): UMAP parameter controlling local connectivity
        b (float): UMAP parameter controlling the strength of repulsion between points
        correlation_weight (float): Weight of the correlation loss term
        learning_rate (float): Learning rate for the optimizer
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device (str): Device to use for computations ('cpu' or 'cuda')
        use_batchnorm (bool): Whether to use batch normalization in the MLP
        use_dropout (bool): Whether to use dropout in the MLP
        model (Optional[MLP]): The neural network model
        is_fitted (bool): Whether the model has been fitted

    """

    def __init__(
        self,
        n_components: int = 2,
        hidden_dim: int = 1024,
        n_layers: int = 3,
        n_neighbors: int = 15,
        a: float = 0.1,
        b: float = 1.0,
        correlation_weight: float = 0.1,
        learning_rate: float = 1e-4,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_batchnorm: bool = False,
        use_dropout: bool = False,
    ) -> None:
        """Initialize ParametricUMAP.

        Parameters
        ----------
        n_components : int
            Number of dimensions in the output embedding
        hidden_dim : int
            Dimension of hidden layers in the MLP
        n_layers : int
            Number of hidden layers in the MLP
        n_neighbors : int
            Number of neighbors to consider for each point
        a, b : float
            UMAP parameters for the optimization
        correlation_weight : float
            Weight of the correlation loss term
        learning_rate : float
            Learning rate for the optimizer
        n_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        device : str
            Device to use for computations ('cpu' or 'cuda')
        use_batchnorm : bool
            Whether to use batch normalization in the MLP
        use_dropout : bool
            Whether to use dropout in the MLP

        """
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.a = a
        self.b = b
        self.correlation_weight = correlation_weight
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.model = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False

    def _init_model(self, input_dim: int) -> None:
        """Initialize the MLP model.

        Parameters
        ----------
        input_dim : int
            The input dimension of the data

        """
        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_components,
            num_layers=self.n_layers,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout,
        ).to(self.device)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        resample_negatives: bool = False,
        n_processes: int = 6,
        low_memory: bool = False,
        random_state: int = 0,
        verbose: bool = True,
    ) -> "ParametricUMAP":
        """Fit the model using X as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or torch tensor.
        resample_negatives : bool, optional (default=False)
            Whether to resample negative edges at each epoch.
        n_processes : int, optional (default=6)
            Number of processes to use for parallel computation.
        low_memory : bool, optional (default=False)
            If True, keeps the dataset on CPU to save GPU memory.
        random_state : int, optional (default=0)
            Random state for reproducibility.
        verbose : bool, optional (default=True)
            Whether to display progress bars and print statements.

        Returns
        -------
        self : ParametricUMAP
            The fitted model.

        """
        X = np.asarray(X).astype(np.float32)

        # Initialize model if not already done
        if self.model is None:
            self._init_model(X.shape[1])

        # Create datasets
        dataset = VariableDataset(X).to(self.device)
        P_sym = compute_all_p_umap(X, k=self.n_neighbors)
        ed = EdgeDataset(P_sym)

        target_dataset = TorchSparseDataset(P_sym) if low_memory else TorchSparseDataset(P_sym).to(self.device)

        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        losses = []

        loader = ed.get_loader(
            batch_size=self.batch_size,
            sample_first=True,
            random_state=random_state,
            n_processes=n_processes,
            verbose=verbose,
        )

        if verbose:
            print("Training...")

        pbar = tqdm(range(self.n_epochs), desc="Epochs", position=0)
        for epoch in pbar:
            epoch_loss = 0
            num_batches = 0

            for edge_batch in tqdm(loader, desc=f"Epoch {epoch + 1}", position=1, leave=False):
                optimizer.zero_grad()

                # Get src and dst indexes from edge_batch
                src_indexes = [i for i, j in edge_batch]
                dst_indexes = [j for i, j in edge_batch]

                # Get values from dataset
                src_values = dataset[src_indexes]
                dst_values = dataset[dst_indexes]
                targets = target_dataset[edge_batch]

                # If low memory, the dataset is not on GPU, so we need to move the values to GPU
                if low_memory:
                    src_values = src_values.to(self.device)
                    dst_values = dst_values.to(self.device)
                    targets = targets.to(self.device)

                # Get embeddings from model
                src_embeddings = self.model(src_values)
                dst_embeddings = self.model(dst_values)

                # Compute distances
                Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1)
                X_distances = torch.norm(src_values - dst_values, dim=1)

                # Compute losses
                qs = torch.pow(1 + self.a * torch.norm(src_embeddings - dst_embeddings, dim=1, p=2 * self.b), -1)
                umap_loss = self.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(X_distances, Z_distances)
                loss = umap_loss + self.correlation_weight * corr_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if resample_negatives:
                loader = ed.get_loader(batch_size=self.batch_size, sample_first=True)

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if verbose:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform. Can be numpy array or torch tensor.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the low-dimensional space.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            X_reduced = self.model(X)

        return X_reduced.cpu().numpy()

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        verbose: bool = True,
        low_memory: bool = False,
    ) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or torch tensor.
        verbose : bool, optional (default=True)
            Whether to display progress bars and print statements.
        low_memory : bool, optional (default=False)
            If True, keeps the dataset on CPU to save GPU memory.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the low-dimensional space.

        """
        self.fit(X, verbose=verbose, low_memory=low_memory)
        return self.transform(X)

    def save(self, path: str) -> None:
        """Save the model to a file.

        Parameters
        ----------
        path : str
            Path to save the model.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "a": self.a,
            "b": self.b,
            "correlation_weight": self.correlation_weight,
            "use_batchnorm": self.use_batchnorm,
            "use_dropout": self.use_dropout,
        }

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> "ParametricUMAP":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, optional (default='cuda' if available else 'cpu')
            Device to load the model to.

        Returns
        -------
        model : ParametricUMAP
            The loaded model instance.

        """
        save_dict = torch.load(path, map_location=device)

        # Create instance with saved parameters
        instance = cls(
            n_components=save_dict["n_components"],
            hidden_dim=save_dict["hidden_dim"],
            n_layers=save_dict["n_layers"],
            a=save_dict["a"],
            b=save_dict["b"],
            correlation_weight=save_dict["correlation_weight"],
            device=device,
            use_batchnorm=save_dict["use_batchnorm"],
            use_dropout=save_dict["use_dropout"],
        )

        # Initialize model architecture
        instance._init_model(input_dim=save_dict["model_state_dict"]["model.0.weight"].shape[1])

        # Load state dict
        instance.model.load_state_dict(save_dict["model_state_dict"])
        instance.is_fitted = True

        return instance
