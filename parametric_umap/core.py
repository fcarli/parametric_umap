"""Parametric UMAP implementation for dimensionality reduction using neural networks."""

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from parametric_umap.datasets.covariates_datasets import VariableDataset
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
        compile_model (bool): Whether to apply ``torch.compile`` to the MLP
        model (Optional[MLP]): The neural network model (possibly compiled)
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
        device: str | None = None,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        compile_model: bool = False,
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
        device : str, optional
            Device to use for computations ('cpu', 'cuda', or 'mps').
            Auto-detected if not specified (CUDA > MPS > CPU).
        use_batchnorm : bool
            Whether to use batch normalization in the MLP
        use_dropout : bool
            Whether to use dropout in the MLP
        compile_model : bool
            Whether to apply ``torch.compile`` to the MLP. Can yield
            10-30 % faster training on PyTorch 2.x at the cost of a
            one-time compilation delay on the first forward pass.

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
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.compile_model = compile_model

        self.model = None
        self.loss_fn = nn.BCELoss()
        self.is_fitted = False

    @property
    def _unwrapped_model(self) -> MLP | None:
        """Return the underlying MLP, unwrapping ``torch.compile`` if needed."""
        if self.model is None:
            return None
        return getattr(self.model, "_orig_mod", self.model)

    def _init_model(self, input_dim: int) -> None:
        """Initialize the MLP model.

        Parameters
        ----------
        input_dim : int
            The input dimension of the data

        """
        model = MLP(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_components,
            num_layers=self.n_layers,
            use_batchnorm=self.use_batchnorm,
            use_dropout=self.use_dropout,
        ).to(self.device)
        self.model = torch.compile(model) if self.compile_model else model

    @staticmethod
    def _precompute_edge_tensors(
        X: np.ndarray, edges: np.ndarray, weights: np.ndarray, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build device-placed tensors for edge weights and input-space distances."""
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        x_dists = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1).astype(np.float32)
        x_dists_t = torch.tensor(x_dists, dtype=torch.float32, device=device)
        return weights_t, x_dists_t

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
            If True, keeps the data and edge weights on CPU and transfers
            per batch.  Trades speed for lower accelerator memory usage.
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

        # Create datasets.  In low_memory mode, keep data on CPU and
        # transfer each batch to the compute device during training.
        dataset = VariableDataset(X) if low_memory else VariableDataset(X).to(self.device)
        P_sym = compute_all_p_umap(X, k=self.n_neighbors)
        ed = EdgeDataset(P_sym)

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

        # Pre-place edge weights and input-space distances on the compute
        # device (or CPU in low_memory mode) so batch slicing is fast.
        _tensor_device = "cpu" if low_memory else self.device
        all_weights_t, all_x_dists_t = self._precompute_edge_tensors(
            X, ed.all_edges, ed.all_weights, _tensor_device
        )

        if verbose:
            print("Training...")

        pbar = tqdm(range(self.n_epochs), desc="Epochs", position=0, disable=not verbose)
        for epoch in pbar:
            epoch_loss, num_batches = 0.0, 0

            for edge_batch, _weight_batch, batch_idx in tqdm(
                loader, desc=f"Epoch {epoch + 1}", position=1, leave=False, disable=not verbose
            ):
                optimizer.zero_grad(set_to_none=True)

                # Get src and dst indexes from edge_batch (numpy array slicing)
                src_indexes = edge_batch[:, 0]
                dst_indexes = edge_batch[:, 1]

                # Get values from dataset; weights and X distances are sliced
                # from pre-placed tensors
                src_values = dataset[src_indexes]
                dst_values = dataset[dst_indexes]
                targets = all_weights_t[batch_idx]
                X_distances = all_x_dists_t[batch_idx]

                # In low_memory mode, move batch tensors to the compute device
                if low_memory:
                    src_values = src_values.to(self.device)
                    dst_values = dst_values.to(self.device)
                    targets = targets.to(self.device)
                    X_distances = X_distances.to(self.device)

                # Get embeddings from model
                src_embeddings = self.model(src_values)
                dst_embeddings = self.model(dst_values)

                # Compute embedding-space distances (L^(2b) norm)
                Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1, p=2 * self.b)

                # Compute losses
                qs = torch.pow(1 + self.a * Z_distances, -1)
                qs = qs.clamp(1e-7, 1 - 1e-7)
                umap_loss = self.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(X_distances, Z_distances)
                loss = umap_loss + self.correlation_weight * corr_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if resample_negatives:
                loader = ed.get_loader(batch_size=self.batch_size, sample_first=True)
                all_weights_t, all_x_dists_t = self._precompute_edge_tensors(
                    X, ed.all_edges, ed.all_weights, _tensor_device
                )

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

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self._unwrapped_model.input_dim:
            msg = f"X has {X.shape[1]} features, but model was fitted with {self._unwrapped_model.input_dim} features"
            raise ValueError(msg)

        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            X_reduced = self.model(X)

        return X_reduced.cpu().numpy()

    def fit_transform(
        self,
        X: np.ndarray | torch.Tensor,
        resample_negatives: bool = False,
        n_processes: int = 6,
        low_memory: bool = False,
        random_state: int = 0,
        verbose: bool = True,
    ) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be numpy array or torch tensor.
        resample_negatives : bool, optional (default=False)
            Whether to resample negative edges at each epoch.
        n_processes : int, optional (default=6)
            Number of processes to use for parallel computation.
        low_memory : bool, optional (default=False)
            If True, keeps the data and edge weights on CPU and transfers
            per batch.  Trades speed for lower accelerator memory usage.
        random_state : int, optional (default=0)
            Random state for reproducibility.
        verbose : bool, optional (default=True)
            Whether to display progress bars and print statements.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data in the low-dimensional space.

        """
        self.fit(
            X,
            resample_negatives=resample_negatives,
            n_processes=n_processes,
            low_memory=low_memory,
            random_state=random_state,
            verbose=verbose,
        )
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
            "model_state_dict": self._unwrapped_model.state_dict(),
            "input_dim": self._unwrapped_model.input_dim,
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
    def load(cls, path: str, device: str | None = None) -> "ParametricUMAP":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, optional
            Device to load the model to. Auto-detected if not specified.

        Returns
        -------
        model : ParametricUMAP
            The loaded model instance.

        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        save_dict = torch.load(path, map_location=device, weights_only=True)

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

        # Initialize model architecture (fall back to weight introspection for old checkpoints)
        input_dim = save_dict.get("input_dim") or save_dict["model_state_dict"]["model.0.weight"].shape[1]
        instance._init_model(input_dim=input_dim)

        # Load state dict
        instance.model.load_state_dict(save_dict["model_state_dict"])
        instance.is_fitted = True

        return instance
