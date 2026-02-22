"""Shared pytest fixtures and configuration for parametric_umap tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from scipy import sparse


@pytest.fixture
def sample_2d_data():
    """Generate simple 2D dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 2).astype(np.float32)
    return X


@pytest.fixture
def sample_3d_data():
    """Generate simple 3D dataset for testing."""
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 3).astype(np.float32)
    return X


@pytest.fixture
def large_sample_data():
    """Generate larger dataset for integration testing."""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10).astype(np.float32)
    return X


@pytest.fixture
def swiss_roll_data():
    """Generate Swiss roll dataset similar to examples."""
    from sklearn.datasets import make_swiss_roll

    X, _ = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
    return X.astype(np.float32)


@pytest.fixture
def sparse_matrix_data():
    """Generate sparse matrix for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    density = 0.1

    # Create random sparse matrix
    nnz = int(n_samples * n_features * density)
    rows = np.random.randint(0, n_samples, nnz)
    cols = np.random.randint(0, n_features, nnz)
    data = np.random.randn(nnz).astype(np.float32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
    return matrix


@pytest.fixture
def torch_tensor_data(sample_2d_data):
    """Convert sample data to torch tensor."""
    return torch.tensor(sample_2d_data, dtype=torch.float32)


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for testing device handling."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA unavailability for testing CPU fallback."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def temp_model_file():
    """Create temporary file for model save/load testing."""
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def default_umap_params():
    """Return default ParametricUMAP parameters for testing."""
    return {
        "n_components": 2,
        "hidden_dim": 1024,  # Actual default
        "n_layers": 3,
        "n_neighbors": 15,
        "a": 0.1,  # Actual default
        "b": 1.0,  # Actual default
        "correlation_weight": 0.1,  # Actual default
        "learning_rate": 1e-4,  # Actual default
        "n_epochs": 10,
        "batch_size": 32,  # Actual default
        "device": "cpu",
        "use_batchnorm": False,  # Actual default
        "use_dropout": False,
    }


@pytest.fixture
def minimal_umap_params():
    """Minimal ParametricUMAP parameters for quick testing."""
    return {
        "n_components": 2,
        "n_epochs": 2,
        "batch_size": 32,
        "device": "cpu",
    }


@pytest.fixture
def mock_faiss_computation():
    """Mock FAISS computations for isolated testing."""

    def mock_compute_all_p_umap(X, n_neighbors=15, **kwargs):
        """Mock graph computation function."""
        n_samples = X.shape[0]
        # Create mock sparse adjacency matrix
        row_ind = np.random.randint(0, n_samples, n_samples * n_neighbors)
        col_ind = np.random.randint(0, n_samples, n_samples * n_neighbors)
        data = np.random.uniform(0, 1, n_samples * n_neighbors)

        return sparse.csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(n_samples, n_samples),
        )

    with patch("parametric_umap.core.compute_all_p_umap", side_effect=mock_compute_all_p_umap):
        yield


@pytest.fixture
def mock_mlp_model():
    """Mock MLP model for testing without actual neural network training."""
    mock_model = Mock()
    mock_model.train = Mock()
    mock_model.eval = Mock()
    mock_model.to = Mock(return_value=mock_model)
    mock_model.parameters = Mock(return_value=[])

    # Mock forward pass
    def mock_forward(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, 2)  # Return random 2D embeddings

    mock_model.forward = mock_forward
    mock_model.__call__ = mock_forward

    return mock_model


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def edge_list_data():
    """Generate edge list data for EdgeDataset testing."""
    n_edges = 200
    n_nodes = 50

    # Create random edge list
    sources = np.random.randint(0, n_nodes, n_edges)
    targets = np.random.randint(0, n_nodes, n_edges)
    weights = np.random.uniform(0, 1, n_edges)

    return np.column_stack([sources, targets, weights])


@pytest.fixture
def mock_progress_bar():
    """Mock tqdm progress bar for cleaner test output."""
    with patch("parametric_umap.core.tqdm") as mock_tqdm:
        mock_tqdm.side_effect = lambda x, *args, **kwargs: x
        yield mock_tqdm
