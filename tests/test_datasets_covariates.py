"""Comprehensive tests for parametric_umap.datasets.covariates_datasets module."""

import numpy as np
import pytest
import torch
from scipy import sparse

from parametric_umap.datasets.covariates_datasets import TorchSparseDataset, VariableDataset


class TestTorchSparseDataset:
    """Test TorchSparseDataset class."""

    def test_basic_initialization(self, sparse_matrix_data):
        """Test basic initialization with sparse matrix."""
        dataset = TorchSparseDataset(sparse_matrix_data)

        assert hasattr(dataset, "_keys")
        assert hasattr(dataset, "_values")
        assert hasattr(dataset, "device")
        assert dataset.device == "cpu"
        assert isinstance(dataset._keys, torch.Tensor)
        assert isinstance(dataset._values, torch.Tensor)

    def test_initialization_with_device(self, sparse_matrix_data):
        """Test initialization with specific device."""
        dataset = TorchSparseDataset(sparse_matrix_data, device="cpu")

        assert dataset.device == "cpu"
        assert dataset._keys.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            dataset_gpu = TorchSparseDataset(sparse_matrix_data, device="cuda:0")
            assert dataset_gpu.device == "cuda:0"
            assert dataset_gpu._keys.device.type == "cuda"

    def test_sparse_tensor_properties(self, sparse_matrix_data):
        """Test that dataset maintains properties of original matrix."""
        dataset = TorchSparseDataset(sparse_matrix_data)

        # Check dimension
        assert dataset._n == sparse_matrix_data.shape[0]

        # Check that number of non-zeros is preserved
        original_nnz = sparse_matrix_data.nnz
        assert len(dataset) == original_nnz

    def test_getitem_single_index(self, sparse_matrix_data):
        """Test __getitem__ with single index tuple."""
        dataset = TorchSparseDataset(sparse_matrix_data)

        # Test accessing specific indices
        i, j = 0, 1
        value = dataset[(i, j)]

        assert isinstance(value, torch.Tensor)
        assert value.numel() == 1  # Should be scalar tensor

        # Check value matches original matrix
        expected_value = sparse_matrix_data[i, j]
        assert torch.isclose(value, torch.tensor(expected_value), atol=1e-6)

    def test_getitem_multiple_indices(self, sparse_matrix_data):
        """Test __getitem__ with multiple index tuples."""
        dataset = TorchSparseDataset(sparse_matrix_data)

        # Test accessing multiple indices
        indices = [(0, 1), (1, 2), (2, 3)]
        values = dataset[indices]

        assert isinstance(values, torch.Tensor)
        assert values.shape[0] == len(indices)

        # Check values match original matrix
        for idx, (i, j) in enumerate(indices):
            expected_value = sparse_matrix_data[i, j]
            assert torch.isclose(values[idx], torch.tensor(expected_value), atol=1e-6)

    def test_getitem_zero_values(self):
        """Test __getitem__ returns zero for empty indices."""
        # Create simple sparse matrix with known pattern
        n = 5
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(n, n))
        dataset = TorchSparseDataset(sparse_mat)

        # Test accessing zero element
        value = dataset[(0, 1)]  # Should be zero
        assert torch.isclose(value, torch.tensor(0.0), atol=1e-6)

        # Test accessing non-zero element
        value = dataset[(0, 0)]  # Should be 1.0
        assert torch.isclose(value, torch.tensor(1.0), atol=1e-6)

    def test_len(self, sparse_matrix_data):
        """Test __len__ returns number of non-zero elements."""
        dataset = TorchSparseDataset(sparse_matrix_data)

        length = len(dataset)
        assert length == sparse_matrix_data.nnz
        assert isinstance(length, int)
        assert length >= 0

    def test_to_device(self, sparse_matrix_data):
        """Test moving dataset to different device."""
        dataset = TorchSparseDataset(sparse_matrix_data, device="cpu")

        # Move to CPU (should work)
        result = dataset.to("cpu")
        assert result is dataset  # Should return self
        assert dataset.device == "cpu"
        assert dataset._keys.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            dataset.to("cuda:0")
            assert dataset.device == "cuda:0"
            assert dataset._keys.device.type == "cuda"

            # Move back to CPU
            dataset.to("cpu")
            assert dataset.device == "cpu"
            assert dataset._keys.device == torch.device("cpu")

    def test_empty_sparse_matrix(self):
        """Test with empty sparse matrix."""
        empty_matrix = sparse.csr_matrix((5, 5))
        dataset = TorchSparseDataset(empty_matrix)

        assert len(dataset) == 0
        assert dataset._n == 5

        # Accessing any element should return zero
        value = dataset[(0, 0)]
        assert torch.isclose(value, torch.tensor(0.0), atol=1e-6)

    def test_large_sparse_matrix(self):
        """Test with large sparse matrix."""
        # Create large sparse matrix
        n = 1000
        nnz = 5000

        np.random.seed(42)
        rows = np.random.randint(0, n, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.uniform(0, 1, nnz)

        large_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        dataset = TorchSparseDataset(large_matrix)

        assert dataset._n == n
        assert len(dataset) <= nnz  # May be less due to duplicates

    def test_data_type_preservation(self):
        """Test that data types are properly handled."""
        # Create sparse matrix with specific data type
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        data = np.array([1.5, 2.5, 3.5], dtype=np.float64)

        sparse_mat = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        dataset = TorchSparseDataset(sparse_mat)

        # Check that values use float32
        assert dataset._values.dtype == torch.float32

        # Check values are correctly converted
        value = dataset[(0, 0)]
        assert torch.isclose(value, torch.tensor(1.5), atol=1e-6)

    def test_index_bounds(self, sparse_matrix_data):
        """Test handling of index bounds."""
        dataset = TorchSparseDataset(sparse_matrix_data)
        n_rows, n_cols = sparse_matrix_data.shape

        # Test valid bounds
        value = dataset[(0, 0)]
        assert isinstance(value, torch.Tensor)

        value = dataset[(n_rows - 1, n_cols - 1)]
        assert isinstance(value, torch.Tensor)

        # Out-of-bounds indices return 0 (key won't match in searchsorted)
        value = dataset[(n_rows, 0)]
        assert torch.isclose(value, torch.tensor(0.0), atol=1e-6)

        value = dataset[(0, n_cols)]
        assert torch.isclose(value, torch.tensor(0.0), atol=1e-6)


class TestVariableDataset:
    """Test VariableDataset class."""

    def test_basic_initialization(self, sample_2d_data):
        """Test basic initialization with numpy array."""
        dataset = VariableDataset(sample_2d_data)

        assert hasattr(dataset, "X")
        assert hasattr(dataset, "indexes_map")
        assert isinstance(dataset.X, torch.Tensor)
        assert dataset.X.dtype == torch.float32
        assert dataset.indexes_map is None  # No indexes provided

    def test_initialization_with_indexes(self, sample_2d_data):
        """Test initialization with index mapping."""
        indexes = [10, 20, 30, 40, 50]
        n_samples = min(len(sample_2d_data), len(indexes))

        dataset = VariableDataset(sample_2d_data[:n_samples], indexes)

        assert dataset.indexes_map is not None
        assert len(dataset.indexes_map) == n_samples

        # Check mapping is correct
        for i, idx in enumerate(indexes):
            assert dataset.indexes_map[idx] == i

    def test_len(self, sample_2d_data):
        """Test __len__ returns number of samples."""
        dataset = VariableDataset(sample_2d_data)

        length = len(dataset)
        assert length == len(sample_2d_data)
        assert isinstance(length, int)

    def test_getitem_integer_index(self, sample_2d_data):
        """Test __getitem__ with integer index."""
        dataset = VariableDataset(sample_2d_data)

        # Test single index access
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (sample_2d_data.shape[1],)

        # Check value matches original data
        expected = torch.tensor(sample_2d_data[0], dtype=torch.float32)
        assert torch.allclose(item, expected, atol=1e-6)

    def test_getitem_list_index(self, sample_2d_data):
        """Test __getitem__ with list of indices."""
        dataset = VariableDataset(sample_2d_data)

        # Test multiple index access
        indices = [0, 2, 4]
        items = dataset[indices]

        assert isinstance(items, torch.Tensor)
        assert items.shape == (len(indices), sample_2d_data.shape[1])

        # Check values match original data
        expected = torch.tensor(sample_2d_data[indices], dtype=torch.float32)
        assert torch.allclose(items, expected, atol=1e-6)

    def test_getitem_slice(self, sample_2d_data):
        """Test __getitem__ with slice."""
        dataset = VariableDataset(sample_2d_data)

        # Test slice access
        items = dataset[1:5]

        assert isinstance(items, torch.Tensor)
        assert items.shape == (4, sample_2d_data.shape[1])

        # Check values match original data
        expected = torch.tensor(sample_2d_data[1:5], dtype=torch.float32)
        assert torch.allclose(items, expected, atol=1e-6)

    def test_to_device(self, sample_2d_data):
        """Test moving dataset to different device."""
        dataset = VariableDataset(sample_2d_data)

        # Move to CPU (should work)
        result = dataset.to("cpu")
        assert result is dataset  # Should return self
        assert dataset.X.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            dataset.to("cuda:0")
            assert dataset.X.device.type == "cuda"

            # Move back to CPU
            dataset.to("cpu")
            assert dataset.X.device == torch.device("cpu")

    def test_get_index_with_mapping(self, sample_2d_data):
        """Test get_index method with index mapping."""
        indexes = [100, 200, 300]
        n_samples = min(len(sample_2d_data), len(indexes))

        dataset = VariableDataset(sample_2d_data[:n_samples], indexes)

        # Test getting position for each index
        for i, idx in enumerate(indexes):
            position = dataset.get_index(idx)
            assert position == i

    def test_get_index_without_mapping(self, sample_2d_data):
        """Test get_index method without index mapping raises error."""
        dataset = VariableDataset(sample_2d_data)

        with pytest.raises(ValueError, match="Indexes map not initialized"):
            dataset.get_index(0)

    def test_get_values_by_indexes(self, sample_2d_data):
        """Test get_values_by_indexes method."""
        indexes = [10, 20, 30, 40]
        n_samples = min(len(sample_2d_data), len(indexes))

        dataset = VariableDataset(sample_2d_data[:n_samples], indexes)

        # Test getting values by original indexes
        values = dataset.get_values_by_indexes([10, 30])

        assert isinstance(values, torch.Tensor)
        assert values.shape == (2, sample_2d_data.shape[1])

        # Check values are correct
        expected_0 = torch.tensor(sample_2d_data[0], dtype=torch.float32)  # Index 10 -> position 0
        expected_2 = torch.tensor(sample_2d_data[2], dtype=torch.float32)  # Index 30 -> position 2

        assert torch.allclose(values[0], expected_0, atol=1e-6)
        assert torch.allclose(values[1], expected_2, atol=1e-6)

    def test_get_values_by_indexes_invalid(self, sample_2d_data):
        """Test get_values_by_indexes with invalid indexes."""
        indexes = [10, 20, 30]
        dataset = VariableDataset(sample_2d_data[:3], indexes)

        # Test with index not in mapping
        with pytest.raises(KeyError):
            dataset.get_values_by_indexes([40])  # Index 40 not in mapping

    def test_empty_dataset(self):
        """Test with empty dataset."""
        empty_data = np.array([]).reshape(0, 2)
        dataset = VariableDataset(empty_data)

        assert len(dataset) == 0
        assert dataset.X.shape == (0, 2)

    def test_single_sample(self):
        """Test with single sample."""
        single_sample = np.array([[1.0, 2.0]])
        dataset = VariableDataset(single_sample)

        assert len(dataset) == 1
        item = dataset[0]
        assert item.shape == (2,)

        expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
        assert torch.allclose(item, expected, atol=1e-6)

    def test_different_data_types(self):
        """Test with different input data types."""
        # Test with different numpy dtypes
        data_int = np.array([[1, 2], [3, 4]], dtype=np.int32)
        dataset_int = VariableDataset(data_int)
        assert dataset_int.X.dtype == torch.float32

        data_float64 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        dataset_float64 = VariableDataset(data_float64)
        assert dataset_float64.X.dtype == torch.float32

    def test_index_bounds(self, sample_2d_data):
        """Test handling of index bounds."""
        dataset = VariableDataset(sample_2d_data)
        n_samples = len(sample_2d_data)

        # Test valid bounds
        item = dataset[0]
        assert isinstance(item, torch.Tensor)

        item = dataset[n_samples - 1]
        assert isinstance(item, torch.Tensor)

        # Test invalid bounds
        with pytest.raises(IndexError):
            dataset[n_samples]

        with pytest.raises(IndexError):
            dataset[-n_samples - 1]

    def test_multidimensional_data(self):
        """Test with different dimensional data."""
        # 1D data
        data_1d = np.random.randn(50, 1).astype(np.float32)
        dataset_1d = VariableDataset(data_1d)
        assert dataset_1d.X.shape == (50, 1)

        # 3D data
        data_3d = np.random.randn(30, 3).astype(np.float32)
        dataset_3d = VariableDataset(data_3d)
        assert dataset_3d.X.shape == (30, 3)

        # High-dimensional data
        data_hd = np.random.randn(20, 100).astype(np.float32)
        dataset_hd = VariableDataset(data_hd)
        assert dataset_hd.X.shape == (20, 100)


class TestDatasetsIntegration:
    """Integration tests for dataset classes."""

    def test_torch_sparse_with_real_umap_data(self):
        """Test TorchSparseDataset with realistic UMAP probability matrix."""
        # Create realistic sparse probability matrix
        n = 100
        density = 0.1

        np.random.seed(42)
        # Create symmetric probability matrix
        prob_matrix = sparse.random(n, n, density=density, format="csr")
        prob_matrix = (prob_matrix + prob_matrix.T) / 2  # Make symmetric
        prob_matrix.data = np.abs(prob_matrix.data)  # Ensure non-negative

        dataset = TorchSparseDataset(prob_matrix)

        # Test that symmetry is preserved
        i, j = 10, 20
        value_ij = dataset[(i, j)]
        value_ji = dataset[(j, i)]
        assert torch.isclose(value_ij, value_ji, atol=1e-6)

        # Test that probabilities are in valid range
        assert len(dataset) > 0
        assert len(dataset) == prob_matrix.nnz

    def test_variable_dataset_with_embedding_data(self, swiss_roll_data):
        """Test VariableDataset with realistic embedding data."""
        # Use Swiss roll data as realistic test case
        dataset = VariableDataset(swiss_roll_data)

        assert len(dataset) == len(swiss_roll_data)
        assert dataset.X.shape == swiss_roll_data.shape

        # Test accessing subsets
        subset = dataset[:50]
        assert subset.shape == (50, swiss_roll_data.shape[1])

        # Test with index mapping
        indexes = list(range(0, len(swiss_roll_data), 10))  # Every 10th sample
        subset_data = swiss_roll_data[indexes]

        dataset_with_mapping = VariableDataset(subset_data, indexes)

        # Test that mapping works correctly
        for i, original_idx in enumerate(indexes[:5]):  # Test first 5
            position = dataset_with_mapping.get_index(original_idx)
            assert position == i

    def test_datasets_device_consistency(self, sparse_matrix_data, sample_2d_data):
        """Test that both datasets handle device operations consistently."""
        sparse_dataset = TorchSparseDataset(sparse_matrix_data)
        variable_dataset = VariableDataset(sample_2d_data)

        # Both should start on CPU
        assert sparse_dataset.device == "cpu"
        assert variable_dataset.X.device == torch.device("cpu")

        # Test moving both to same device
        sparse_dataset.to("cpu")
        variable_dataset.to("cpu")

        assert sparse_dataset.device == "cpu"
        assert variable_dataset.X.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            sparse_dataset.to("cuda:0")
            variable_dataset.to("cuda:0")

            assert sparse_dataset.device == "cuda:0"
            assert variable_dataset.X.device.type == "cuda"

    def test_datasets_memory_efficiency(self):
        """Test that datasets handle large data efficiently."""
        # Create moderately large sparse matrix
        n = 500
        density = 0.05

        large_sparse = sparse.random(n, n, density=density, format="csr")
        sparse_dataset = TorchSparseDataset(large_sparse)

        # Should preserve nnz count
        assert len(sparse_dataset) == large_sparse.nnz

        # Create large dense data
        large_dense = np.random.randn(1000, 50).astype(np.float32)
        variable_dataset = VariableDataset(large_dense)

        # Should handle large dense data
        assert len(variable_dataset) == 1000
        assert variable_dataset.X.shape == (1000, 50)


class TestDatasetsErrorHandling:
    """Test error handling in dataset classes."""

    def test_torch_sparse_invalid_input(self):
        """Test TorchSparseDataset with invalid input."""
        # Test with non-sparse matrix
        with pytest.raises((AttributeError, TypeError)):
            TorchSparseDataset("not_a_matrix")

        # Test with invalid device
        valid_matrix = sparse.csr_matrix(([1, 2], ([0, 1], [0, 1])), shape=(2, 2))
        with pytest.raises((RuntimeError, ValueError)):
            TorchSparseDataset(valid_matrix, device="invalid_device")

    def test_variable_dataset_invalid_input(self):
        """Test VariableDataset with invalid input."""
        # Test with non-array input
        with pytest.raises((TypeError, ValueError)):
            VariableDataset("not_an_array")

        # Test with wrong index list length
        data = np.array([[1, 2], [3, 4]])
        wrong_indexes = [10, 20, 30]  # Length mismatch

        # This should create the dataset but may fail when accessing
        dataset = VariableDataset(data, wrong_indexes)
        assert len(dataset) == 2  # Based on data length

    def test_index_access_errors(self, sparse_matrix_data, sample_2d_data):
        """Test index access error handling."""
        sparse_dataset = TorchSparseDataset(sparse_matrix_data)
        variable_dataset = VariableDataset(sample_2d_data)

        # Test invalid index types for sparse dataset
        with pytest.raises((TypeError, IndexError, ValueError)):
            sparse_dataset["invalid_index"]

        # Test invalid index types for variable dataset
        with pytest.raises((TypeError, IndexError, ValueError)):
            variable_dataset["invalid_index"]

        # Out-of-bounds access returns 0 for sparse dataset (searchsorted-based)
        n_rows, n_cols = sparse_matrix_data.shape
        value = sparse_dataset[(n_rows + 10, 0)]
        assert torch.isclose(value, torch.tensor(0.0), atol=1e-6)

        with pytest.raises(IndexError):
            variable_dataset[len(sample_2d_data) + 10]
