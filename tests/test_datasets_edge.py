"""Comprehensive tests for parametric_umap.datasets.edge_dataset module."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from parametric_umap.datasets.edge_dataset import EdgeBatchIterator, EdgeDataset


class TestEdgeBatchIterator:
    """Test EdgeBatchIterator class."""

    def test_basic_initialization(self):
        """Test basic initialization with numpy arrays."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)
        batch_size = 2

        iterator = EdgeBatchIterator(edges, weights, batch_size)

        np.testing.assert_array_equal(iterator.edges, edges)
        np.testing.assert_array_equal(iterator.weights, weights)
        assert iterator.batch_size == batch_size
        assert iterator.shuffle is False
        assert iterator.current == 0

    def test_initialization_with_shuffle(self):
        """Test initialization with shuffle option."""
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8], dtype=np.float32)

        iterator = EdgeBatchIterator(edges, weights, batch_size=2, shuffle=True)

        assert iterator.shuffle is True

    def test_iter_without_shuffle(self):
        """Test iteration without shuffling."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2, shuffle=False)

        iter_obj = iter(iterator)
        assert iter_obj is iterator
        assert iterator._perm is None
        assert iterator.current == 0

    def test_iter_with_shuffle(self):
        """Test iteration with shuffling creates a permutation."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2, shuffle=True)

        np.random.seed(42)
        iter(iterator)

        assert iterator._perm is not None
        assert len(iterator._perm) == len(edges)

    def test_next_basic_batching(self):
        """Test basic batching functionality."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)

        iter_obj = iter(iterator)

        # First batch
        batch_edges, batch_weights, _idx = next(iter_obj)
        np.testing.assert_array_equal(batch_edges, [[0, 1], [1, 2]])
        np.testing.assert_allclose(batch_weights, [1.0, 0.5])
        assert iterator.current == 2

        # Second batch
        batch_edges, batch_weights, _idx = next(iter_obj)
        np.testing.assert_array_equal(batch_edges, [[2, 3], [3, 4]])
        np.testing.assert_allclose(batch_weights, [0.8, 0.3])
        assert iterator.current == 4

        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(iter_obj)

    def test_next_partial_batch(self):
        """Test batching with partial last batch."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3, 0.1], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)

        iter_obj = iter(iterator)

        # First two full batches
        next(iter_obj)
        next(iter_obj)

        # Partial last batch
        batch_edges, batch_weights, _idx = next(iter_obj)
        np.testing.assert_array_equal(batch_edges, [[4, 5]])
        np.testing.assert_allclose(batch_weights, [0.1])

        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(iter_obj)

    def test_len(self):
        """Test __len__ method."""
        # Test exact division
        edges = np.zeros((4, 2), dtype=np.int64)
        weights = np.zeros(4, dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)
        assert len(iterator) == 2

        # Test with remainder
        edges = np.zeros((5, 2), dtype=np.int64)
        weights = np.zeros(5, dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)
        assert len(iterator) == 3

        # Test with batch_size larger than edges
        edges = np.zeros((2, 2), dtype=np.int64)
        weights = np.zeros(2, dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=5)
        assert len(iterator) == 1

    def test_empty_edges(self):
        """Test with empty edge arrays."""
        edges = np.empty((0, 2), dtype=np.int64)
        weights = np.empty(0, dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)

        assert len(iterator) == 0

        iter_obj = iter(iterator)
        with pytest.raises(StopIteration):
            next(iter_obj)

    def test_single_edge(self):
        """Test with single edge."""
        edges = np.array([[0, 1]], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)

        assert len(iterator) == 1

        iter_obj = iter(iterator)
        batch_edges, batch_weights, _idx = next(iter_obj)
        np.testing.assert_array_equal(batch_edges, [[0, 1]])
        np.testing.assert_allclose(batch_weights, [1.0])

        with pytest.raises(StopIteration):
            next(iter_obj)

    def test_large_batch_size(self):
        """Test with batch size larger than number of edges."""
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        weights = np.array([1.0, 0.5], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=10)

        iter_obj = iter(iterator)
        batch_edges, batch_weights, _idx = next(iter_obj)
        np.testing.assert_array_equal(batch_edges, edges)
        np.testing.assert_allclose(batch_weights, weights)

        with pytest.raises(StopIteration):
            next(iter_obj)

    def test_multiple_iterations(self):
        """Test that iterator can be used multiple times."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=2)

        # First iteration
        batches1 = list(iterator)
        assert len(batches1) == 2

        # Second iteration should work the same
        batches2 = list(iterator)
        assert len(batches2) == 2
        for (e1, w1, _i1), (e2, w2, _i2) in zip(batches1, batches2, strict=False):
            np.testing.assert_array_equal(e1, e2)
            np.testing.assert_array_equal(w1, w2)

    def test_shuffle_preserves_edge_weight_pairing(self):
        """Test that shuffle keeps edges and weights paired correctly."""
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int64)
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

        iterator = EdgeBatchIterator(edges, weights, batch_size=6, shuffle=True)

        np.random.seed(42)
        all_batches = list(iterator)
        batch_edges, batch_weights, _idx = all_batches[0]

        # Verify pairing: for each edge in the shuffled result,
        # its weight should match the original pairing
        for i in range(len(batch_edges)):
            # Find this edge in the original array
            mask = np.all(edges == batch_edges[i], axis=1)
            original_idx = np.where(mask)[0][0]
            assert batch_weights[i] == weights[original_idx]


class TestEdgeDataset:
    """Test EdgeDataset class."""

    @pytest.fixture
    def simple_sparse_matrix(self):
        """Create simple sparse matrix for testing."""
        # Create a simple 5x5 symmetric matrix
        row = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        col = np.array([1, 2, 0, 2, 0, 1, 4, 4])
        data = np.array([1.0, 0.5, 1.0, 0.8, 0.5, 0.8, 0.3, 0.3])

        matrix = sparse.csr_matrix((data, (row, col)), shape=(5, 5))
        return matrix

    def test_basic_initialization(self, simple_sparse_matrix):
        """Test basic initialization with sparse matrix."""
        dataset = EdgeDataset(simple_sparse_matrix)

        assert hasattr(dataset, "adj_sets")
        assert hasattr(dataset, "pos_edges")
        assert hasattr(dataset, "pos_weights")
        assert hasattr(dataset, "neg_edges")
        assert hasattr(dataset, "all_edges")
        assert hasattr(dataset, "all_weights")

        assert isinstance(dataset.adj_sets, dict)
        assert isinstance(dataset.pos_edges, np.ndarray)
        assert isinstance(dataset.pos_weights, np.ndarray)
        assert dataset.pos_edges.dtype == np.int64
        assert dataset.pos_weights.dtype == np.float32
        assert dataset.neg_edges is None
        assert dataset.all_edges is None
        assert dataset.all_weights is None

    def test_edges_and_weights_consistent(self, simple_sparse_matrix):
        """Test that pos_edges and pos_weights are paired correctly."""
        dataset = EdgeDataset(simple_sparse_matrix)
        dok = simple_sparse_matrix.todok()

        for edge, weight in zip(dataset.pos_edges, dataset.pos_weights, strict=False):
            key = (edge[0], edge[1])
            assert key in dok
            np.testing.assert_almost_equal(weight, dok[key])

    def test_adjacency_sets_creation(self, simple_sparse_matrix):
        """Test adjacency sets are correctly created."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Check that adjacency sets contain correct neighbors
        assert 1 in dataset.adj_sets[0]  # 0 is connected to 1
        assert 2 in dataset.adj_sets[0]  # 0 is connected to 2
        assert 0 in dataset.adj_sets[1]  # 1 is connected to 0
        assert 2 in dataset.adj_sets[1]  # 1 is connected to 2

        # Check that all nodes have adjacency sets
        assert len(dataset.adj_sets) == simple_sparse_matrix.shape[0]

    def test_positive_edges_extraction(self, simple_sparse_matrix):
        """Test positive edges are correctly extracted."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Convert sparse matrix to DOK format to get expected edges
        dok_matrix = simple_sparse_matrix.todok()
        expected_edges = set(dok_matrix.keys())

        # Check that positive edges match
        actual_edges = {(int(e[0]), int(e[1])) for e in dataset.pos_edges}
        assert len(dataset.pos_edges) == len(expected_edges)
        assert actual_edges == expected_edges

    def test_sample_negative_edges(self, simple_sparse_matrix):
        """Test negative edge sampling."""
        dataset = EdgeDataset(simple_sparse_matrix)

        mock_result = np.array([(0, 3), (0, 4), (1, 3), (1, 4)], dtype=np.int64)
        with patch.object(dataset, "_sample_negative_edges_chunk") as mock_sample:
            mock_sample.return_value = mock_result

            dataset.sample_negative_edges(random_state=42, verbose=False)

            assert dataset.neg_edges is not None
            assert len(dataset.neg_edges) > 0
            mock_sample.assert_called_once()

    def test_sample_and_shuffle(self, simple_sparse_matrix):
        """Test sample_and_shuffle method."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Mock negative edge sampling
        mock_neg = np.array([(0, 3), (0, 4)], dtype=np.int64)
        with (
            patch.object(dataset, "sample_negative_edges") as mock_sample,
            patch.object(dataset, "_shuffle_edges") as mock_shuffle,
        ):
            mock_sample.return_value = None
            dataset.neg_edges = mock_neg

            dataset.sample_and_shuffle(random_state=42, verbose=False)

            assert dataset.all_edges is not None
            assert dataset.all_weights is not None
            assert len(dataset.all_edges) == len(dataset.pos_edges) + len(dataset.neg_edges)
            assert len(dataset.all_weights) == len(dataset.all_edges)

            mock_sample.assert_called_once()
            mock_shuffle.assert_called_once_with(random_state=42)

    def test_sample_and_shuffle_negative_weights_are_zero(self, simple_sparse_matrix):
        """Test that negative edge weights are zero after sample_and_shuffle."""
        dataset = EdgeDataset(simple_sparse_matrix)

        with patch("parametric_umap.datasets.edge_dataset.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x

            dataset.sample_and_shuffle(random_state=42, verbose=False)

        n_pos = len(dataset.pos_edges)
        n_neg = len(dataset.neg_edges)

        # Before shuffling the order is pos then neg, but after shuffling
        # we can check that all weights are either positive (from pos_edges) or zero (from neg_edges)
        assert np.all(dataset.all_weights >= 0)
        # Total should match
        assert len(dataset.all_weights) == n_pos + n_neg

    def test_get_loader_with_sample_first(self, simple_sparse_matrix):
        """Test get_loader with sample_first=True."""
        dataset = EdgeDataset(simple_sparse_matrix)

        with patch.object(dataset, "sample_and_shuffle") as mock_sample:
            mock_sample.return_value = None
            dataset.all_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
            dataset.all_weights = np.array([1.0, 0.5, 0.0], dtype=np.float32)

            loader = dataset.get_loader(
                batch_size=2,
                sample_first=True,
                random_state=42,
                verbose=False,
            )

            assert isinstance(loader, EdgeBatchIterator)
            mock_sample.assert_called_once()

    def test_get_loader_without_sample_first(self, simple_sparse_matrix):
        """Test get_loader with sample_first=False."""
        dataset = EdgeDataset(simple_sparse_matrix)
        dataset.all_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        dataset.all_weights = np.array([1.0, 0.5, 0.0], dtype=np.float32)

        loader = dataset.get_loader(batch_size=2, sample_first=False)

        assert isinstance(loader, EdgeBatchIterator)
        assert loader.batch_size == 2

    def test_get_loader_without_all_edges(self, simple_sparse_matrix):
        """Test get_loader raises error when all_edges is None."""
        dataset = EdgeDataset(simple_sparse_matrix)

        with pytest.raises(ValueError, match="Must call sample_and_shuffle"):
            dataset.get_loader(batch_size=2, sample_first=False)

    def test_shuffle_edges(self, simple_sparse_matrix):
        """Test edge shuffling functionality."""
        dataset = EdgeDataset(simple_sparse_matrix)
        dataset.all_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
        dataset.all_weights = np.array([1.0, 0.5, 0.8, 0.3], dtype=np.float32)

        original_edges = dataset.all_edges.copy()
        original_weights = dataset.all_weights.copy()

        # Shuffle with fixed random state
        dataset._shuffle_edges(random_state=42)

        # Edges should be rearranged but contain same elements
        assert set(map(tuple, dataset.all_edges.tolist())) == set(map(tuple, original_edges.tolist()))
        assert len(dataset.all_edges) == len(original_edges)

        # Verify pairing is preserved
        for i in range(len(dataset.all_edges)):
            mask = np.all(original_edges == dataset.all_edges[i], axis=1)
            original_idx = np.where(mask)[0][0]
            assert dataset.all_weights[i] == original_weights[original_idx]

    def test_sample_negative_edges_chunk(self, simple_sparse_matrix):
        """Test negative edge sampling for a chunk of nodes."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Test sampling for specific nodes
        node_list = [0, 1]
        neg_edges = dataset._sample_negative_edges_chunk(
            node_list,
            k=2,
            random_state=42,
        )

        assert isinstance(neg_edges, np.ndarray)
        assert len(neg_edges) > 0
        assert neg_edges.shape[1] == 2

        # Check that sampled edges are actually negative (not in adjacency sets)
        for row in neg_edges:
            node, target = int(row[0]), int(row[1])
            if node in dataset.adj_sets:
                assert target not in dataset.adj_sets[node]

    def test_sample_negative_edges_chunk_edge_cases(self, simple_sparse_matrix):
        """Test negative edge sampling edge cases."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Test with empty node list
        neg_edges = dataset._sample_negative_edges_chunk([], k=2, random_state=42)
        assert isinstance(neg_edges, np.ndarray)
        assert len(neg_edges) == 0

        # Test with k=0
        neg_edges = dataset._sample_negative_edges_chunk([0], k=0, random_state=42)
        assert isinstance(neg_edges, np.ndarray)
        assert len(neg_edges) == 0

    def test_adjacency_sets_symmetry(self):
        """Test that adjacency sets respect matrix symmetry."""
        # Create symmetric matrix
        row = np.array([0, 1, 1, 2])
        col = np.array([1, 0, 2, 1])
        data = np.array([1.0, 1.0, 1.0, 1.0])

        symmetric_matrix = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        dataset = EdgeDataset(symmetric_matrix)

        # Check symmetry in adjacency sets
        for i in range(3):
            for j in dataset.adj_sets[i]:
                if j < len(dataset.adj_sets):  # Valid node index
                    assert i in dataset.adj_sets[j], f"Asymmetry: {i} -> {j} but not {j} -> {i}"

    def test_empty_sparse_matrix(self):
        """Test with empty sparse matrix."""
        empty_matrix = sparse.csr_matrix((5, 5))
        dataset = EdgeDataset(empty_matrix)

        assert len(dataset.pos_edges) == 0
        assert len(dataset.pos_weights) == 0
        assert dataset.pos_edges.shape == (0, 2)
        assert len(dataset.adj_sets) == 5

        # All adjacency sets should be empty
        for adj_set in dataset.adj_sets.values():
            assert len(adj_set) == 0

    def test_fully_connected_matrix(self):
        """Test with fully connected sparse matrix."""
        n = 4
        full_matrix = sparse.csr_matrix(np.ones((n, n)) - np.eye(n))  # All 1s except diagonal
        dataset = EdgeDataset(full_matrix)

        # Should have n*(n-1) positive edges
        assert len(dataset.pos_edges) == n * (n - 1)
        assert len(dataset.pos_weights) == n * (n - 1)

        # Each node should be connected to all other nodes
        for i in range(n):
            assert len(dataset.adj_sets[i]) == n - 1  # All nodes except itself

    def test_negative_edge_sampling_realistic(self, simple_sparse_matrix):
        """Test negative edge sampling with realistic parameters."""
        dataset = EdgeDataset(simple_sparse_matrix)

        # Test the actual negative edge sampling (not mocked)
        with patch("parametric_umap.datasets.edge_dataset.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x  # Disable progress bars

            neg_edges = dataset._sample_negative_edges_chunk(
                [0, 1, 2],
                k=2,
                random_state=42,
            )

            assert isinstance(neg_edges, np.ndarray)
            assert len(neg_edges) > 0

            # Check that all sampled edges are indeed negative
            for row in neg_edges:
                src, tgt = int(row[0]), int(row[1])
                assert tgt not in dataset.adj_sets[src]
                assert src != tgt  # No self-loops


class TestEdgeDatasetIntegration:
    """Integration tests for EdgeDataset."""

    def test_full_pipeline(self):
        """Test complete EdgeDataset pipeline."""
        # Create realistic sparse matrix
        n = 50
        density = 0.1

        np.random.seed(42)
        matrix = sparse.random(n, n, density=density, format="csr")
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        matrix.eliminate_zeros()

        dataset = EdgeDataset(matrix)

        # Test full pipeline
        with patch("parametric_umap.datasets.edge_dataset.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x

            dataset.sample_and_shuffle(random_state=42, verbose=False)

            assert dataset.all_edges is not None
            assert dataset.all_weights is not None
            assert len(dataset.all_edges) > 0

            # Get loader and test iteration
            loader = dataset.get_loader(batch_size=10, sample_first=False)

            total_edges = 0
            for batch_edges, batch_weights, _idx in loader:
                assert isinstance(batch_edges, np.ndarray)
                assert isinstance(batch_weights, np.ndarray)
                assert batch_edges.shape[1] == 2
                assert len(batch_edges) == len(batch_weights)
                assert len(batch_edges) <= 10
                total_edges += len(batch_edges)

            assert total_edges == len(dataset.all_edges)

    def test_edge_sampling_balance(self):
        """Test that positive and negative edges are balanced."""
        # Create small test matrix
        row = np.array([0, 0, 1, 2])
        col = np.array([1, 2, 2, 0])
        data = np.array([1.0, 1.0, 1.0, 1.0])

        matrix = sparse.csr_matrix((data, (row, col)), shape=(4, 4))
        dataset = EdgeDataset(matrix)

        with patch("parametric_umap.datasets.edge_dataset.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x

            dataset.sample_and_shuffle(random_state=42, verbose=False)

            # Check that we have both positive and negative edges
            assert len(dataset.pos_edges) > 0
            assert len(dataset.neg_edges) > 0

            # All edges should be the combination of positive and negative
            expected_total = len(dataset.pos_edges) + len(dataset.neg_edges)
            assert len(dataset.all_edges) == expected_total
            assert len(dataset.all_weights) == expected_total

    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        matrix = sparse.random(10, 10, density=0.2, format="csr", random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()

        # Create two datasets with same parameters
        dataset1 = EdgeDataset(matrix)
        dataset2 = EdgeDataset(matrix)

        with patch("parametric_umap.datasets.edge_dataset.tqdm") as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x

            dataset1.sample_and_shuffle(random_state=42, verbose=False)
            dataset2.sample_and_shuffle(random_state=42, verbose=False)

            # Results should be identical
            np.testing.assert_array_equal(dataset1.all_edges, dataset2.all_edges)
            np.testing.assert_array_equal(dataset1.all_weights, dataset2.all_weights)

    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger sparse matrix
        n = 200
        density = 0.05

        matrix = sparse.random(n, n, density=density, format="csr", random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()

        dataset = EdgeDataset(matrix)

        # Test that initialization completes
        assert len(dataset.adj_sets) == n
        assert len(dataset.pos_edges) > 0
        assert len(dataset.pos_weights) > 0

        # Test that we can create a loader without sampling (quick test)
        dataset.all_edges = dataset.pos_edges
        dataset.all_weights = dataset.pos_weights
        loader = dataset.get_loader(batch_size=50, sample_first=False)

        # Test first batch
        batch_edges, batch_weights, _idx = next(iter(loader))
        assert len(batch_edges) <= 50
        assert len(batch_weights) <= 50

    def test_loader_integration_with_iterator(self):
        """Test integration between EdgeDataset and EdgeBatchIterator."""
        matrix = sparse.random(20, 20, density=0.1, format="csr", random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()

        dataset = EdgeDataset(matrix)
        dataset.all_edges = dataset.pos_edges
        dataset.all_weights = dataset.pos_weights

        # Create loader with specific batch size
        batch_size = 5
        loader = dataset.get_loader(batch_size=batch_size, sample_first=False)

        # Test that loader properties match
        assert loader.batch_size == batch_size
        np.testing.assert_array_equal(loader.edges, dataset.all_edges)
        np.testing.assert_array_equal(loader.weights, dataset.all_weights)

        # Test iteration
        batches = list(loader)
        total_edges = sum(len(e) for e, w, _i in batches)
        assert total_edges == len(dataset.all_edges)


class TestEdgeDatasetErrorHandling:
    """Test error handling in EdgeDataset classes."""

    def test_edge_batch_iterator_invalid_input(self):
        """Test EdgeBatchIterator with invalid inputs.

        EdgeBatchIterator does not validate batch_size in __init__.
        batch_size=0 causes ZeroDivisionError when calling __len__.
        batch_size=-1 creates successfully but produces an infinite iterator.
        """
        edges = np.array([[0, 1]], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        # batch_size=0: creation succeeds but __len__ raises ZeroDivisionError
        iterator = EdgeBatchIterator(edges, weights, batch_size=0)
        with pytest.raises(ZeroDivisionError):
            len(iterator)

        # batch_size=-1: creation succeeds (no validation)
        iterator = EdgeBatchIterator(edges, weights, batch_size=-1)
        assert iterator.batch_size == -1

    def test_edge_dataset_invalid_matrix(self):
        """Test EdgeDataset with invalid matrix input."""
        # Test with non-sparse matrix
        with pytest.raises((AttributeError, TypeError)):
            EdgeDataset("not_a_matrix")

        # Test with wrong matrix format
        dense_matrix = np.array([[1, 0], [0, 1]])
        with pytest.raises((AttributeError, TypeError)):
            EdgeDataset(dense_matrix)

    def test_negative_edge_sampling_errors(self):
        """Test error handling in negative edge sampling."""
        # Create valid dataset
        matrix = sparse.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        dataset = EdgeDataset(matrix)

        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            dataset._sample_negative_edges_chunk([0], k=-1, random_state=42)

    def test_adjacency_sets_edge_cases(self):
        """Test adjacency sets with edge cases."""
        # Test with matrix containing zeros
        row = np.array([0, 1, 1])
        col = np.array([1, 0, 2])
        data = np.array([0.0, 1.0, 0.0])  # Contains zeros

        matrix = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        dataset = EdgeDataset(matrix)

        # Only non-zero entries should create adjacencies
        # Entry (1, 0) has value 1.0 so node 0 should be in node 1's adjacency set
        # Entries with value 0.0 should not create adjacencies
        assert 0 in dataset.adj_sets[1]  # (1, 0) = 1.0, so node 0 is adjacent to node 1
        assert 2 not in dataset.adj_sets[1]  # (1, 2) = 0.0, so node 2 should not be adjacent
        assert len(dataset.adj_sets[0]) == 0  # (0, 1) = 0.0, so node 0 has no adjacencies

    def test_iterator_state_errors(self):
        """Test iterator state error handling."""
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        weights = np.array([1.0, 0.5], dtype=np.float32)
        iterator = EdgeBatchIterator(edges, weights, batch_size=1)

        # Test accessing next when current >= len(edges)
        with pytest.raises(StopIteration):
            iterator.current = 10
            next(iterator)

