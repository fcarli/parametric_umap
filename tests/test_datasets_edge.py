"""Comprehensive tests for parametric_umap.datasets.edge_dataset module."""

import pytest
import numpy as np
from scipy import sparse
from unittest.mock import patch, Mock

from parametric_umap.datasets.edge_dataset import EdgeBatchIterator, EdgeDataset


class TestEdgeBatchIterator:
    """Test EdgeBatchIterator class."""
    
    def test_basic_initialization(self, edge_list_data):
        """Test basic initialization with edge list."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        batch_size = 2
        
        iterator = EdgeBatchIterator(edges, batch_size)
        
        assert iterator.edges == edges
        assert iterator.batch_size == batch_size
        assert iterator.shuffle is False
        assert iterator.stratify is False
        assert iterator.current == 0
    
    def test_initialization_with_options(self):
        """Test initialization with shuffle and stratify options."""
        edges = [(0, 1), (1, 2), (2, 3)]
        
        iterator = EdgeBatchIterator(edges, batch_size=2, shuffle=True, stratify=True)
        
        assert iterator.shuffle is True
        assert iterator.stratify is True
    
    def test_iter_without_shuffle(self):
        """Test iteration without shuffling."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        iterator = EdgeBatchIterator(edges, batch_size=2, shuffle=False)
        
        # Start iteration
        iter_obj = iter(iterator)
        assert iter_obj is iterator
        
        # Check that edges are not shuffled
        assert iterator.current_edges == edges
        assert iterator.current == 0
    
    def test_iter_with_shuffle(self):
        """Test iteration with shuffling."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        iterator = EdgeBatchIterator(edges, batch_size=2, shuffle=True)
        
        # Set seed for reproducibility
        np.random.seed(42)
        iter_obj = iter(iterator)
        
        # Check that original edges are not modified
        assert iterator.edges == edges
        
        # Check that current_edges might be different (shuffled)
        # Note: might be same by chance, so we just check it's a copy
        assert iterator.current_edges is not iterator.edges
    
    def test_next_basic_batching(self):
        """Test basic batching functionality."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        
        iter_obj = iter(iterator)
        
        # First batch
        batch1 = next(iter_obj)
        assert batch1 == [(0, 1), (1, 2)]
        assert iterator.current == 2
        
        # Second batch
        batch2 = next(iter_obj)
        assert batch2 == [(2, 3), (3, 4)]
        assert iterator.current == 4
        
        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(iter_obj)
    
    def test_next_partial_batch(self):
        """Test batching with partial last batch."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        
        iter_obj = iter(iterator)
        
        # First two full batches
        batch1 = next(iter_obj)
        batch2 = next(iter_obj)
        
        # Partial last batch
        batch3 = next(iter_obj)
        assert batch3 == [(4, 5)]
        
        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(iter_obj)
    
    def test_len(self):
        """Test __len__ method."""
        # Test exact division
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        assert len(iterator) == 2
        
        # Test with remainder
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        assert len(iterator) == 3
        
        # Test with batch_size larger than edges
        edges = [(0, 1), (1, 2)]
        iterator = EdgeBatchIterator(edges, batch_size=5)
        assert len(iterator) == 1
    
    def test_empty_edges(self):
        """Test with empty edge list."""
        edges = []
        iterator = EdgeBatchIterator(edges, batch_size=2)
        
        assert len(iterator) == 0
        
        iter_obj = iter(iterator)
        with pytest.raises(StopIteration):
            next(iter_obj)
    
    def test_single_edge(self):
        """Test with single edge."""
        edges = [(0, 1)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        
        assert len(iterator) == 1
        
        iter_obj = iter(iterator)
        batch = next(iter_obj)
        assert batch == [(0, 1)]
        
        with pytest.raises(StopIteration):
            next(iter_obj)
    
    def test_large_batch_size(self):
        """Test with batch size larger than number of edges."""
        edges = [(0, 1), (1, 2)]
        iterator = EdgeBatchIterator(edges, batch_size=10)
        
        iter_obj = iter(iterator)
        batch = next(iter_obj)
        assert batch == edges
        
        with pytest.raises(StopIteration):
            next(iter_obj)
    
    def test_multiple_iterations(self):
        """Test that iterator can be used multiple times."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        iterator = EdgeBatchIterator(edges, batch_size=2)
        
        # First iteration
        batches1 = list(iterator)
        assert len(batches1) == 2
        
        # Second iteration should work the same
        batches2 = list(iterator)
        assert len(batches2) == 2
        assert batches1 == batches2
    
    def test_shuffle_reproducibility(self):
        """Test that shuffle is reproducible with same random state."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        
        # Create two iterators with same random seed
        np.random.seed(42)
        iterator1 = EdgeBatchIterator(edges, batch_size=2, shuffle=True)
        batches1 = list(iterator1)
        
        np.random.seed(42)
        iterator2 = EdgeBatchIterator(edges, batch_size=2, shuffle=True)
        batches2 = list(iterator2)
        
        # Results should be identical
        assert batches1 == batches2


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
        
        assert hasattr(dataset, 'adj_sets')
        assert hasattr(dataset, 'pos_edges')
        assert hasattr(dataset, 'neg_edges')
        assert hasattr(dataset, 'all_edges')
        
        assert isinstance(dataset.adj_sets, dict)
        assert isinstance(dataset.pos_edges, list)
        assert dataset.neg_edges is None
        assert dataset.all_edges is None
    
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
        expected_edges = list(dok_matrix.keys())
        
        # Check that positive edges match
        assert len(dataset.pos_edges) == len(expected_edges)
        assert set(dataset.pos_edges) == set(expected_edges)
    
    def test_sample_negative_edges(self, simple_sparse_matrix):
        """Test negative edge sampling."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        # Mock the parallel processing to avoid complexity
        with patch.object(dataset, '_sample_negative_edges') as mock_sample:
            mock_sample.return_value = [(0, 3), (0, 4), (1, 3), (1, 4)]
            
            dataset.sample_negative_edges(random_state=42, n_processes=1, verbose=False)
            
            assert dataset.neg_edges is not None
            assert len(dataset.neg_edges) > 0
            mock_sample.assert_called_once()
    
    def test_sample_and_shuffle(self, simple_sparse_matrix):
        """Test sample_and_shuffle method."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        # Mock negative edge sampling
        with patch.object(dataset, 'sample_negative_edges') as mock_sample, \
             patch.object(dataset, '_shuffle_edges') as mock_shuffle:
            
            mock_sample.return_value = None
            dataset.neg_edges = [(0, 3), (0, 4)]  # Set mock negative edges
            
            dataset.sample_and_shuffle(random_state=42, n_processes=1, verbose=False)
            
            assert dataset.all_edges is not None
            assert len(dataset.all_edges) == len(dataset.pos_edges) + len(dataset.neg_edges)
            
            mock_sample.assert_called_once()
            mock_shuffle.assert_called_once_with(random_state=42)
    
    def test_get_loader_with_sample_first(self, simple_sparse_matrix):
        """Test get_loader with sample_first=True."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        with patch.object(dataset, 'sample_and_shuffle') as mock_sample:
            mock_sample.return_value = None
            dataset.all_edges = [(0, 1), (1, 2), (2, 3)]  # Set mock all_edges
            
            loader = dataset.get_loader(
                batch_size=2, 
                sample_first=True, 
                random_state=42,
                n_processes=1,
                verbose=False
            )
            
            assert isinstance(loader, EdgeBatchIterator)
            mock_sample.assert_called_once()
    
    def test_get_loader_without_sample_first(self, simple_sparse_matrix):
        """Test get_loader with sample_first=False."""
        dataset = EdgeDataset(simple_sparse_matrix)
        dataset.all_edges = [(0, 1), (1, 2), (2, 3)]  # Set mock all_edges
        
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
        dataset.all_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        original_edges = dataset.all_edges.copy()
        
        # Shuffle with fixed random state
        dataset._shuffle_edges(random_state=42)
        
        # Edges should be rearranged but contain same elements
        assert set(dataset.all_edges) == set(original_edges)
        assert len(dataset.all_edges) == len(original_edges)
    
    def test_sample_negative_edges_chunk(self, simple_sparse_matrix):
        """Test negative edge sampling for a chunk of nodes."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        # Test sampling for specific nodes
        node_list = [0, 1]
        neg_edges = dataset._sample_negative_edges_chunk(
            node_list, k=2, random_state=42
        )
        
        assert isinstance(neg_edges, list)
        assert len(neg_edges) > 0
        
        # Check that sampled edges are actually negative (not in adjacency sets)
        for node, target in neg_edges:
            if node in dataset.adj_sets:
                assert target not in dataset.adj_sets[node]
    
    def test_sample_negative_edges_chunk_edge_cases(self, simple_sparse_matrix):
        """Test negative edge sampling edge cases."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        # Test with empty node list
        neg_edges = dataset._sample_negative_edges_chunk([], k=2, random_state=42)
        assert neg_edges == []
        
        # Test with k=0
        neg_edges = dataset._sample_negative_edges_chunk([0], k=0, random_state=42)
        assert neg_edges == []
    
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
        
        # Each node should be connected to all other nodes
        for i in range(n):
            assert len(dataset.adj_sets[i]) == n - 1  # All nodes except itself
    
    def test_negative_edge_sampling_realistic(self, simple_sparse_matrix):
        """Test negative edge sampling with realistic parameters."""
        dataset = EdgeDataset(simple_sparse_matrix)
        
        # Test the actual negative edge sampling (not mocked)
        with patch('parametric_umap.datasets.edge_dataset.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x  # Disable progress bars
            
            neg_edges = dataset._sample_negative_edges(
                [0, 1, 2], k=2, random_state=42, n_processes=1, verbose=False
            )
            
            assert isinstance(neg_edges, list)
            assert len(neg_edges) > 0
            
            # Check that all sampled edges are indeed negative
            for src, tgt in neg_edges:
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
        matrix = sparse.random(n, n, density=density, format='csr')
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        matrix.eliminate_zeros()
        
        dataset = EdgeDataset(matrix)
        
        # Test full pipeline
        with patch('parametric_umap.datasets.edge_dataset.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x
            
            dataset.sample_and_shuffle(random_state=42, n_processes=1, verbose=False)
            
            assert dataset.all_edges is not None
            assert len(dataset.all_edges) > 0
            
            # Get loader and test iteration
            loader = dataset.get_loader(batch_size=10, sample_first=False)
            
            total_edges = 0
            for batch in loader:
                assert isinstance(batch, list)
                assert len(batch) <= 10
                total_edges += len(batch)
            
            assert total_edges == len(dataset.all_edges)
    
    def test_edge_sampling_balance(self):
        """Test that positive and negative edges are balanced."""
        # Create small test matrix
        row = np.array([0, 0, 1, 2])
        col = np.array([1, 2, 2, 0])
        data = np.array([1.0, 1.0, 1.0, 1.0])
        
        matrix = sparse.csr_matrix((data, (row, col)), shape=(4, 4))
        dataset = EdgeDataset(matrix)
        
        with patch('parametric_umap.datasets.edge_dataset.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x
            
            dataset.sample_and_shuffle(random_state=42, n_processes=1, verbose=False)
            
            # Check that we have both positive and negative edges
            assert len(dataset.pos_edges) > 0
            assert len(dataset.neg_edges) > 0
            
            # All edges should be the combination of positive and negative
            expected_total = len(dataset.pos_edges) + len(dataset.neg_edges)
            assert len(dataset.all_edges) == expected_total
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        matrix = sparse.random(10, 10, density=0.2, format='csr', random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()
        
        # Create two datasets with same parameters
        dataset1 = EdgeDataset(matrix)
        dataset2 = EdgeDataset(matrix)
        
        with patch('parametric_umap.datasets.edge_dataset.tqdm') as mock_tqdm:
            mock_tqdm.side_effect = lambda x, *args, **kwargs: x
            
            dataset1.sample_and_shuffle(random_state=42, n_processes=1, verbose=False)
            dataset2.sample_and_shuffle(random_state=42, n_processes=1, verbose=False)
            
            # Results should be identical
            assert dataset1.all_edges == dataset2.all_edges
    
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger sparse matrix
        n = 200
        density = 0.05
        
        matrix = sparse.random(n, n, density=density, format='csr', random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()
        
        dataset = EdgeDataset(matrix)
        
        # Test that initialization completes
        assert len(dataset.adj_sets) == n
        assert len(dataset.pos_edges) > 0
        
        # Test that we can create a loader without sampling (quick test)
        dataset.all_edges = dataset.pos_edges  # Skip negative sampling for speed
        loader = dataset.get_loader(batch_size=50, sample_first=False)
        
        # Test first batch
        first_batch = next(iter(loader))
        assert len(first_batch) <= 50
    
    def test_loader_integration_with_iterator(self):
        """Test integration between EdgeDataset and EdgeBatchIterator."""
        matrix = sparse.random(20, 20, density=0.1, format='csr', random_state=42)
        matrix = (matrix + matrix.T) / 2
        matrix.eliminate_zeros()
        
        dataset = EdgeDataset(matrix)
        dataset.all_edges = dataset.pos_edges  # Skip negative sampling
        
        # Create loader with specific batch size
        batch_size = 5
        loader = dataset.get_loader(batch_size=batch_size, sample_first=False)
        
        # Test that loader properties match
        assert loader.batch_size == batch_size
        assert loader.edges == dataset.all_edges
        
        # Test iteration
        batches = list(loader)
        total_edges = sum(len(batch) for batch in batches)
        assert total_edges == len(dataset.all_edges)


class TestEdgeDatasetErrorHandling:
    """Test error handling in EdgeDataset classes."""
    
    def test_edge_batch_iterator_invalid_input(self):
        """Test EdgeBatchIterator with invalid inputs."""
        # Test with invalid batch size
        with pytest.raises((ValueError, TypeError)):
            EdgeBatchIterator([(0, 1)], batch_size=0)
        
        with pytest.raises((ValueError, TypeError)):
            EdgeBatchIterator([(0, 1)], batch_size=-1)
    
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
        assert 1 in dataset.adj_sets[1] or len(dataset.adj_sets[1]) == 0
    
    def test_iterator_state_errors(self):
        """Test iterator state error handling."""
        edges = [(0, 1), (1, 2)]
        iterator = EdgeBatchIterator(edges, batch_size=1)
        
        # Test accessing next without iter
        with pytest.raises(StopIteration):
            # Should raise StopIteration when current >= len(edges)
            iterator.current = 10
            next(iterator)
    
    def test_multiprocessing_errors(self):
        """Test multiprocessing error handling."""
        matrix = sparse.csr_matrix(([1], ([0], [1])), shape=(2, 2))
        dataset = EdgeDataset(matrix)
        
        # Test with invalid n_processes
        with patch('parametric_umap.datasets.edge_dataset.os.cpu_count', return_value=4):
            # Should handle n_processes gracefully
            result = dataset._sample_negative_edges(
                [0], k=1, random_state=42, n_processes=100, verbose=False
            )
            assert isinstance(result, list)