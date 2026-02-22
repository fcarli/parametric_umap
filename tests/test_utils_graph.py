"""Comprehensive tests for parametric_umap.utils.graph module."""

import numpy as np
import pytest
from scipy import sparse

from parametric_umap.utils.graph import (
    compute_all_p_umap,
    compute_p_umap,
    compute_p_umap_symmetric,
    compute_sigma_i,
)


class TestComputeSigmaI:
    """Test the compute_sigma_i function."""

    def test_basic_functionality(self, sample_2d_data):
        """Test basic sigma computation functionality."""
        k = 5
        sigma, rho, distances, neighbors = compute_sigma_i(sample_2d_data, k)

        n_samples = len(sample_2d_data)

        # Check output shapes
        assert sigma.shape == (n_samples,)
        assert rho.shape == (n_samples,)
        assert distances.shape == (n_samples, k)
        assert neighbors.shape == (n_samples, k)

        # Check data types
        assert sigma.dtype == np.float32
        assert rho.dtype == np.float32
        assert distances.dtype == np.float32

        # Check value ranges
        assert np.all(sigma > 0)  # Sigma should be positive
        assert np.all(rho >= 0)  # Rho should be non-negative
        assert np.all(distances >= 0)  # Distances should be non-negative

    def test_different_k_values(self, sample_2d_data):
        """Test with different k values."""
        k_values = [3, 5, 10, 15]
        n_samples = len(sample_2d_data)

        for k in k_values:
            sigma, rho, distances, neighbors = compute_sigma_i(sample_2d_data, k)

            assert sigma.shape == (n_samples,)
            assert distances.shape == (n_samples, k)
            assert neighbors.shape == (n_samples, k)

    def test_small_dataset(self):
        """Test with small dataset."""
        # Create small dataset
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
        k = 2

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)

        assert sigma.shape == (4,)
        assert rho.shape == (4,)
        assert distances.shape == (4, k)
        assert neighbors.shape == (4, k)

        # Check that neighbors are valid indices
        assert np.all(neighbors >= 0)
        assert np.all(neighbors < len(X))

    def test_identical_points(self):
        """Test with some identical points."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
        k = 2

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)

        # Should handle identical points gracefully
        assert sigma.shape == (4,)
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0)

    def test_convergence_parameters(self, sample_2d_data):
        """Test different convergence parameters."""
        k = 5

        # Test with different tolerance
        sigma1, _, _, _ = compute_sigma_i(sample_2d_data, k, tol=1e-3)
        sigma2, _, _, _ = compute_sigma_i(sample_2d_data, k, tol=1e-7)

        # Both should converge but with different precision
        assert np.all(sigma1 > 0)
        assert np.all(sigma2 > 0)

        # Test with different max_iter
        sigma3, _, _, _ = compute_sigma_i(sample_2d_data, k, max_iter=10)
        sigma4, _, _, _ = compute_sigma_i(sample_2d_data, k, max_iter=200)

        assert np.all(sigma3 > 0)
        assert np.all(sigma4 > 0)

    def test_high_dimensional_data(self):
        """Test with high dimensional data."""
        np.random.seed(42)
        X = np.random.randn(50, 20).astype(np.float32)
        k = 5

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)

        assert sigma.shape == (50,)
        assert distances.shape == (50, k)
        assert np.all(sigma > 0)

    def test_single_point(self):
        """Test with single data point."""
        X = np.array([[1.0, 2.0]], dtype=np.float32)
        k = 1

        with pytest.raises((ValueError, IndexError)):
            compute_sigma_i(X, k)

    def test_k_larger_than_dataset(self):
        """Test when k is larger than dataset size."""
        X = np.random.randn(10, 2).astype(np.float32)
        k = 15  # Larger than dataset

        with pytest.raises((ValueError, IndexError)):
            compute_sigma_i(X, k)

    @pytest.mark.parametrize("dtype", [np.float64, np.int32, np.int64])
    def test_input_data_types(self, dtype):
        """Test with different input data types."""
        X = np.random.randn(20, 3).astype(dtype)
        k = 5

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)

        # Output should always be float32
        assert sigma.dtype == np.float32
        assert rho.dtype == np.float32
        assert distances.dtype == np.float32


class TestComputePUmap:
    """Test the compute_p_umap function."""

    def test_basic_functionality(self, sample_2d_data):
        """Test basic probability computation."""
        k = 5
        sigma, rho, distances, neighbors = compute_sigma_i(sample_2d_data, k)
        P = compute_p_umap(sigma, rho, distances, neighbors)

        n_samples = len(sample_2d_data)

        # Check output properties
        assert isinstance(P, sparse.csr_matrix)
        assert P.shape == (n_samples, n_samples)
        assert P.dtype == np.float32

        # Check probability values
        assert np.all(P.data >= 0)  # Probabilities should be non-negative
        assert np.all(P.data <= 1)  # Probabilities should be <= 1

    def test_zero_sigma_handling(self):
        """Test handling of zero sigma values."""
        n_samples = 10
        k = 3

        # Create mock data with some zero sigma values
        sigma = np.array([0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.4, 0.0, 0.5], dtype=np.float32)
        rho = np.random.uniform(0, 1, n_samples).astype(np.float32)
        distances = np.random.uniform(0, 2, (n_samples, k)).astype(np.float32)
        neighbors = np.random.randint(0, n_samples, (n_samples, k))

        P = compute_p_umap(sigma, rho, distances, neighbors)

        # Should handle zero sigma without division by zero
        assert np.all(np.isfinite(P.data))
        assert np.all(P.data >= 0)

    def test_probability_properties(self):
        """Test mathematical properties of computed probabilities."""
        # Create controlled test data
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        k = 2

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)
        P = compute_p_umap(sigma, rho, distances, neighbors)

        # Check that probabilities sum to reasonable values
        row_sums = np.array(P.sum(axis=1)).flatten()

        # Each row should have positive sum (since we have k neighbors)
        assert np.all(row_sums > 0)

        # Check sparsity pattern
        expected_nnz = len(X) * k  # Each point has k neighbors
        assert P.nnz <= expected_nnz * 2  # Allow some flexibility

    def test_different_array_shapes(self):
        """Test with different array shapes and sizes."""
        test_cases = [
            (20, 2, 5),  # 20 samples, 2D, k=5
            (50, 3, 10),  # 50 samples, 3D, k=10
            (10, 5, 3),  # 10 samples, 5D, k=3
        ]

        for n_samples, n_features, k in test_cases:
            X = np.random.randn(n_samples, n_features).astype(np.float32)
            sigma, rho, distances, neighbors = compute_sigma_i(X, k)
            P = compute_p_umap(sigma, rho, distances, neighbors)

            assert P.shape == (n_samples, n_samples)
            assert np.all(P.data >= 0)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        n_samples = 10
        k = 3

        # Test with very large distances
        sigma = np.ones(n_samples, dtype=np.float32)
        rho = np.zeros(n_samples, dtype=np.float32)
        distances = np.full((n_samples, k), 100.0, dtype=np.float32)  # Large distances
        neighbors = np.random.randint(0, n_samples, (n_samples, k))

        P = compute_p_umap(sigma, rho, distances, neighbors)

        # Should handle large distances without overflow
        assert np.all(np.isfinite(P.data))
        assert np.all(P.data >= 0)


class TestComputePUmapSymmetric:
    """Test the compute_p_umap_symmetric function."""

    def test_basic_functionality(self, sample_2d_data):
        """Test basic symmetric probability computation."""
        k = 5
        sigma, rho, distances, neighbors = compute_sigma_i(sample_2d_data, k)
        P = compute_p_umap(sigma, rho, distances, neighbors)
        P_sym = compute_p_umap_symmetric(P)

        n_samples = len(sample_2d_data)

        # Check output properties
        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (n_samples, n_samples)
        assert P_sym.dtype == np.float32

        # Check probability values
        assert np.all(P_sym.data >= 0)  # Should be non-negative

    def test_symmetry_property(self):
        """Test that the result is actually symmetric."""
        # Create small controlled test
        X = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        k = 2

        sigma, rho, distances, neighbors = compute_sigma_i(X, k)
        P = compute_p_umap(sigma, rho, distances, neighbors)
        P_sym = compute_p_umap_symmetric(P)

        # Check symmetry: P_sym should equal P_sym.T
        P_sym_dense = P_sym.toarray()
        P_sym_T_dense = P_sym.T.toarray()

        np.testing.assert_allclose(P_sym_dense, P_sym_T_dense, rtol=1e-6)

    def test_mathematical_formula(self):
        """Test the mathematical formula: P + P.T - P * P.T."""
        # Create simple test matrix
        n = 5
        np.random.seed(42)
        data = np.random.uniform(0, 1, 10)
        rows = np.random.randint(0, n, 10)
        cols = np.random.randint(0, n, 10)

        P = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        P_sym = compute_p_umap_symmetric(P)

        # Manually compute the expected result
        P_transpose = P.transpose()
        expected = P + P_transpose - P.multiply(P_transpose)
        expected.eliminate_zeros()

        # Compare results
        np.testing.assert_array_almost_equal(P_sym.toarray(), expected.toarray())

    def test_zero_matrix(self):
        """Test with zero matrix."""
        n = 5
        P = sparse.csr_matrix((n, n), dtype=np.float32)
        P_sym = compute_p_umap_symmetric(P)

        assert P_sym.shape == (n, n)
        assert P_sym.nnz == 0  # Should remain zero

    def test_sparsity_preservation(self):
        """Test that sparsity is reasonably preserved."""
        # Create sparse matrix
        n = 100
        density = 0.1
        nnz = int(n * n * density)

        np.random.seed(42)
        rows = np.random.randint(0, n, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.random.uniform(0, 1, nnz)

        P = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        P_sym = compute_p_umap_symmetric(P)

        # Symmetric matrix should have reasonable sparsity
        assert P_sym.nnz <= n * n * 0.5  # Some reasonable upper bound


class TestComputeAllPUmap:
    """Test the compute_all_p_umap wrapper function."""

    def test_basic_functionality(self, sample_2d_data):
        """Test basic wrapper functionality."""
        k = 5
        P_sym = compute_all_p_umap(sample_2d_data, k)

        n_samples = len(sample_2d_data)

        # Check output properties
        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (n_samples, n_samples)
        assert np.all(P_sym.data >= 0)

        # Check symmetry
        P_sym_dense = P_sym.toarray()
        P_sym_T_dense = P_sym.T.toarray()
        np.testing.assert_allclose(P_sym_dense, P_sym_T_dense, rtol=1e-6)

    def test_return_distances_and_neighbors(self, sample_2d_data):
        """Test with return_dist_and_neigh=True."""
        k = 5
        result = compute_all_p_umap(sample_2d_data, k, return_dist_and_neigh=True)

        assert len(result) == 3
        P_sym, distances, neighbors = result

        n_samples = len(sample_2d_data)

        # Check P_sym
        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (n_samples, n_samples)

        # Check distances and neighbors
        assert distances.shape == (n_samples, k)
        assert neighbors.shape == (n_samples, k)
        assert np.all(distances >= 0)
        assert np.all(neighbors >= 0)
        assert np.all(neighbors < n_samples)

    def test_parameter_passing(self, sample_2d_data):
        """Test that parameters are passed correctly to sub-functions."""
        k = 5

        # Test with custom parameters
        P_sym = compute_all_p_umap(
            sample_2d_data,
            k,
            tol=1e-3,
            max_iter=50,
        )

        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (len(sample_2d_data), len(sample_2d_data))

    def test_different_datasets(self):
        """Test with different dataset types."""
        datasets = [
            np.random.randn(30, 2).astype(np.float32),  # 2D
            np.random.randn(40, 3).astype(np.float32),  # 3D
            np.random.randn(25, 5).astype(np.float32),  # 5D
        ]

        for X in datasets:
            k = min(5, X.shape[0] - 1)  # Ensure k < n_samples
            P_sym = compute_all_p_umap(X, k)

            assert isinstance(P_sym, sparse.csr_matrix)
            assert P_sym.shape == (X.shape[0], X.shape[0])
            assert np.all(P_sym.data >= 0)


class TestGraphUtilsIntegration:
    """Integration tests for graph utilities."""

    def test_end_to_end_pipeline(self, swiss_roll_data):
        """Test complete pipeline from data to symmetric probabilities."""
        k = 15

        # Run complete pipeline
        P_sym = compute_all_p_umap(swiss_roll_data, k)

        n_samples = len(swiss_roll_data)

        # Check final result properties
        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (n_samples, n_samples)
        assert np.all(P_sym.data >= 0)

        # Check symmetry
        P_sym_dense = P_sym.toarray()
        np.testing.assert_allclose(P_sym_dense, P_sym_dense.T, rtol=1e-6)

        # Check sparsity (should be much sparser than dense)
        sparsity = P_sym.nnz / (n_samples * n_samples)
        assert sparsity < 0.5  # Should be sparse

    def test_reproducibility(self, sample_2d_data):
        """Test that results are reproducible."""
        k = 5

        # Run twice
        P_sym1 = compute_all_p_umap(sample_2d_data, k)
        P_sym2 = compute_all_p_umap(sample_2d_data, k)

        # Results should be identical
        np.testing.assert_array_equal(P_sym1.toarray(), P_sym2.toarray())

    def test_increasing_k_values(self, sample_2d_data):
        """Test behavior with increasing k values."""
        k_values = [3, 5, 10]
        n_samples = len(sample_2d_data)

        for k in k_values:
            P_sym = compute_all_p_umap(sample_2d_data, k)

            assert P_sym.shape == (n_samples, n_samples)
            assert np.all(P_sym.data >= 0)

            # Higher k should generally result in more connections
            sparsity = P_sym.nnz / (n_samples * n_samples)
            assert sparsity > 0  # Should have some connections

    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with larger dataset."""
        # Create larger dataset for performance testing
        np.random.seed(42)
        X = np.random.randn(1000, 10).astype(np.float32)
        k = 15

        P_sym = compute_all_p_umap(X, k)

        assert isinstance(P_sym, sparse.csr_matrix)
        assert P_sym.shape == (1000, 1000)
        assert np.all(P_sym.data >= 0)


class TestGraphUtilsErrorHandling:
    """Test error handling in graph utilities."""

    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        # Test with non-array input
        with pytest.raises((TypeError, AttributeError)):
            compute_all_p_umap("not_an_array", 5)

        # Test with wrong dimensional input
        with pytest.raises((ValueError, IndexError)):
            compute_all_p_umap(np.array([1, 2, 3]), 5)

    def test_invalid_k_values(self, sample_2d_data):
        """Test handling of invalid k values."""
        # Test with k <= 0
        with pytest.raises((ValueError, RuntimeError)):
            compute_all_p_umap(sample_2d_data, 0)

        with pytest.raises((ValueError, RuntimeError)):
            compute_all_p_umap(sample_2d_data, -1)

    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]], dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            compute_all_p_umap(X, 2)

    def test_inf_input_handling(self):
        """Test handling of infinite inputs."""
        X = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]], dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            compute_all_p_umap(X, 2)


@pytest.mark.unit
class TestGraphUtilsUnitTests:
    """Unit tests for individual components."""

    def test_compute_sigma_i_convergence(self):
        """Test that sigma computation converges properly."""
        # Create simple test case where we can verify convergence
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        k = 2

        sigma, rho, distances, neighbors = compute_sigma_i(X, k, tol=1e-8, max_iter=1000)

        # Verify convergence by checking probability sums
        target = np.log2(k)

        for i in range(len(X)):
            exponent = -np.maximum(distances[i] - rho[i], 0) / sigma[i]
            probs = np.exp(exponent)
            prob_sum = probs.sum()

            # Should be close to target
            assert abs(prob_sum - target) < 1e-4

    def test_probability_computation_manual(self):
        """Test probability computation with manual verification."""
        sigma = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        rho = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        distances = np.array([[1.0, 2.0], [1.5, 2.5], [1.2, 2.2]], dtype=np.float32)
        neighbors = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)

        P = compute_p_umap(sigma, rho, distances, neighbors)

        # Manually compute expected values for first row
        expected_01 = np.exp(-max(1.0 - 0.5, 0) / 1.0)  # exp(-0.5)
        expected_02 = np.exp(-max(2.0 - 0.5, 0) / 1.0)  # exp(-1.5)

        P_dense = P.toarray()

        assert abs(P_dense[0, 1] - expected_01) < 1e-6
        assert abs(P_dense[0, 2] - expected_02) < 1e-6
