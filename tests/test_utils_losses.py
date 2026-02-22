"""Comprehensive tests for parametric_umap.utils.losses module."""

import pytest
import torch

from parametric_umap.utils.losses import compute_correlation_loss


class TestComputeCorrelationLoss:
    """Test the compute_correlation_loss function."""

    def test_basic_functionality(self):
        """Test basic correlation loss computation."""
        # Create simple test data with known correlation
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect positive correlation

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should be close to -1 (negative of perfect positive correlation)
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Should be scalar

    def test_perfect_positive_correlation(self):
        """Test with perfectly positively correlated data."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([2.0, 4.0, 6.0, 8.0])  # Perfect positive correlation (scaled)

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should be close to -1
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_perfect_negative_correlation(self):
        """Test with perfectly negatively correlated data."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([4.0, 3.0, 2.0, 1.0])  # Perfect negative correlation

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should be close to 1 (negative of perfect negative correlation)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)

    def test_no_correlation(self):
        """Test with uncorrelated data."""
        # Create data with zero correlation
        X_distances = torch.tensor([1.0, 2.0, 1.0, 2.0])
        Z_distances = torch.tensor([1.0, 1.0, 2.0, 2.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should be close to 0 (negative of zero correlation)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_identical_values(self):
        """Test with identical values (zero variance)."""
        X_distances = torch.tensor([2.0, 2.0, 2.0, 2.0])
        Z_distances = torch.tensor([3.0, 3.0, 3.0, 3.0])

        # Zero variance results in NaN correlation
        loss = compute_correlation_loss(X_distances, Z_distances)
        assert torch.isnan(loss)

    def test_zero_variance_single_input(self):
        """Test with zero variance in one input."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Has variance
        Z_distances = torch.tensor([5.0, 5.0, 5.0, 5.0])  # No variance

        # Zero variance in one input results in NaN correlation
        loss = compute_correlation_loss(X_distances, Z_distances)
        assert torch.isnan(loss)

    def test_single_element(self):
        """Test with single element tensors."""
        X_distances = torch.tensor([1.0])
        Z_distances = torch.tensor([2.0])

        # Single element has no variance, results in NaN
        loss = compute_correlation_loss(X_distances, Z_distances)
        assert torch.isnan(loss)

    def test_two_elements(self):
        """Test with two element tensors."""
        X_distances = torch.tensor([1.0, 2.0])
        Z_distances = torch.tensor([3.0, 4.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Two elements with same direction should give perfect correlation
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_different_scales(self):
        """Test with data at different scales."""
        X_distances = torch.tensor([0.1, 0.2, 0.3, 0.4])
        Z_distances = torch.tensor([100.0, 200.0, 300.0, 400.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Correlation should be scale-invariant
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_negative_distances(self):
        """Test with negative distance values."""
        # Note: In practice, distances should be non-negative,
        # but mathematically the correlation can handle negative values
        X_distances = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        Z_distances = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should still compute correlation correctly
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_large_values(self):
        """Test with large distance values."""
        X_distances = torch.tensor([1000.0, 2000.0, 3000.0, 4000.0])
        Z_distances = torch.tensor([5000.0, 10000.0, 15000.0, 20000.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should handle large values without numerical issues
        assert torch.isfinite(loss)
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_small_values(self):
        """Test with very small distance values."""
        X_distances = torch.tensor([1e-6, 2e-6, 3e-6, 4e-6])
        Z_distances = torch.tensor([2e-6, 4e-6, 6e-6, 8e-6])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should handle small values without numerical issues
        assert torch.isfinite(loss)
        assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)

    def test_random_data_properties(self):
        """Test properties with random data."""
        torch.manual_seed(42)

        # Generate random data
        batch_size = 100
        X_distances = torch.rand(batch_size) * 10
        Z_distances = torch.rand(batch_size) * 5

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Loss should be finite
        assert torch.isfinite(loss)

        # Loss should be in reasonable range for correlation
        assert -1.0 <= loss.item() <= 1.0

    def test_gradient_computation(self):
        """Test that gradients are properly computed."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        Z_distances = torch.tensor([1.5, 2.5, 3.5, 4.5], requires_grad=True)

        loss = compute_correlation_loss(X_distances, Z_distances)
        loss.backward()

        # Gradients should exist and be finite
        assert X_distances.grad is not None
        assert Z_distances.grad is not None
        assert torch.isfinite(X_distances.grad).all()
        assert torch.isfinite(Z_distances.grad).all()

    def test_batch_independence(self):
        """Test that function works with different batch sizes."""
        batch_sizes = [2, 5, 10, 50, 100]

        for batch_size in batch_sizes:
            torch.manual_seed(42)  # For reproducibility
            X_distances = torch.randn(batch_size).abs() + 0.1  # Ensure positive
            Z_distances = X_distances * 2 + torch.randn(batch_size) * 0.1  # Correlated with noise

            loss = compute_correlation_loss(X_distances, Z_distances)

            assert torch.isfinite(loss)
            assert isinstance(loss, torch.Tensor)
            assert loss.numel() == 1

    def test_numerical_stability(self):
        """Test numerical stability with extreme cases."""
        # Test with values that might cause numerical issues
        test_cases = [
            # Very close to zero variance
            (torch.tensor([1.0, 1.0001, 1.0002, 1.0003]), torch.tensor([2.0, 2.0001, 2.0002, 2.0003])),
            # Different magnitudes
            (torch.tensor([1e-8, 2e-8, 3e-8, 4e-8]), torch.tensor([1e8, 2e8, 3e8, 4e8])),
        ]

        for X_distances, Z_distances in test_cases:
            loss = compute_correlation_loss(X_distances, Z_distances)

            assert torch.isfinite(loss)
            assert not torch.isnan(loss)

    def test_data_types(self):
        """Test with different tensor data types."""
        X_distances_float32 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        Z_distances_float32 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        loss_float32 = compute_correlation_loss(X_distances_float32, Z_distances_float32)

        X_distances_float64 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        Z_distances_float64 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

        loss_float64 = compute_correlation_loss(X_distances_float64, Z_distances_float64)

        # Results should be similar regardless of precision
        assert torch.isclose(loss_float32, loss_float64.float(), atol=1e-6)

    def test_device_compatibility(self):
        """Test that function works on different devices."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Test on CPU
        loss_cpu = compute_correlation_loss(X_distances, Z_distances)

        assert loss_cpu.device == torch.device("cpu")

        # Test on GPU if available
        if torch.cuda.is_available():
            X_distances_gpu = X_distances.cuda()
            Z_distances_gpu = Z_distances.cuda()

            loss_gpu = compute_correlation_loss(X_distances_gpu, Z_distances_gpu)

            assert loss_gpu.device.type == "cuda"
            assert torch.isclose(loss_cpu, loss_gpu.cpu(), atol=1e-6)

    def test_mathematical_correctness(self):
        """Test mathematical correctness against manual calculation."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        Z_distances = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Manual calculation
        X_mean = X_distances.mean()
        Z_mean = Z_distances.mean()
        X_centered = X_distances - X_mean
        Z_centered = Z_distances - Z_mean

        numerator = (X_centered * Z_centered).mean()
        X_std = torch.sqrt((X_centered**2).mean())
        Z_std = torch.sqrt((Z_centered**2).mean())

        expected_correlation = numerator / (X_std * Z_std)
        expected_loss = -expected_correlation

        assert torch.isclose(loss, expected_loss, atol=1e-6)


class TestCorrelationLossIntegration:
    """Integration tests for correlation loss."""

    def test_optimization_scenario(self):
        """Test correlation loss in optimization scenario."""
        # Simulate optimization where we want to maximize correlation
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # Start with uncorrelated Z_distances
        Z_distances = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0], requires_grad=True)

        optimizer = torch.optim.SGD([Z_distances], lr=0.1)

        initial_loss = compute_correlation_loss(X_distances, Z_distances).item()

        # Optimize for a few steps
        for _ in range(100):
            optimizer.zero_grad()
            loss = compute_correlation_loss(X_distances, Z_distances)
            loss.backward()
            optimizer.step()

        final_loss = compute_correlation_loss(X_distances, Z_distances).item()

        # Loss should decrease (correlation should increase)
        assert final_loss < initial_loss

    def test_with_realistic_distances(self):
        """Test with realistic distance values from embeddings."""
        # Simulate realistic scenario with embedding distances
        torch.manual_seed(42)

        # Original space distances (e.g., Euclidean distances between data points)
        X_distances = torch.rand(200) * 10  # Distances up to 10 units

        # Embedding space distances (should be correlated but not identical)
        Z_distances = X_distances * 0.8 + torch.randn(200) * 0.5
        Z_distances = torch.clamp(Z_distances, min=0)  # Ensure non-negative

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Should be finite and in reasonable range
        assert torch.isfinite(loss)
        assert -1.0 <= loss.item() <= 1.0

        # Should be negative (positive correlation)
        assert loss.item() < 0

    def test_loss_symmetry(self):
        """Test that loss is symmetric with respect to input order."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([1.5, 2.5, 3.5, 4.5])

        loss1 = compute_correlation_loss(X_distances, Z_distances)
        loss2 = compute_correlation_loss(Z_distances, X_distances)

        # Should be identical (correlation is symmetric)
        assert torch.isclose(loss1, loss2, atol=1e-6)

    def test_multiple_batch_scenarios(self):
        """Test with multiple different batch scenarios."""
        scenarios = [
            # Highly correlated
            (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.1, 2.1, 3.1, 4.1])),
            # Moderately correlated
            (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.5, 2.2, 3.3, 3.8])),
            # Weakly correlated
            (torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([2.1, 1.8, 3.2, 3.9])),
        ]

        losses = []
        for X_distances, Z_distances in scenarios:
            loss = compute_correlation_loss(X_distances, Z_distances)
            losses.append(loss.item())

            assert torch.isfinite(loss)

        # More correlated scenarios should have more negative loss
        assert losses[0] < losses[1] < losses[2]


class TestCorrelationLossErrorHandling:
    """Test error handling in correlation loss."""

    def test_mismatched_tensor_sizes(self):
        """Test handling of mismatched tensor sizes."""
        X_distances = torch.tensor([1.0, 2.0, 3.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Different size

        with pytest.raises((RuntimeError, ValueError)):
            compute_correlation_loss(X_distances, Z_distances)

    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        X_distances = torch.tensor([])
        Z_distances = torch.tensor([])

        # Empty tensors result in NaN
        loss = compute_correlation_loss(X_distances, Z_distances)
        assert torch.isnan(loss)

    def test_nan_input(self):
        """Test handling of NaN inputs."""
        X_distances = torch.tensor([1.0, float("nan"), 3.0, 4.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Result should be NaN due to NaN input
        assert torch.isnan(loss)

    def test_inf_input(self):
        """Test handling of infinite inputs."""
        X_distances = torch.tensor([1.0, float("inf"), 3.0, 4.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss = compute_correlation_loss(X_distances, Z_distances)

        # Result handling depends on implementation, should not crash
        assert isinstance(loss, torch.Tensor)

    def test_wrong_tensor_dimensions(self):
        """Test handling of wrong tensor dimensions."""
        # 2D tensor instead of 1D
        X_distances = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        with pytest.raises((RuntimeError, ValueError)):
            compute_correlation_loss(X_distances, Z_distances)


@pytest.mark.unit
class TestCorrelationLossProperties:
    """Test mathematical properties of correlation loss."""

    def test_correlation_bounds(self):
        """Test that correlation is bounded between -1 and 1."""
        # Generate multiple random test cases
        torch.manual_seed(42)

        for _ in range(50):
            batch_size = torch.randint(5, 100, (1,)).item()
            X_distances = torch.rand(batch_size) * 100
            Z_distances = torch.rand(batch_size) * 100

            loss = compute_correlation_loss(X_distances, Z_distances)

            # Negative correlation should be between -1 and 1
            assert -1.0 <= loss.item() <= 1.0

    def test_linearity_invariance(self):
        """Test that correlation is invariant to linear transformations."""
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([2.0, 4.0, 6.0, 8.0])

        loss_original = compute_correlation_loss(X_distances, Z_distances)

        # Apply linear transformation to Z_distances
        Z_distances_scaled = Z_distances * 3 + 5
        loss_scaled = compute_correlation_loss(X_distances, Z_distances_scaled)

        # Correlation should be identical
        assert torch.isclose(loss_original, loss_scaled, atol=1e-6)

    def test_sign_consistency(self):
        """Test that the sign of correlation is consistent."""
        # Positive correlation case
        X_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])
        Z_distances = torch.tensor([1.0, 2.0, 3.0, 4.0])

        loss_positive = compute_correlation_loss(X_distances, Z_distances)
        assert loss_positive.item() < 0  # Negative of positive correlation

        # Negative correlation case
        Z_distances_neg = torch.tensor([4.0, 3.0, 2.0, 1.0])
        loss_negative = compute_correlation_loss(X_distances, Z_distances_neg)
        assert loss_negative.item() > 0  # Negative of negative correlation
