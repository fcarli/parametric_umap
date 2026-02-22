"""Comprehensive tests for parametric_umap.models.mlp module."""

import pytest
import torch
from torch import nn

from parametric_umap.models.mlp import MLP


class TestMLPInitialization:
    """Test MLP initialization and architecture construction."""

    def test_basic_initialization(self):
        """Test basic MLP initialization with default parameters."""
        input_dim = 10
        hidden_dim = 64
        output_dim = 2

        mlp = MLP(input_dim, hidden_dim, output_dim)

        assert isinstance(mlp, nn.Module)
        assert hasattr(mlp, "model")
        assert isinstance(mlp.model, nn.Sequential)

    def test_custom_parameters(self):
        """Test MLP initialization with custom parameters."""
        mlp = MLP(
            input_dim=5,
            hidden_dim=32,
            output_dim=3,
            num_layers=4,
            use_batchnorm=True,
            use_dropout=True,
            dropout_prob=0.3,
        )

        assert isinstance(mlp.model, nn.Sequential)

        # Check that the model has the expected number of layers
        # With batchnorm and dropout, we expect more layers
        layer_count = len(mlp.model)
        assert layer_count > 4  # Should have many layers due to BN and dropout

    def test_layer_composition_no_extras(self):
        """Test layer composition without batch norm or dropout."""
        mlp = MLP(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            num_layers=2,
            use_batchnorm=False,
            use_dropout=False,
        )

        # Expected layers: Linear -> ReLU -> Linear -> ReLU -> Linear (output)
        # Total: 5 layers
        assert len(mlp.model) == 5

        # Check layer types
        assert isinstance(mlp.model[0], nn.Linear)  # First hidden layer
        assert isinstance(mlp.model[1], nn.ReLU)  # First activation
        assert isinstance(mlp.model[2], nn.Linear)  # Second hidden layer
        assert isinstance(mlp.model[3], nn.ReLU)  # Second activation
        assert isinstance(mlp.model[4], nn.Linear)  # Output layer

    def test_layer_composition_with_batchnorm(self):
        """Test layer composition with batch normalization."""
        mlp = MLP(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            num_layers=2,
            use_batchnorm=True,
            use_dropout=False,
        )

        # Expected: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm -> ReLU -> Linear
        # Total: 7 layers
        assert len(mlp.model) == 7

        # Check for batch norm layers
        batch_norm_layers = [layer for layer in mlp.model if isinstance(layer, nn.BatchNorm1d)]
        assert len(batch_norm_layers) == 2  # Should have 2 batch norm layers

    def test_layer_composition_with_dropout(self):
        """Test layer composition with dropout."""
        mlp = MLP(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            num_layers=2,
            use_batchnorm=False,
            use_dropout=True,
            dropout_prob=0.4,
        )

        # Expected: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        # Total: 7 layers
        assert len(mlp.model) == 7

        # Check for dropout layers
        dropout_layers = [layer for layer in mlp.model if isinstance(layer, nn.Dropout)]
        assert len(dropout_layers) == 2  # Should have 2 dropout layers

        # Check dropout probability
        for dropout_layer in dropout_layers:
            assert dropout_layer.p == 0.4

    def test_layer_composition_with_both_extras(self):
        """Test layer composition with both batch norm and dropout."""
        mlp = MLP(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            num_layers=2,
            use_batchnorm=True,
            use_dropout=True,
            dropout_prob=0.3,
        )

        # Expected: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear
        # Total: 9 layers
        assert len(mlp.model) == 9

        # Check for both types of layers
        batch_norm_layers = [layer for layer in mlp.model if isinstance(layer, nn.BatchNorm1d)]
        dropout_layers = [layer for layer in mlp.model if isinstance(layer, nn.Dropout)]

        assert len(batch_norm_layers) == 2
        assert len(dropout_layers) == 2

    def test_single_layer_network(self):
        """Test MLP with single hidden layer."""
        mlp = MLP(
            input_dim=5,
            hidden_dim=10,
            output_dim=3,
            num_layers=1,
            use_batchnorm=False,
            use_dropout=False,
        )

        # Expected: Linear -> ReLU -> Linear
        # Total: 3 layers
        assert len(mlp.model) == 3

        assert isinstance(mlp.model[0], nn.Linear)  # Hidden layer
        assert isinstance(mlp.model[1], nn.ReLU)  # Activation
        assert isinstance(mlp.model[2], nn.Linear)  # Output layer

    def test_deep_network(self):
        """Test MLP with many hidden layers."""
        num_layers = 5
        mlp = MLP(
            input_dim=3,
            hidden_dim=16,
            output_dim=2,
            num_layers=num_layers,
            use_batchnorm=False,
            use_dropout=False,
        )

        # Expected: (Linear -> ReLU) * num_layers + final Linear
        # Total: num_layers * 2 + 1 = 11 layers
        expected_layers = num_layers * 2 + 1
        assert len(mlp.model) == expected_layers

    @pytest.mark.parametrize(
        "input_dim,hidden_dim,output_dim",
        [
            (1, 2, 1),  # Minimal dimensions
            (100, 50, 10),  # Large dimensions
            (784, 128, 10),  # Common sizes (e.g., MNIST)
        ],
    )
    def test_various_dimensions(self, input_dim, hidden_dim, output_dim):
        """Test MLP with various input/output dimensions."""
        mlp = MLP(input_dim, hidden_dim, output_dim)

        assert isinstance(mlp.model, nn.Sequential)

        # Check first layer input dimension
        first_layer = mlp.model[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == input_dim
        assert first_layer.out_features == hidden_dim

        # Check last layer output dimension
        last_layer = mlp.model[-1]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.out_features == output_dim

    def test_valid_parameter_ranges(self):
        """Test that MLP works with valid parameter ranges."""
        # Test valid positive dimensions
        mlp1 = MLP(1, 10, 2)  # Minimal valid dimensions
        mlp2 = MLP(100, 50, 10)  # Normal dimensions
        mlp3 = MLP(10, 10, 2, num_layers=1)  # Single layer
        mlp4 = MLP(10, 10, 2, dropout_prob=0.5)  # Valid dropout
        mlp5 = MLP(10, 10, 2, dropout_prob=0.0)  # Zero dropout (valid)

        # All should create valid instances
        assert isinstance(mlp1, MLP)
        assert isinstance(mlp2, MLP)
        assert isinstance(mlp3, MLP)
        assert isinstance(mlp4, MLP)
        assert isinstance(mlp5, MLP)

        # Check that models have expected structure
        assert len(mlp3.model) == 3  # Single layer: Linear -> ReLU -> Linear
        assert mlp1.model[0].in_features == 1
        assert mlp1.model[0].out_features == 10


class TestMLPForwardPass:
    """Test MLP forward pass functionality."""

    def test_basic_forward_pass(self):
        """Test basic forward pass functionality."""
        batch_size = 32
        input_dim = 10
        hidden_dim = 64
        output_dim = 2

        mlp = MLP(input_dim, hidden_dim, output_dim)

        # Create input tensor
        x = torch.randn(batch_size, input_dim)

        # Forward pass
        output = mlp(x)

        # Check output shape
        assert output.shape == (batch_size, output_dim)
        assert output.dtype == torch.float32

    def test_single_sample_forward(self):
        """Test forward pass with single sample."""
        input_dim = 5
        hidden_dim = 10
        output_dim = 3

        mlp = MLP(input_dim, hidden_dim, output_dim)

        # Single sample
        x = torch.randn(1, input_dim)
        output = mlp(x)

        assert output.shape == (1, output_dim)

    def test_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        input_dim = 8
        hidden_dim = 16
        output_dim = 4

        mlp = MLP(input_dim, hidden_dim, output_dim)

        batch_sizes = [1, 5, 32, 128]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, input_dim)
            output = mlp(x)

            assert output.shape == (batch_size, output_dim)
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

    def test_forward_with_batchnorm(self):
        """Test forward pass with batch normalization."""
        mlp = MLP(
            input_dim=10,
            hidden_dim=20,
            output_dim=2,
            num_layers=2,
            use_batchnorm=True,
        )

        # Test in training mode
        mlp.train()
        x = torch.randn(16, 10)  # Need sufficient batch size for BN
        output = mlp(x)

        assert output.shape == (16, 2)
        assert not torch.isnan(output).any()

        # Test in evaluation mode
        mlp.eval()
        output_eval = mlp(x)

        assert output_eval.shape == (16, 2)
        assert not torch.isnan(output_eval).any()

    def test_forward_with_dropout(self):
        """Test forward pass with dropout."""
        mlp = MLP(
            input_dim=10,
            hidden_dim=20,
            output_dim=2,
            num_layers=2,
            use_dropout=True,
            dropout_prob=0.5,
        )

        x = torch.randn(32, 10)

        # Test in training mode (dropout active)
        mlp.train()
        output_train = mlp(x)

        assert output_train.shape == (32, 2)
        assert not torch.isnan(output_train).any()

        # Test in evaluation mode (dropout inactive)
        mlp.eval()
        output_eval = mlp(x)

        assert output_eval.shape == (32, 2)
        assert not torch.isnan(output_eval).any()

        # Outputs should be different due to dropout
        # (though this might occasionally fail due to randomness)
        # Note: We can't reliably test that outputs are different due to randomness

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        x = torch.randn(10, 5, requires_grad=True)
        output = mlp(x)

        # Create dummy loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that model parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3, use_dropout=False)

        # Set to evaluation mode to ensure deterministic behavior
        mlp.eval()

        x = torch.randn(10, 5)

        # Multiple forward passes should give same result
        output1 = mlp(x)
        output2 = mlp(x)

        torch.testing.assert_close(output1, output2)

    def test_device_compatibility(self):
        """Test that MLP works correctly on different devices."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        # Test on CPU
        x_cpu = torch.randn(10, 5)
        output_cpu = mlp(x_cpu)

        assert output_cpu.device == torch.device("cpu")
        assert output_cpu.shape == (10, 3)

        # Test GPU if available
        if torch.cuda.is_available():
            mlp_gpu = mlp.cuda()
            x_gpu = x_cpu.cuda()

            output_gpu = mlp_gpu(x_gpu)

            assert output_gpu.device.type == "cuda"
            assert output_gpu.shape == (10, 3)


class TestMLPParameterHandling:
    """Test MLP parameter handling and properties."""

    def test_parameter_count(self):
        """Test parameter counting for different architectures."""
        # Simple case: 2 layers, no extras
        mlp = MLP(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_layers=2,
            use_batchnorm=False,
            use_dropout=False,
        )

        total_params = sum(p.numel() for p in mlp.parameters())

        expected_params = (10 * 20 + 20) + (20 * 20 + 20) + (20 * 5 + 5)
        assert total_params == expected_params

    def test_parameter_count_with_batchnorm(self):
        """Test parameter count with batch normalization."""
        mlp = MLP(
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_layers=1,
            use_batchnorm=True,
            use_dropout=False,
        )

        total_params = sum(p.numel() for p in mlp.parameters())

        expected_params = (10 * 20 + 20) + (20 * 2) + (20 * 5 + 5)
        assert total_params == expected_params

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        for param in mlp.parameters():
            # Parameters should not be all zeros
            assert not torch.allclose(param, torch.zeros_like(param))

            # Parameters should be finite
            assert torch.isfinite(param).all()

            # Parameters should have reasonable range
            assert param.abs().max() < 10.0  # Reasonable initialization range

    def test_parameter_sharing(self):
        """Test that parameters are not shared between different instances."""
        mlp1 = MLP(input_dim=5, hidden_dim=10, output_dim=3)
        mlp2 = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        # Parameters should be different between instances
        for p1, p2 in zip(mlp1.parameters(), mlp2.parameters(), strict=False):
            assert not torch.allclose(p1, p2)

    def test_trainable_parameters(self):
        """Test that all parameters are trainable by default."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3, use_batchnorm=True)

        for param in mlp.parameters():
            assert param.requires_grad


class TestMLPEdgeCases:
    """Test MLP edge cases and error conditions."""

    def test_zero_input(self):
        """Test forward pass with zero input."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        x = torch.zeros(10, 5)
        output = mlp(x)

        assert output.shape == (10, 3)
        assert torch.isfinite(output).all()
        # Output should not be all zeros due to bias terms
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_large_input_values(self):
        """Test forward pass with large input values."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        x = torch.full((10, 5), 100.0)  # Large values
        output = mlp(x)

        assert output.shape == (10, 3)
        assert torch.isfinite(output).all()

    def test_minimal_architecture(self):
        """Test with minimal architecture (1x1x1)."""
        mlp = MLP(input_dim=1, hidden_dim=1, output_dim=1, num_layers=1)

        x = torch.randn(5, 1)
        output = mlp(x)

        assert output.shape == (5, 1)
        assert torch.isfinite(output).all()

    def test_empty_batch(self):
        """Test forward pass with empty batch."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        x = torch.randn(0, 5)  # Empty batch
        output = mlp(x)

        assert output.shape == (0, 3)

    def test_wrong_input_dimensions(self):
        """Test handling of wrong input dimensions."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        # Wrong number of features
        x_wrong = torch.randn(10, 3)  # Expected 5 features, got 3

        with pytest.raises(RuntimeError):
            mlp(x_wrong)

        # Wrong number of dimensions
        x_1d = torch.randn(10)  # 1D tensor instead of 2D

        with pytest.raises((RuntimeError, IndexError)):
            mlp(x_1d)


class TestMLPIntegration:
    """Integration tests for MLP."""

    def test_training_integration(self):
        """Test MLP in a simple training scenario."""
        # Create simple regression problem
        mlp = MLP(input_dim=2, hidden_dim=10, output_dim=1)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Generate training data
        x = torch.randn(100, 2)
        y = (x[:, 0] + x[:, 1]).unsqueeze(1)  # Simple sum

        initial_loss = None
        final_loss = None

        # Training loop
        for _ in range(50):
            optimizer.zero_grad()

            output = mlp(x)
            loss = criterion(output, y)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease with training
        assert final_loss < initial_loss
        assert final_loss < 1.0  # Should learn this simple function

    def test_overfitting_small_dataset(self):
        """Test that MLP can overfit a small dataset."""
        mlp = MLP(input_dim=2, hidden_dim=50, output_dim=1)  # Large network
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Very small dataset
        x = torch.randn(5, 2)
        y = torch.randn(5, 1)

        # Train for many epochs
        for _ in range(500):
            optimizer.zero_grad()

            output = mlp(x)
            loss = criterion(output, y)

            loss.backward()
            optimizer.step()

        # Should be able to overfit small dataset
        final_output = mlp(x)
        final_loss = criterion(final_output, y)

        assert final_loss < 0.01  # Should overfit very well

    def test_different_optimizers(self):
        """Test MLP with different optimizers."""
        mlp = MLP(input_dim=5, hidden_dim=10, output_dim=3)

        # Test different optimizers
        optimizers = [
            torch.optim.SGD(mlp.parameters(), lr=0.01),
            torch.optim.Adam(mlp.parameters(), lr=0.001),
            torch.optim.RMSprop(mlp.parameters(), lr=0.001),
        ]

        x = torch.randn(32, 5)
        y = torch.randn(32, 3)
        criterion = nn.MSELoss()

        for optimizer in optimizers:
            # Reset parameters
            for param in mlp.parameters():
                param.data.normal_()

            # Train for a few steps
            for _ in range(10):
                optimizer.zero_grad()
                output = mlp(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            # Should not have NaN or infinite values
            for param in mlp.parameters():
                assert torch.isfinite(param).all()

    @pytest.mark.slow
    def test_large_network_performance(self):
        """Test MLP performance with large network."""
        # Create large network
        mlp = MLP(
            input_dim=1000,
            hidden_dim=500,
            output_dim=100,
            num_layers=5,
            use_batchnorm=True,
            use_dropout=True,
        )

        # Large batch
        x = torch.randn(256, 1000)

        # Forward pass should complete without errors
        output = mlp(x)

        assert output.shape == (256, 100)
        assert torch.isfinite(output).all()

        # Backward pass should work
        loss = output.sum()
        loss.backward()

        # All parameters should have gradients
        for param in mlp.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
