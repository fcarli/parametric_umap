import torch
from torch import nn


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) with flexible architecture.

    A flexible implementation of a Multi-Layer Perceptron that supports:
    - Variable number of hidden layers
    - Optional batch normalization
    - Optional dropout
    - Configurable hidden dimensions

    Parameters
    ----------
    input_dim : int
        Dimension of the input features
    hidden_dim : int
        Dimension of the hidden layers (embedding dimension)
    output_dim : int
        Dimension of the output layer
    num_layers : int, optional (default=2)
        Number of hidden layers
    use_batchnorm : bool, optional (default=False)
        If True, includes Batch Normalization after each linear layer
    use_dropout : bool, optional (default=False)
        If True, includes Dropout after each activation function
    dropout_prob : float, optional (default=0.5)
        Probability of an element to be zeroed in dropout layers

    Attributes
    ----------
    model : nn.Sequential
        The sequential container of all layers in the network

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
    ) -> None:
        super(MLP, self).__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim

        for i in range(num_layers):
            # Linear layer
            layers.append(nn.Linear(in_dim, hidden_dim))

            # Batch Normalization (optional)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation function
            layers.append(nn.ReLU())

            # Dropout (optional)
            if use_dropout:
                layers.append(nn.Dropout(dropout_prob))

            # Update input dimension for next layer
            in_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))

        # Optionally, you can add an activation function for the output
        # For example, use Sigmoid for binary classification or Softmax for multi-class
        # layers.append(nn.Softmax(dim=1))

        # Combine all layers into a Sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)

        """
        return self.model(x)
