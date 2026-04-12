"""
Multi-Layer Perceptron (MLP) for Buy/Sell/Hold classification.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected MLP that takes a flattened feature window as input
    and outputs logits over 3 classes (Hold, Buy, Sell).

    Args:
        input_size: total number of input features (window_size * num_features).
        hidden_sizes: list of hidden layer widths.
        num_classes: number of output classes (default 3).
        dropout: dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list = [256, 128, 64],
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        raise NotImplementedError("MLP.__init__ is not yet implemented.")

    def forward(self, x):
        raise NotImplementedError("MLP.forward is not yet implemented.")
