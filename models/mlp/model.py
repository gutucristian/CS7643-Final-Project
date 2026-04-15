"""
Multi-Layer Perceptron (MLP) for Hold / Long / Short classification.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Fully-connected MLP with BatchNorm and Dropout after each hidden layer.

    Architecture (default hidden_sizes=[256, 128, 64]):
        Linear(input -> 256) -> BN -> ReLU -> Dropout
        Linear(256 -> 128)   -> BN -> ReLU -> Dropout
        Linear(128 -> 64)    -> BN -> ReLU -> Dropout
        Linear(64 -> num_classes)

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

        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten any input with more than 2 dims: (B, W, F) -> (B, W*F)
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
