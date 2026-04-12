"""
1D Convolutional Neural Network for Buy/Sell/Hold classification.
"""

import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D CNN that treats the feature window as a sequence of channels.

    Input shape: (batch, num_features, seq_len)
    Output shape: (batch, num_classes)

    Args:
        input_channels: number of input feature channels (num_features).
        num_classes: number of output classes (default 3).
        dropout: dropout probability before the classifier head.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        raise NotImplementedError("CNN1D.__init__ is not yet implemented.")

    def forward(self, x):
        raise NotImplementedError("CNN1D.forward is not yet implemented.")
