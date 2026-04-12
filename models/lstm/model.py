"""
LSTM model for Buy/Sell/Hold classification.
"""

import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Stacked LSTM followed by a linear classification head.

    Input shape: (batch, seq_len, num_features)
    Output shape: (batch, num_classes)

    Args:
        input_size: number of features per timestep.
        hidden_size: LSTM hidden state dimension.
        num_layers: number of stacked LSTM layers.
        num_classes: number of output classes (default 3).
        dropout: dropout between LSTM layers (only applied when num_layers > 1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        raise NotImplementedError("LSTMModel.__init__ is not yet implemented.")

    def forward(self, x):
        raise NotImplementedError("LSTMModel.forward is not yet implemented.")
