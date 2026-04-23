"""
LSTM model for Buy/Sell/Hold classification.
"""

import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers > 1:
            dropout_val = dropout
        else:
            dropout_val = 0.0


        self.lstm = nn.LSTM(
            input_size = input_size, hidden_size=hidden_size,
            num_layers = num_layers, batch_first = True,
            dropout = dropout_val
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        # take the final hidden state from the top LSTM layer
        # shape: (batch, hidden_size)
        final_hidden = h_n[-1]

        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)

        return logits

