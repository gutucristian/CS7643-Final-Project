# LSTM model definition/architecture for Buy/Sell/Hold classification


import torch
import torch.nn as nn



class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size = 128,
        num_layers = 2,
        num_classes = 3,
        dropout = 0.3,
        pooling = "mean",
        head_hidden_size = None,
    ):
        super().__init__()

        # hidden state
        self.hidden_size = hidden_size

        # number of layers
        self.num_layers = num_layers

        # pooling
        self.pooling = pooling

        # configure dropout
        if num_layers > 1:
            dropout_val = dropout
        else:
            dropout_val = 0.0


        self.lstm = nn.LSTM(
            input_size = input_size, 
            hidden_size = hidden_size,
            num_layers = num_layers, 
            batch_first = True,
            dropout = dropout_val
        )

        self.dropout = nn.Dropout(dropout)
        head_hidden_size = head_hidden_size or hidden_size


        # add classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_classes),
        )

    def forward(self, x):
        # forward pass
        
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.pooling == "last":
            pooled = h_n[-1]

        elif self.pooling == "mean":
            pooled = lstm_out.mean(dim = 1)

        else:
            pooled = torch.max(lstm_out, dim = 1).values

        pooled = self.dropout(pooled)

        logits = self.head(pooled)

        return logits
    
