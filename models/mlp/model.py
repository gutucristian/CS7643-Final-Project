import torch
import torch.nn as nn


class MLP(nn.Module):
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
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)
