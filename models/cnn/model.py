import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_channels: list = [64, 128],
        kernel_size: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels

        layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            layers += [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"input shape mismatch, got {x.shape}")
        if x.size(1) != self.input_channels and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)
        elif x.size(1) != self.input_channels:
            raise ValueError(f"Expected input with {self.input_channels} feature channels, but got shape {x.shape}")

        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
