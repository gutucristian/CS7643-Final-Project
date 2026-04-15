"""
Loss functions for training.

Both losses share a unified forward signature:
    loss = criterion(logits, labels, fwd_returns)

This keeps the Trainer loss-agnostic — switching between cross-entropy and
Sharpe is a one-line change in the training script.

Approach 1 — cross_entropy_loss: standard categorical cross-entropy.
Approach 2 — sharpe_loss: differentiable negative Sharpe ratio that maps
    model output probabilities to a continuous position in [-1, 1] and
    optimises directly for risk-adjusted returns (ignores labels).
"""

import torch
import torch.nn as nn


class _CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, fwd_returns: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, labels)


class _SharpeLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, fwd_returns: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)  # (B, 3): [Hold, Long, Short]
        # position: long → +1, short → -1, hold → 0
        position = probs[:, 1] - probs[:, 2]   # in (-1, 1)
        strategy_returns = position * fwd_returns
        mean = strategy_returns.mean()
        std = strategy_returns.std() + self.eps
        return -mean / std  # minimise negative Sharpe


def cross_entropy_loss() -> nn.Module:
    """Return a CrossEntropyLoss with the unified (logits, labels, returns) signature."""
    return _CrossEntropyLoss()


def sharpe_loss() -> nn.Module:
    """Return a Sharpe-ratio loss with the unified (logits, labels, returns) signature."""
    return _SharpeLoss()
