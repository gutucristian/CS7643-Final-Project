"""
Loss functions for training.

Approach 1 — cross_entropy_loss: standard categorical cross-entropy
    optimises for label accuracy (Triple Barrier labels as targets).

Approach 2 — sharpe_loss: differentiable negative Sharpe ratio
    maps model output probabilities to continuous position sizes and
    optimises directly for risk-adjusted returns (no labels needed).
"""

import torch
import torch.nn as nn


def cross_entropy_loss() -> nn.Module:
    """
    Return a standard CrossEntropyLoss instance.

    Usage:
        criterion = cross_entropy_loss()
        loss = criterion(logits, labels)   # labels in {0, 1, 2}
    """
    raise NotImplementedError("cross_entropy_loss is not yet implemented.")


def sharpe_loss(logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Differentiable negative Sharpe ratio loss.

    Maps softmax probabilities to a scalar position in [-1, 1], computes
    per-step strategy returns, then returns the negative Sharpe ratio.

    Args:
        logits: (batch, 3) raw model outputs [Hold, Buy, Sell].
        returns: (batch,) actual 1-day forward returns for each timestep.

    Returns:
        Scalar tensor — negative Sharpe ratio (minimise this).
    """
    raise NotImplementedError("sharpe_loss is not yet implemented.")
