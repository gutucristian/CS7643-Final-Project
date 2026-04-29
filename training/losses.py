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
import torch.nn.functional as F


class _CrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, fwd_returns: torch.Tensor) -> torch.Tensor:
        return self._ce(logits, labels)


class _FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, fwd_returns: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        ce = F.nll_loss(
            log_probs,
            labels,
            weight=self.alpha,
            reduction="none",
        )
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


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


def cross_entropy_loss(
    class_counts: list = None,
    num_classes: int = 3,
    class_weight_multipliers: list = None,
) -> nn.Module:
    """Return a CrossEntropyLoss with the unified (logits, labels, returns) signature.

    Args:
        class_counts: list of per-class sample counts. If provided, weights are
                      set to 1/count (normalized), so rare classes get higher weight.
        num_classes: total number of classes (used only when class_counts is None).
        class_weight_multipliers: optional per-class multipliers applied after the
                                  inverse-frequency weights, e.g. [1.0, 1.0, 1.25].
    """
    weight = None
    if class_counts is not None:
        counts = torch.tensor(class_counts, dtype=torch.float)
        weight = 1.0 / counts
        if class_weight_multipliers is not None:
            multipliers = torch.tensor(class_weight_multipliers, dtype=torch.float)
            if len(multipliers) != num_classes:
                raise ValueError(
                    f"class_weight_multipliers must have length {num_classes}, "
                    f"got {len(multipliers)}"
                )
            weight = weight * multipliers
        weight = weight / weight.sum() * num_classes   # normalize so avg weight == 1
    return _CrossEntropyLoss(weight=weight)


def focal_loss(
    num_classes: int = 3,
    gamma: float = 2.0,
    alpha: list = None,
) -> nn.Module:
    """Return a focal loss with the unified (logits, labels, returns) signature.

    Args:
        num_classes: total number of classes.
        gamma: focusing parameter; larger values emphasize hard examples more.
        alpha: optional per-class weights, e.g. [1.0, 1.0, 1.25].
    """
    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        if len(alpha_tensor) != num_classes:
            raise ValueError(
                f"alpha must have length {num_classes}, got {len(alpha_tensor)}"
            )
        alpha_tensor = alpha_tensor / alpha_tensor.sum() * num_classes

    return _FocalLoss(gamma=gamma, alpha=alpha_tensor)


def sharpe_loss() -> nn.Module:
    """Return a Sharpe-ratio loss with the unified (logits, labels, returns) signature."""
    return _SharpeLoss()
