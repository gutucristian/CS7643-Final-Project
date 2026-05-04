# Loss functions for training. Supports cross_entropy_loss, focal_loss, sharpe_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class _CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels, fwd_returns):
        return self._ce(logits, labels)

class _FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, labels, fwd_returns):
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
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, labels, fwd_returns):
        probs = torch.softmax(logits, dim=-1)
        # position: long -> +1, short -> -1, hold -> 0
        position = probs[:, 1] - probs[:, 2]
        strategy_returns = position * fwd_returns
        mean = strategy_returns.mean()
        std = strategy_returns.std() + self.eps
        return -mean / std  # minimise negative Sharpe


def cross_entropy_loss( class_counts=None, num_classes=3, class_weight_multipliers=None):
    # Return a CrossEntropyLoss with the unified (logits, labels, returns) signature.

    weight = None
    if class_counts is not None:
        counts = torch.tensor(class_counts, dtype=torch.float)
        weight = 1.0 / counts
        if class_weight_multipliers is not None:
            multipliers = torch.tensor(class_weight_multipliers, dtype=torch.float)
            if len(multipliers) != num_classes:
                raise ValueError(
                    f"class_weight_multipliers must have length {num_classes}, but got {len(multipliers)}")
            weight = weight * multipliers
        weight = weight / weight.sum() * num_classes # normalize
    return _CrossEntropyLoss(weight=weight)


def focal_loss(num_classes=3, gamma=2.0, alpha=None):
    # Return a focal loss with the unified (logits, labels, returns) signature.
    alpha_tensor = None
    if alpha is not None:
        alpha_tensor = torch.tensor(alpha, dtype=torch.float)
        alpha_tensor = alpha_tensor / alpha_tensor.sum() * num_classes
    return _FocalLoss(gamma=gamma, alpha=alpha_tensor)


def sharpe_loss():
    return _SharpeLoss()
