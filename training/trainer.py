"""
Generic training loop for all three model architectures.
"""

import torch
from torch.utils.data import DataLoader


class Trainer:
    """
    Manages training and evaluation for a PyTorch model.

    Args:
        model: nn.Module to train.
        optimizer: torch Optimizer instance.
        criterion: loss function (callable).
        device: torch device string, e.g. 'cpu' or 'cuda'.
    """

    def __init__(self, model, optimizer, criterion, device: str = "cpu"):
        raise NotImplementedError("Trainer.__init__ is not yet implemented.")

    def train(self, dataloader: DataLoader, epochs: int = 10) -> list:
        """
        Run the training loop.

        Returns:
            List of per-epoch average training losses.
        """
        raise NotImplementedError("Trainer.train is not yet implemented.")

    def evaluate(self, dataloader: DataLoader) -> dict:
        """
        Evaluate the model on a dataloader.

        Returns:
            Dict with keys: 'loss', 'accuracy'.
        """
        raise NotImplementedError("Trainer.evaluate is not yet implemented.")
