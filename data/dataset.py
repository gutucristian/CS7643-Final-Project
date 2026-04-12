"""
PyTorch Dataset for SPY OHLCV + indicator data with sliding-window sequences.
"""

import torch
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    """
    Sliding-window dataset over the enriched SPY indicator CSV.

    Each sample is a (window_size, num_features) tensor of indicator values
    paired with a scalar label {0=Hold, 1=Buy, 2=Sell}.

    Args:
        df: DataFrame of features (rows = timesteps, columns = features).
        labels: Series or array of integer labels aligned with df.
        window_size: number of look-back timesteps per sample.
    """

    def __init__(self, df, labels, window_size: int = 20):
        raise NotImplementedError("SPYDataset.__init__ is not yet implemented.")

    def __len__(self) -> int:
        raise NotImplementedError("SPYDataset.__len__ is not yet implemented.")

    def __getitem__(self, idx):
        raise NotImplementedError("SPYDataset.__getitem__ is not yet implemented.")
