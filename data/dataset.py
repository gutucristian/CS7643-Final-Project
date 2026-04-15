"""
PyTorch Dataset for SPY OHLCV + indicator data with sliding-window sequences.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    """
    Sliding-window dataset over SPY feature data.

    Each sample is a (window_size, num_features) tensor paired with a scalar
    label {0=Hold, 1=Long, 2=Short} and the forward 1-day return for that day.

    Args:
        df: DataFrame of features (rows = timesteps, columns = features).
        labels: Series of integer labels aligned with df.
        window_size: number of look-back timesteps per sample.
        feature_cols: list of column names to use as features. If None, all
            numeric columns in df are used.
        price_col: column used to compute forward returns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        window_size: int = 20,
        feature_cols: list = None,
        price_col: str = "Close",
    ):
        # align df and labels to common index
        common_idx = df.index.intersection(labels.index)
        df = df.loc[common_idx]
        labels = labels.loc[common_idx]

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.window_size = window_size

        # forward return: (close[t+1] / close[t]) - 1, last row gets 0
        close = df[price_col]
        fwd_returns = (close.shift(-1) / close - 1).fillna(0)

        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = labels.values.astype(np.int64)
        self.fwd_returns = fwd_returns.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.features[idx : idx + self.window_size]          # (W, F)
        label = self.labels[idx + self.window_size - 1]
        fwd_return = self.fwd_returns[idx + self.window_size - 1]

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(fwd_return, dtype=torch.float32),
        )
