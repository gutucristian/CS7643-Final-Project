import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    # PyTorch dataset for sliding-window SPY features and labels

    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        window_size: int = 20,
        feature_cols: list = None,
        price_col: str = "Close",
        fwd_returns: pd.Series = None,
    ):
        # align df, labels, and optional returns to a common index
        common_idx = df.index.intersection(labels.index)
        if fwd_returns is not None:
            common_idx = common_idx.intersection(fwd_returns.index)

        df = df.loc[common_idx]
        labels = labels.loc[common_idx]
        if fwd_returns is not None:
            fwd_returns = fwd_returns.loc[common_idx]

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.window_size = window_size

        if fwd_returns is None:
            if price_col not in df.columns:
                raise ValueError(
                    f"price_col '{price_col}' not found and no precomputed fwd_returns were provided."
                )

            # forward return: (close[t+1] / close[t]) - 1, last row gets 0
            close = df[price_col]
            fwd_returns = (close.shift(-1) / close - 1).fillna(0)
        else:
            fwd_returns = fwd_returns.astype(np.float32).fillna(0)

        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = labels.values.astype(np.int64)
        self.fwd_returns = fwd_returns.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        window = self.features[idx : idx + self.window_size]
        label = self.labels[idx + self.window_size - 1]
        fwd_return = self.fwd_returns[idx + self.window_size - 1]

        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(fwd_return, dtype=torch.float32),
        )
