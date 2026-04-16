"""
Direction-change labeling for Buy / Sell / Hold signals.
"""

import numpy as np
import pandas as pd


def direction_change_labels(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> pd.Series:
    """
    Generate Hold / Long / Short labels based on direction changes.

    Label encoding (cross-entropy compatible):
      0 = Hold  (default)
      1 = Long  if next-day direction is up AND prior direction was flat/down
      2 = Short if next-day direction is down AND prior direction was flat/up

    Args:
        df: DataFrame with at least a price column indexed by Date.
        price_col: column name to use as price series.

    Returns:
        pd.Series of integer labels {0, 1, 2} aligned with a cleaned index
        (first and last rows are dropped due to shift boundary conditions).
    """
    close = df[price_col]

    fwd_dir = np.sign(close.shift(-1) - close)   # direction T -> T+1
    prev_dir = np.sign(close - close.shift(1))    # direction T-1 -> T

    labels = pd.Series(0, index=close.index, dtype=int)
    labels[(fwd_dir > 0) & (prev_dir <= 0)] = 1   # Long
    labels[(fwd_dir < 0) & (prev_dir >= 0)] = 2   # Short

    # drop first row (no prev_dir) and last row (no fwd_dir)
    labels = labels.iloc[1:-1]

    return labels
