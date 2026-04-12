"""
Triple Barrier Method labeling for Buy / Sell / Hold signals.
"""

import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    upper_barrier: float = 0.02,
    lower_barrier: float = -0.02,
    max_holding_period: int = 20,
    price_col: str = "Close",
) -> pd.Series:
    """
    Generate Buy / Sell / Hold labels using the Triple Barrier Method.

    For each timestep, look forward up to `max_holding_period` days:
      - Label = 1 (Buy)  if the upper barrier is hit first
      - Label = -1 (Sell) if the lower barrier is hit first
      - Label = 0 (Hold) if the time barrier is reached without hitting either

    Args:
        df: DataFrame with at least a price column indexed by Date.
        upper_barrier: fractional profit-taking threshold (e.g. 0.02 = +2%).
        lower_barrier: fractional stop-loss threshold (e.g. -0.02 = -2%).
        max_holding_period: maximum look-forward window in trading days.
        price_col: column name to use as price series.

    Returns:
        pd.Series of integer labels {-1, 0, 1} aligned with df.index.
    """
    raise NotImplementedError("triple_barrier_labels is not yet implemented.")
