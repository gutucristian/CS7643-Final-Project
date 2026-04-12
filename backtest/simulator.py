"""
Portfolio backtesting simulator.

Uses an all-in strategy:
  - Buy signal  → invest all available cash into SPY
  - Sell signal → liquidate all holdings to cash
  - Hold signal → do nothing

Benchmark: buy-and-hold (invest all capital on day 0, sell on final day).
"""


class Backtester:
    """
    Simulate a trading strategy from model predictions and compute metrics.

    Args:
        prices: array-like of daily close prices aligned with predictions.
        initial_capital: starting portfolio value in USD.
    """

    def __init__(self, prices, initial_capital: float = 10_000.0):
        raise NotImplementedError("Backtester.__init__ is not yet implemented.")

    def run(self, signals) -> dict:
        """
        Execute the all-in strategy using predicted signals.

        Args:
            signals: array-like of {-1=Sell, 0=Hold, 1=Buy} per timestep.

        Returns:
            Dict with keys: 'portfolio_values' (list), 'final_value' (float).
        """
        raise NotImplementedError("Backtester.run is not yet implemented.")

    def metrics(self, portfolio_values) -> dict:
        """
        Compute financial performance metrics.

        Returns:
            Dict with keys: 'sharpe_ratio', 'max_drawdown', 'total_return',
            'win_rate', 'profit_factor', 'benchmark_return'.
        """
        raise NotImplementedError("Backtester.metrics is not yet implemented.")
