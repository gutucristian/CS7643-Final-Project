"""
Portfolio backtesting simulator.

Supports two modes:
  long_only  — trades between cash and long; no shorting.
  long_short — always in the market, flips between long and short.

Benchmark: buy-and-hold (invest all capital on day 0, sell on final day).
"""

import numpy as np


class Backtester:
    """
    Simulate a trading strategy from model predictions and compute metrics.

    Args:
        prices: array-like of daily close prices aligned with predictions.
        initial_capital: starting portfolio value in USD.
        mode: 'long_only' or 'long_short'.
    """

    def __init__(self, prices, initial_capital: float = 10_000.0, mode: str = "long_only"):
        self.prices = np.asarray(prices, dtype=float)
        self.initial_capital = float(initial_capital)
        assert mode in ("long_only", "long_short"), f"Unknown mode: {mode}"
        self.mode = mode

    def run(self, signals) -> dict:
        """
        Execute the strategy using predicted signals.

        Signal convention: 1=Long, 0=Hold, -1=Short.
        Signal from day i is executed at day i+1 (no lookahead).

        Returns:
            Dict with keys: 'portfolio_values' (list), 'final_value' (float).
        """
        signals = np.asarray(signals, dtype=int)
        prices = self.prices
        n = min(len(signals), len(prices) - 1)
        prices = prices[: n + 1]

        if self.mode == "long_only":
            portfolio_values = self._run_long_only(signals[:n], prices)
        else:
            portfolio_values = self._run_long_short(signals[:n], prices)

        benchmark_values = self.benchmark_values(prices)

        return {
            "portfolio_values": portfolio_values,
            "final_value": portfolio_values[-1],
            "benchmark_values": benchmark_values,
            "benchmark_final_value": benchmark_values[-1],
        }

    def _run_long_only(self, signals, prices):
        cash = self.initial_capital
        shares = 0.0
        values = []

        for i, sig in enumerate(signals):
            price_today = prices[i]
            price_next = prices[i + 1]

            if sig == 1 and shares == 0:        # Long signal, currently flat → buy
                shares = cash / price_today
                cash = 0.0
            elif sig == -1 and shares > 0:      # Short signal, currently long → liquidate
                cash = shares * price_today
                shares = 0.0
            # Hold → do nothing

            portfolio_value = cash + shares * price_next
            values.append(portfolio_value)

        return values

    def _run_long_short(self, signals, prices):
        equity = self.initial_capital
        position = 0  # -1 = short, 0 = flat, +1 = long
        values = []

        for i, sig in enumerate(signals):
            price_today = prices[i]
            price_next = prices[i + 1]

            if sig == 1:
                position = 1
            elif sig == -1:
                position = -1

            asset_return = (price_next / price_today) - 1.0
            equity *= 1.0 + position * asset_return
            values.append(equity)

        return values

    def benchmark_values(self, prices=None):
        """
        Compute the buy-and-hold curve for the specified price window.
        """
        if prices is None:
            prices = self.prices

        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return np.asarray([self.initial_capital], dtype=float)
        return self.initial_capital * (prices[1:] / prices[0])

    def metrics(self, portfolio_values, benchmark_values=None) -> dict:
        """
        Compute financial performance metrics.

        Returns:
            Dict with keys: 'total_return', 'sharpe_ratio', 'max_drawdown',
            'benchmark_max_drawdown', 'win_rate', 'profit_factor',
            'benchmark_return', 'benchmark_final_value'.
        """
        values = np.asarray(portfolio_values, dtype=float)

        total_return = (values[-1] - self.initial_capital) / self.initial_capital

        daily_returns = np.diff(values) / values[:-1]
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        max_drawdown = drawdowns.min()

        if benchmark_values is None:
            benchmark_values = self.benchmark_values()
        bh_values = np.asarray(benchmark_values, dtype=float)
        bh_peak = np.maximum.accumulate(bh_values)
        benchmark_max_drawdown = ((bh_values - bh_peak) / bh_peak).min()

        winning_days = daily_returns[daily_returns > 0]
        losing_days = daily_returns[daily_returns < 0]
        win_rate = len(winning_days) / len(daily_returns) if len(daily_returns) > 0 else 0.0
        profit_factor = (
            winning_days.sum() / (-losing_days.sum())
            if losing_days.sum() != 0 else float("inf")
        )

        benchmark_return = (bh_values[-1] - self.initial_capital) / self.initial_capital

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "benchmark_max_drawdown": benchmark_max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "benchmark_return": benchmark_return,
            "benchmark_final_value": bh_values[-1],
        }
