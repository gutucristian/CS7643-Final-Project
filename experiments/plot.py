"""
Plotting utilities for backtest results.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_backtest(
    dates,
    prices,
    signals,
    portfolio_values,
    initial_capital: float,
    mode: str,
    run_tag: str,
    save_path: str,
):
    """
    Two-panel plot:
      Top    — price with buy (^) and sell (v) signal markers
      Bottom — strategy portfolio value vs buy-and-hold benchmark
    """
    dates = list(dates)
    prices = np.asarray(prices, dtype=float)
    signals = np.asarray(signals, dtype=int)
    portfolio_values = np.asarray(portfolio_values, dtype=float)

    # prices and signals are aligned: signal[i] is executed at prices[i+1]
    # use prices[:-1] (the signal day prices) for marker positions
    signal_prices = prices[: len(signals)]

    buy_dates  = [dates[i] for i in range(len(signals)) if signals[i] == 1]
    buy_prices = [signal_prices[i] for i in range(len(signals)) if signals[i] == 1]
    sell_dates  = [dates[i] for i in range(len(signals)) if signals[i] == -1]
    sell_prices = [signal_prices[i] for i in range(len(signals)) if signals[i] == -1]

    # benchmark: buy-and-hold, normalised to initial_capital
    benchmark = initial_capital * (prices[1: len(portfolio_values) + 1] / prices[0])
    port_dates = dates[: len(portfolio_values)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{run_tag}  —  {mode}", fontsize=13, fontweight="bold")

    # ── top panel: price + signals ──────────────────────────────────────────
    ax1.plot(dates[: len(signals)], signal_prices, color="#555555", linewidth=0.8, label="Price")
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, marker="^", color="#16a34a", s=40, zorder=5, label="Buy (Long)")
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, marker="v", color="#dc2626", s=40, zorder=5, label="Sell (Short)")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── bottom panel: portfolio vs benchmark ────────────────────────────────
    ax2.plot(port_dates, portfolio_values, color="#2563eb", linewidth=1.2, label="Strategy")
    ax2.plot(port_dates, benchmark,        color="#f59e0b", linewidth=1.0, linestyle="--", label="Buy & Hold")
    ax2.axhline(initial_capital, color="#9ca3af", linewidth=0.7, linestyle=":")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {save_path}")
