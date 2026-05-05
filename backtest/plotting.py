from pathlib import Path

import numpy as np
import pandas as pd

def _load_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    return plt, mdates, FuncFormatter

def save_equity_curve_csv(path, dates,strategy_values,benchmark_values,):
    curve_df = pd.DataFrame(
        {
            "strategy_value": strategy_values,
            "benchmark_value": benchmark_values
        },
        index=pd.to_datetime(dates),
    )
    curve_df.index.name = "Date"

    path = Path(path)
    curve_df.to_csv(path)
    return path

def plot_equity_curves(path,dates,strategy_values,benchmark_values,*,title,
    strategy_label="Model Strategy",
    benchmark_label="Buy & Hold SPY",
    initial_capital=None,
):
    dates = pd.to_datetime(dates).to_numpy()
    strategy_values = np.asarray(strategy_values, dtype=float)
    benchmark_values = np.asarray(benchmark_values, dtype=float)

    path = Path(path)
    plt, mdates, FuncFormatter = _load_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(dates, strategy_values, label=strategy_label, color="#0b6e4f", linewidth=2.4)
    ax.plot(dates, benchmark_values, label=benchmark_label, color="#1f77b4", linewidth=2.2)

    if initial_capital is not None:
        ax.axhline(
            initial_capital,
            color="#6b7280",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Initial Capital",
        )

    strategy_final = strategy_values[-1]
    benchmark_final = benchmark_values[-1]
    summary = (
        f"{strategy_label}: ${strategy_final:,.0f}\n"
        f"{benchmark_label}: ${benchmark_final:,.0f}"
    )
    ax.text(0.98,0.98,summary,transform=ax.transAxes,va="top",ha="right",
        fontsize=10, bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"})

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    y_min = min(strategy_values.min(), benchmark_values.min())
    y_max = max(strategy_values.max(), benchmark_values.max())
    y_range = max(y_max - y_min, 1.0)
    ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.12 * y_range)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
