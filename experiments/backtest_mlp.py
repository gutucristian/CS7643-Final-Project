"""
Backtest script for MLP predictions.

Loads saved test predictions and runs the backtester.

Usage:
    python experiments/backtest_mlp.py --config configs/mlp.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from backtest.simulator import Backtester
from data.data_utils import load_ohlcv_csv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp.yaml")
    parser.add_argument(
        "--predictions",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlp", "test_predictions.csv"),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    bc = cfg["backtest"]

    # --------------------------------------------------------- load predictions
    if not os.path.exists(args.predictions):
        print(f"Predictions file not found: {args.predictions}")
        print("Run train_mlp.py first to generate predictions.")
        sys.exit(1)

    preds_df = pd.read_csv(args.predictions, index_col=0, parse_dates=True)
    print(f"Loaded predictions: {len(preds_df)} samples from {args.predictions}")

    raw_preds = preds_df["pred_label"].values
    true_labels = preds_df["true_label"].values
    pred_dates = preds_df.index

    # --------------------------------------------------------- load prices
    df = load_ohlcv_csv(dc["csv_path"])
    price_col = dc["price_col"]

    # align prices to prediction dates, then extend one day forward for returns
    all_dates = df.index
    start_loc = all_dates.get_loc(pred_dates[0])
    end_loc = min(start_loc + len(pred_dates) + 1, len(all_dates))
    backtest_prices = df[price_col].iloc[start_loc:end_loc].values

    # --------------------------------------------------------- signal mapping
    signal_map = {0: 0, 1: 1, 2: -1}
    signals = np.array([signal_map[p] for p in raw_preds])

    # --------------------------------------------------------- label summary
    label_names = {0: "Hold", 1: "Long", 2: "Short"}
    print("\nPrediction distribution:")
    unique, counts = np.unique(raw_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")

    correct = (raw_preds == true_labels).sum()
    print(f"\nTest accuracy: {correct}/{len(true_labels)} ({correct/len(true_labels)*100:.2f}%)")

    # --------------------------------------------------------- backtest
    print()
    for mode in bc["modes"]:
        bt = Backtester(backtest_prices, initial_capital=bc["initial_capital"], mode=mode)
        result = bt.run(signals)
        m = bt.metrics(result["portfolio_values"])
        print(f"--- Backtest: {mode} ---")
        print(f"  Total return:     {m['total_return']*100:+.2f}%")
        print(f"  Benchmark return: {m['benchmark_return']*100:+.2f}%")
        print(f"  Sharpe ratio:     {m['sharpe_ratio']:.3f}")
        print(f"  Max drawdown:     {m['max_drawdown']*100:.2f}%")
        print(f"  Win rate:         {m['win_rate']*100:.1f}%")
        print(f"  Profit factor:    {m['profit_factor']:.3f}")
        print()


if __name__ == "__main__":
    main()
