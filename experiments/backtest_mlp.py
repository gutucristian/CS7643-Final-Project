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
from sklearn.metrics import classification_report, confusion_matrix

from backtest.simulator import Backtester
from data.data_utils import load_ohlcv_csv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp.yaml")
    args = parser.parse_args()

    run_tag = os.path.splitext(os.path.basename(args.config))[0]
    default_predictions = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mlp", f"{run_tag}_test_predictions.csv"
    )

    cfg = load_config(args.config)
    dc = cfg["data"]
    bc = cfg["backtest"]

    # --------------------------------------------------------- load predictions
    args.predictions = default_predictions

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

    # --------------------------------------------------------- classification metrics
    label_names = {0: "Hold", 1: "Long", 2: "Short"}
    target_names = [label_names[i] for i in sorted(label_names)]

    print("\nPrediction distribution:")
    unique, counts = np.unique(raw_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")

    print()
    print(classification_report(true_labels, raw_preds, target_names=target_names, digits=3))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, raw_preds, labels=[0, 1, 2])
    header = f"{'':>8}" + "".join(f"{n:>8}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>8}" + "".join(f"{v:>8}" for v in row))

    # --------------------------------------------------------- backtest
    print()
    for mode in bc["modes"]:
        bt = Backtester(backtest_prices, initial_capital=bc["initial_capital"], mode=mode)
        result = bt.run(signals)
        m = bt.metrics(result["portfolio_values"])
        print(f"--- Backtest: {mode} ---")
        print(f"  Total return:          {m['total_return']*100:+.2f}%")
        print(f"  Benchmark return:      {m['benchmark_return']*100:+.2f}%")
        print(f"  Sharpe ratio:          {m['sharpe_ratio']:.3f}")
        print(f"  Max drawdown:          {m['max_drawdown']*100:.2f}%")
        print(f"  Benchmark max drawdown:{m['benchmark_max_drawdown']*100:.2f}%")
        print(f"  Win rate:              {m['win_rate']*100:.1f}%")
        print(f"  Profit factor:         {m['profit_factor']:.3f}")
        print()


if __name__ == "__main__":
    main()
