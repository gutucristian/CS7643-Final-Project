"""
Backtest script for CNN predictions.

Loads saved CNN test predictions, aligns them to the raw SPY price series.

Compares the strategy performance with a buy and hold SPY benchmark over the
same test window and starting capital.

Usage:
    python experiments/backtest_cnn.py --config configs/cnn.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from backtest.plotting import plot_equity_curves, save_equity_curve_csv
from backtest.simulator import Backtester
from data.data_utils import load_ohlcv_csv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(path: str) -> pd.DataFrame:
    preds_df = pd.read_csv(path, index_col=0, parse_dates=True)
    preds_df.index = pd.to_datetime(preds_df.index)
    return preds_df


def aligned_backtest_prices(df: pd.DataFrame, pred_dates: pd.DatetimeIndex, price_col: str) -> pd.Series:
    all_dates = df.index
    start_loc = all_dates.get_loc(pred_dates[0])
    end_loc = start_loc + len(pred_dates)

    if end_loc >= len(all_dates):
        raise ValueError("Not enough price data to extend one step beyond the final prediction date.")

    prices = df[price_col].iloc[start_loc : end_loc + 1].copy()
    expected_dates = pred_dates.append(pd.DatetimeIndex([all_dates[end_loc]]))
    if not prices.index.equals(expected_dates):
        raise ValueError("Price alignment mismatch between prediction dates and raw price data.")

    return prices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn.yaml")
    args = parser.parse_args()

    run_tag = os.path.splitext(os.path.basename(args.config))[0]
    default_predictions = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cnn", f"{run_tag}_test_predictions.csv"
    )

    cfg = load_config(args.config)
    dc = cfg["data"]
    bc = cfg["backtest"]

    predictions_path = default_predictions
    if not os.path.exists(predictions_path):
        print(f"Predictions file not found: {predictions_path}")
        print("Run train_cnn.py first to generate predictions.")
        sys.exit(1)

    preds_df = load_predictions(predictions_path)
    print(f"Loaded predictions: {len(preds_df)} samples from {predictions_path}")

    raw_preds = preds_df["pred_label"].values
    true_labels = preds_df["true_label"].values
    pred_dates = preds_df.index

    raw_df = load_ohlcv_csv(dc["raw_price_csv_path"])
    price_col = dc["price_col"]
    price_series = aligned_backtest_prices(raw_df, pred_dates, price_col=price_col)

    signal_map = {0: 0, 1: 1, 2: -1}
    signals = np.array([signal_map[p] for p in raw_preds], dtype=int)

    label_names = {0: "Hold", 1: "Long", 2: "Short"}
    target_names = [label_names[i] for i in sorted(label_names)]

    print("\nPrediction distribution:")
    unique, counts = np.unique(raw_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            raw_preds,
            labels=[0, 1, 2],
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, raw_preds, labels=[0, 1, 2])
    header = f"{'':>8}" + "".join(f"{n:>8}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{target_names[i]:>8}" + "".join(f"{v:>8}" for v in row))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn")
    os.makedirs(out_dir, exist_ok=True)

    print()
    for mode in bc["modes"]:
        bt = Backtester(price_series.values, initial_capital=bc["initial_capital"], mode=mode)
        result = bt.run(signals)
        metrics = bt.metrics(result["portfolio_values"], benchmark_values=result["benchmark_values"])

        curve_dates = price_series.index[1:]
        curve_path = os.path.join(out_dir, f"{run_tag}_{mode}_backtest_curve.csv")
        save_equity_curve_csv(
            curve_path,
            curve_dates,
            result["portfolio_values"],
            result["benchmark_values"],
        )
        plot_path = os.path.join(out_dir, f"{run_tag}_{mode}_backtest_plot.png")
        plot_saved = False
        try:
            plot_equity_curves(
                plot_path,
                curve_dates,
                result["portfolio_values"],
                result["benchmark_values"],
                title=f"CNN Backtest vs Buy-and-Hold ({mode})",
                strategy_label="CNN Strategy",
                benchmark_label="Buy & Hold SPY",
                initial_capital=bc["initial_capital"],
            )
            plot_saved = True
        except ImportError as exc:
            print(f"  Plot skipped:           {exc}")

        print(f"--- Backtest: {mode} ---")
        print(f"  Initial capital:       ${bc['initial_capital']:,.2f}")
        print(f"  Strategy final value:  ${result['final_value']:,.2f}")
        print(f"  Benchmark final value: ${result['benchmark_final_value']:,.2f}")
        print(f"  Total return:          {metrics['total_return']*100:+.2f}%")
        print(f"  Benchmark return:      {metrics['benchmark_return']*100:+.2f}%")
        print(f"  Sharpe ratio:          {metrics['sharpe_ratio']:.3f}")
        print(f"  Max drawdown:          {metrics['max_drawdown']*100:.2f}%")
        print(f"  Benchmark max drawdown:{metrics['benchmark_max_drawdown']*100:.2f}%")
        print(f"  Win rate:              {metrics['win_rate']*100:.1f}%")
        print(f"  Profit factor:         {metrics['profit_factor']:.3f}")
        print(f"  Saved equity curve to: {curve_path}")
        if plot_saved:
            print(f"  Saved plot to:         {plot_path}")
        print()


if __name__ == "__main__":
    main()
