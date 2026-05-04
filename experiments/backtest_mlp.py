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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlp.yaml")
    args = parser.parse_args()

    run_tag = os.path.splitext(os.path.basename(args.config))[0]
    default_predictions = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"{run_tag}_test_predictions.csv"
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

    lines = []

    lines.append(f"Run: {run_tag}")
    lines.append(f"Predictions: {args.predictions}")
    lines.append(f"Samples: {len(preds_df)}")
    lines.append("")

    lines.append("Prediction distribution:")
    unique, counts = np.unique(raw_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        lines.append(f"  {label_names.get(cls, cls)}: {cnt} ({cnt/len(raw_preds)*100:.1f}%)")
    lines.append("")

    lines.append(classification_report(true_labels, raw_preds, target_names=target_names, digits=3))

    lines.append("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(true_labels, raw_preds, labels=[0, 1, 2])
    header = f"{'':>8}" + "".join(f"{n:>8}" for n in target_names)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"{target_names[i]:>8}" + "".join(f"{v:>8}" for v in row))
    lines.append("")

    # --------------------------------------------------------- backtest
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)

    for mode in bc["modes"]:
        bt = Backtester(backtest_prices, initial_capital=bc["initial_capital"], mode=mode)
        result = bt.run(signals)
        m = bt.metrics(result["portfolio_values"], benchmark_values=result["benchmark_values"])
        curve_dates = pred_dates
        curve_path = os.path.join(results_dir, f"{run_tag}_{mode}_backtest_curve.csv")
        save_equity_curve_csv(
            curve_path,
            curve_dates,
            result["portfolio_values"],
            result["benchmark_values"],
        )
        plot_path = os.path.join(results_dir, f"{run_tag}_{mode}_backtest_plot.png")
        plot_saved = False
        try:
            plot_equity_curves(
                plot_path,
                curve_dates,
                result["portfolio_values"],
                result["benchmark_values"],
                title=f"MLP Backtest vs Buy-and-Hold ({mode})",
                strategy_label="MLP Strategy",
                benchmark_label="Buy & Hold SPY",
                initial_capital=bc["initial_capital"],
            )
            plot_saved = True
        except ImportError as exc:
            lines.append(f"  Plot skipped:           {exc}")

        lines.append(f"--- Backtest: {mode} ---")
        lines.append(f"  Initial capital:       ${bc['initial_capital']:,.2f}")
        lines.append(f"  Strategy final value:  ${result['final_value']:,.2f}")
        lines.append(f"  Benchmark final value: ${result['benchmark_final_value']:,.2f}")
        lines.append(f"  Total return:          {m['total_return']*100:+.2f}%")
        lines.append(f"  Benchmark return:      {m['benchmark_return']*100:+.2f}%")
        lines.append(f"  Sharpe ratio:          {m['sharpe_ratio']:.3f}")
        lines.append(f"  Max drawdown:          {m['max_drawdown']*100:.2f}%")
        lines.append(f"  Benchmark max drawdown:{m['benchmark_max_drawdown']*100:.2f}%")
        lines.append(f"  Win rate:              {m['win_rate']*100:.1f}%")
        lines.append(f"  Profit factor:         {m['profit_factor']:.3f}")
        lines.append(f"  Saved equity curve to: {curve_path}")
        if plot_saved:
            lines.append(f"  Saved plot to:         {plot_path}")
        lines.append("")

    output = "\n".join(lines)
    print(output)

    results_path = os.path.join(results_dir, f"{run_tag}_backtest_results.txt")
    with open(results_path, "w") as f:
        f.write(output)
    print(f"Saved results → {results_path}")


if __name__ == "__main__":
    main()
