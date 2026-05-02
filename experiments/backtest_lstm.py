
    
# back testig scipt for LSTM models. to run, use the following command:
#       python experiments/backtest_lstm.py --config configs/lstm_base.yaml

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from backtest.plotting import plot_equity_curves, save_equity_curve_csv
from data.data_utils import load_ohlcv_csv


def load_config(path: str):

    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(portfolio_values, benchmark_values):
    # helper function to calculate metrics for backtest

    # convert to arrays
    values = np.asarray(portfolio_values, dtype = float)
    benchmark = np.asarray(benchmark_values, dtype = float)

    # get return from predictions
    total_return = (values[-1] - values[0]) / values[0]

    # get return from the benchmark
    benchmark_return = (benchmark[-1] - benchmark[0]) / benchmark[0]

    daily_returns = np.diff(values) / values[:-1]

    # calculate sharpe ratio
    if daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0


    # get peak and max drawdown for the predicitons
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min()

    # get peak and max drawdoen for the baseline
    benchmark_peak = np.maximum.accumulate(benchmark)
    benchmark_drawdowns = (benchmark - benchmark_peak) / benchmark_peak
    benchmark_max_drawdown = benchmark_drawdowns.min()

    # get wins/losses
    win = daily_returns > 0
    loss = daily_returns < 0

    winning_days = daily_returns[win]
    losing_days = daily_returns[loss]

    # calculate win rate
    if len(daily_returns) > 0:
        win_rate = len(winning_days) / len(daily_returns)
    else:
        win_rate = 0.0


    # calculate profit factor
    if losing_days.sum() != 0: # check for division by 0
        profit_factor = winning_days.sum() / (-losing_days.sum())
    else:
        profit_factor = float("inf")

    backtest_metrics = {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor
    }

    return backtest_metrics


def resolve_barrier_return(
        entry_price, open_price, high_price, low_price, upper_barrier, lower_barrier
):
    # helper function to get either the upper barrier or lower barrier
    upper_price = entry_price * (1.0 + upper_barrier)

    lower_price = entry_price * (1.0 + lower_barrier)

    hit_upper = high_price >= upper_price
    hit_lower = low_price <= lower_price

    # upper and lower barrier were hit. take one that is closest to the respective price
    if hit_upper and hit_lower: 

        dist_to_upper = abs(open_price - upper_price)
        dist_to_lower = abs(open_price - lower_price)

        if dist_to_upper < dist_to_lower:
            return upper_barrier
        else:
            return lower_barrier
        

    # only upper was hit
    if hit_upper:
        return upper_barrier
    
    # only lower was hit
    if hit_lower:
        return lower_barrier
    
    # neither lower or upper was hit
    return None


def run_triple_barrier_backtest(
    dates, opens, highs, lows, closes,
    signals, upper_barrier, lower_barrier, max_holding_period,
    initial_capital = 10000.0
):
    # util function for running the backtest

    # convert everything to arrays
    dates = np.asarray(dates)
    opens = np.asarray(opens, dtype = float)
    highs = np.asarray(highs, dtype = float)
    lows = np.asarray(lows, dtype = float)
    closes = np.asarray(closes, dtype = float)
    signals = np.asarray(signals, dtype = float)

    # populate initial values
    capital = float(initial_capital)
    entry_capital = capital

    position_size, trade_count = 0.0, 0
    entry_price, entry_day, exit_day = None, None, None


    portfolio_values = []
    for day in range(len(closes)):

        if entry_day is not None and day > entry_day:

            barrier_return = resolve_barrier_return(
                entry_price = entry_price,
                open_price = opens[day],
                high_price = highs[day],
                low_price = lows[day],
                upper_barrier = upper_barrier,
                lower_barrier = lower_barrier
            )


            mark_return = closes[day] / entry_price - 1.0

            if barrier_return is not None:
                mark_return = barrier_return

            portfolio_value = entry_capital * (1.0 + position_size * mark_return)

            portfolio_values.append(portfolio_value)

            if barrier_return is not None or day >= exit_day:

                capital = portfolio_value
                entry_capital = capital
                
                entry_price, entry_day, exit_day  = None, None, None

                position_size = 0.0

        else:
            portfolio_values.append(capital)

        if entry_day is None and day < len(signals) and abs(signals[day]) > 1e-12 and day < len(closes) - 1:
            trade_count += 1

            position_size = float(signals[day])

            entry_day = day

            exit_day = min(day + max_holding_period, len(closes) - 1)

            entry_price = closes[day]
            entry_capital = capital

    benchmark_values = initial_capital * (closes / closes[0])

    metrics = compute_metrics(
        portfolio_values, 
        benchmark_values
    )

    metrics["trade_count"] = trade_count

    # convert portfolio vals to an array
    array_portfolio_values = np.asarray(portfolio_values, dtype=float)

    backtest_results = {
        "dates": dates,
        "portfolio_values": array_portfolio_values,
        "benchmark_values": benchmark_values,
        "metrics": metrics,
    }

    return backtest_results


def format_metrics(lines, name: str, metrics: dict):

    lines.append(f"--- {name} ---")
    lines.append(f"  Total return:           {metrics['total_return']*100:+.2f}%")
    lines.append(f"  Benchmark return:       {metrics['benchmark_return']*100:+.2f}%")
    lines.append(f"  Sharpe ratio:           {metrics['sharpe_ratio']:.3f}")
    lines.append(f"  Max drawdown:           {metrics['max_drawdown']*100:.2f}%")
    lines.append(f"  Benchmark max drawdown: {metrics['benchmark_max_drawdown']*100:.2f}%")
    lines.append(f"  Win rate:               {metrics['win_rate']*100:.1f}%")
    lines.append(f"  Profit factor:          {metrics['profit_factor']:.3f}")
    lines.append(f"  Trades taken:           {metrics['trade_count']}")
    lines.append("")


def create_directories(project_root, run_tag):
    # create directpries to store experiments
    experiments_root = os.path.join(
        project_root, "results", "lstm_experiments"
    )

    run_dir = os.path.join(experiments_root, run_tag)
    
    return experiments_root, run_dir


def main():
    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        default = "configs/lstm_base.yaml"
    )
    args = parser.parse_args()

    run_tag = os.path.splitext(os.path.basename(args.config))[0]

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    directories = create_directories(project_root, run_tag)
    #experiments_root = directories[0]
    run_dir = directories[1]


    # get default predictions
    default_predictions = os.path.join(run_dir, "test_predictions.csv")
    args.predictions = default_predictions


    # load and extract data from configuration
    config = load_config(args.config)
    raw_data = config["data"]
    back_test = config["backtest"]
    training = config["training"]
    labeling = config.get("labeling", {})

    if not os.path.exists(args.predictions):
        print(f"Predictions file not found: {args.predictions}")
        print("Run train_lstm.py first to generate predictions.")
        sys.exit(1)

    preds_df = pd.read_csv(
        args.predictions, 
        index_col = 0, 
        parse_dates = True
    )
    print(f"Loaded predictions: {len(preds_df)} samples from {args.predictions}")

    # get column values
    raw_preds = preds_df["pred_label"].values
    true_labels = preds_df["true_label"].values
    pred_dates = preds_df.index
    df = load_ohlcv_csv(raw_data["csv_path"])
    price_col = raw_data["price_col"]

    # get date index
    all_dates = df.index

    # get start/end indexes based on max holding period and the number of lookahead dayes
    start_index = all_dates.get_loc(pred_dates[0])

    # max holding period
    max_holding_period = int(labeling.get("max_holding_period", raw_data.get("return_horizon", 10)))
    # lookahead days
    lookahead_days = max(max_holding_period, int(raw_data.get("return_horizon", 1)))

    # get end index
    end_index = min(start_index + len(pred_dates) + lookahead_days, len(all_dates))

    eval_df = df.iloc[start_index:end_index].copy()
    eval_dates = eval_df.index

    eval_opens = eval_df["Open"].values

    eval_highs = eval_df["High"].values

    eval_lows = eval_df["Low"].values

    eval_closes = eval_df[price_col].values

    # normalize labels to 0,1,2
    signal_map = {
        0: 0.0, 
        1: 1.0, 
        2: -1.0
    }

    signals = [signal_map[p] for p in raw_preds]
    class_signals = np.array(signals, dtype = float)

    # map normalized labels to names
    label_names = {
        0: "Hold",
        1: "Long", 
        2: "Short"
    }

    target_names = [label_names[i] for i in sorted(label_names)]

    report_dict = classification_report(
        true_labels,
        raw_preds,
        target_names = target_names, 
        digits = 3,
        output_dict = True
    )

    lines = []
    lines.append(f"Run: {run_tag}")
    lines.append(f"Predictions: {args.predictions}")
    lines.append(f"Samples: {len(preds_df)}")
    lines.append("")

    lines.append("Prediction distribution:")

    unique_counts = np.unique(
        raw_preds,
        return_counts = True
    )

    unique = unique_counts[0]
    counts = unique_counts[1]

    for cls, count in zip(unique, counts):

        count_pct = count/len(raw_preds)*100
        lines.append(f"  {label_names.get(cls, cls)}: {count} ({count_pct:.1f}%)")

        
    lines.append("")

    report_text = classification_report(
        true_labels,
        raw_preds,
        target_names = target_names,
        digits = 3
    )
    
    lines.append(report_text)

    lines.append("Confusion matrix (rows = true, cols = pred):")

    cm = confusion_matrix(
        true_labels, 
        raw_preds, 
        labels = [0, 1, 2]
    )


    header = f"{'':>8}" + "".join(f"{n:>8}" for n in target_names)
    lines.append(header)

    for i, row in enumerate(cm):
        lines.append(f"{target_names[i]:>8}" + "".join(f"{v:>8}" for v in row))

    lines.append("")

    os.makedirs(run_dir, exist_ok = True)

    upper_bar = float(labeling.get("upper_barrier", 0.03))
    lower_bar = float(labeling.get("lower_barrier", -0.03))

    tb_result = run_triple_barrier_backtest(
        dates = eval_dates,
        opens = eval_opens,
        highs = eval_highs,
        lows = eval_lows,
        closes = eval_closes,
        signals = class_signals,
        upper_barrier = upper_bar,
        lower_barrier = lower_bar,
        max_holding_period = max_holding_period,
        initial_capital = back_test["initial_capital"]
    )


    format_metrics(lines, "Backtest: Classifier / Triple Barrier", tb_result["metrics"])

    tb_plot_path = os.path.join(run_dir, "classifier_semantic_plot.png")
    tb_curve_path = os.path.join(run_dir, "classifier_semantic_curve.csv")

    save_equity_curve_csv(
        tb_curve_path,
        tb_result["dates"],
        tb_result["portfolio_values"],
        tb_result["benchmark_values"],
    )
    plot_equity_curves(
        tb_plot_path,
        tb_result["dates"],
        tb_result["portfolio_values"],
        tb_result["benchmark_values"],
        title = "LSTM Backtest vs Buy-and-Hold",
        strategy_label = "LSTM Strategy",
        benchmark_label = "Buy & Hold SPY",
        initial_capital = back_test["initial_capital"],
    )

    output = "\n".join(lines)
    print(output)

    results_path = os.path.join(run_dir, "backtest_results.txt")

    with open(results_path, "w") as f:
        f.write(output)

    print(f"Saved results -> {results_path}")

    summary = {
        "run_tag": run_tag,
        "loss": training["loss"],
        "samples": int(len(preds_df)),
        "classification": {
            "accuracy": float(report_dict["accuracy"]),
            "macro_f1": float(report_dict["macro avg"]["f1-score"]),
            "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
            "report": report_dict,
            "confusion_matrix": cm.tolist(),
        },
        "backtest": tb_result["metrics"],
    }

    summary_path = os.path.join(run_dir, "backtest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            summary, f, indent = 2
        )

    print(f"Saved summary -> {summary_path}")



if __name__ == "__main__":
    main()

