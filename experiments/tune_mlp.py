import argparse
import copy
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yaml

from train_mlp import run_experiment

BASE_CONFIG = {
    "data": {
        "csv_path": "SPY_ohlcv.csv",
        "feature_cols": ["Open", "High", "Low", "Close", "Volume"],
        "price_col": "Close",
        "window_size": 20,
        "train_frac": 0.7,
        "val_frac": 0.15,
    },
    "model": {
        "num_classes": 3,
    },
    "training": {
        "batch_size": 64,
        "epochs": 30,
        "device": "auto",
    },
    "backtest": {
        "initial_capital": 10000,
        "modes": ["long_only", "long_short"],
    },
    "checkpoint": {
        "dir": "checkpoints",
    },
    "labeling": {
        "method": "direction_change",
    },
}

LR_VALUES      = [0.0001, 0.001]
HIDDEN_SIZES   = [[256, 128, 64, 32], [512, 256, 128, 64]]
DROPOUT_VALUES = [0.2, 0.3]
GAMMA_VALUES   = [1.0, 2.0, 5.0]


def build_grid(loss: str):
    combos = list(itertools.product(LR_VALUES, HIDDEN_SIZES, DROPOUT_VALUES))
    if loss == "focal":
        return [(lr, hs, do, g) for (lr, hs, do) in combos for g in GAMMA_VALUES]
    return [(lr, hs, do, None) for (lr, hs, do) in combos]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], required=True)
    args = parser.parse_args()

    loss = args.loss
    short = "cn" if loss == "cross_entropy" else "focal"
    grid = build_grid(loss)

    print(f"Tuning MLP  loss={loss}  features=ohlcv")
    print(f"Total trials: {len(grid)}\n")

    records = []
    for i, (lr, hs, dropout, gamma) in enumerate(grid):
        run_tag = f"tune_ohlcv_{short}_{i}"
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["model"]["hidden_sizes"] = hs
        cfg["model"]["dropout"] = dropout
        cfg["training"]["learning_rate"] = lr
        cfg["training"]["loss"] = loss
        if gamma is not None:
            cfg["training"]["focal_gamma"] = gamma

        desc = f"lr={lr}  hidden={hs}  dropout={dropout}"
        if gamma is not None:
            desc += f"  gamma={gamma}"
        print(f"\n{'='*64}")
        print(f"Trial {i+1}/{len(grid)}: {desc}")
        print("="*64)

        metrics = run_experiment(cfg, run_tag, save_predictions=False)
        records.append({
            "trial": i,
            "lr": lr,
            "hidden_sizes": str(hs),
            "dropout": dropout,
            "gamma": gamma,
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
            "best_epoch": metrics["best_epoch"],
        })
        print(f"  → val_loss={metrics['val_loss']:.4f}  val_acc={metrics['val_acc']:.4f}  best_epoch={metrics['best_epoch']}")

    results_df = pd.DataFrame(records)

    # select best: val_loss for CE (lower=better), val_acc for focal (higher=better)
    if loss == "cross_entropy":
        best_idx = results_df["val_loss"].idxmin()
        sort_col, ascending = "val_loss", True
    else:
        best_idx = results_df["val_acc"].idxmax()
        sort_col, ascending = "val_acc", False

    best = results_df.loc[best_idx]

    results_sorted = results_df.sort_values(sort_col, ascending=ascending)
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"tune_ohlcv_{short}_results.csv")
    results_sorted.to_csv(csv_path, index=False)

    print(f"\n{'='*64}")
    print("TOP 5 TRIALS:")
    print(results_sorted.head(5).to_string(index=False))

    print(f"\nBest: lr={best['lr']}  hidden={best['hidden_sizes']}  dropout={best['dropout']}", end="")
    if loss == "focal":
        print(f"  gamma={best['gamma']}", end="")
    print(f"\n  val_loss={best['val_loss']:.4f}  val_acc={best['val_acc']:.4f}")
    print(f"\nSaved all results → {csv_path}")

    # build and save best config
    cfg_best = copy.deepcopy(BASE_CONFIG)
    cfg_best["model"]["hidden_sizes"] = list(eval(best["hidden_sizes"]))
    cfg_best["model"]["dropout"] = float(best["dropout"])
    cfg_best["training"]["learning_rate"] = float(best["lr"])
    cfg_best["training"]["loss"] = loss
    if loss == "focal":
        cfg_best["training"]["focal_gamma"] = float(best["gamma"])
    cfg_best["training"]["device"] = "auto"

    config_path = f"configs/mlp_ohlcv_{short}_tuned.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg_best, f, default_flow_style=False, sort_keys=False)
    print(f"Saved best config → {config_path}")

    print(f"\nNext steps:")
    print(f"  python experiments/train_mlp.py --config {config_path}")
    print(f"  python experiments/backtest_mlp.py --config {config_path}")


if __name__ == "__main__":
    main()
